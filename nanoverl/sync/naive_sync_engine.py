from __future__ import annotations

import asyncio
import ctypes
import gc
import inspect
import os
from collections.abc import Iterable
from typing import Any

import ray
import torch

from nanoverl.sync.base_engine import BaseSyncEngine
from nanoverl.sync.weight_payload import (
    align_offset,
    copy_tensor_to_bytes,
    cuda_ipc_payload,
    normalize_weight_transport,
    shared_memory_payload,
    weight_entry,
)


def trim_cpu_memory() -> None:
    gc.collect()
    if os.name != "posix":
        return
    try:
        malloc_trim = ctypes.CDLL("libc.so.6").malloc_trim
    except Exception:
        return
    malloc_trim(0)


class NaiveSyncEngine(BaseSyncEngine):
    """Colocated trainer/rollout bucketed weight sync backend.

    Ray carries only control calls and bucket metadata. Weight bytes are exposed
    through CUDA IPC first, with shared-memory fallback when configured.
    """

    def __init__(self, bucket_size_mb: int = 16, transport: str = "auto") -> None:
        if bucket_size_mb <= 0:
            raise ValueError(f"bucket_size_mb must be positive, got {bucket_size_mb}")
        self.bucket_size_mb = int(bucket_size_mb)
        self.bucket_size_bytes = self.bucket_size_mb * 1024 * 1024
        self.transport = normalize_weight_transport(transport)
        self.bucket: torch.Tensor | None = None

    async def send_weights(
        self,
        weight_stream: Iterable[tuple[str, torch.Tensor]],
        rollout_handle: Any | None = None,
        version: int | None = None,
    ) -> None:
        if rollout_handle is None:
            raise ValueError("NaiveSyncEngine requires rollout_handle")
        if version is None:
            raise ValueError("NaiveSyncEngine requires version")
        for method in ("begin_weight_update", "load_weight_bucket", "finish_weight_update"):
            if not hasattr(rollout_handle, method):
                raise AttributeError(f"rollout_handle must define {method}")

        await self._call(rollout_handle, "begin_weight_update", version)
        entries: list[dict[str, Any]] = []
        offset = 0

        try:
            async def flush_bucket() -> None:
                nonlocal offset
                if not entries:
                    return
                assert self.bucket is not None
                await self._send_bucket(rollout_handle, self.bucket, tuple(entries))
                entries.clear()
                offset = 0

            async for name, tensor in self._aiter_weights(weight_stream):
                if not isinstance(tensor, torch.Tensor):
                    raise TypeError(f"expected tensor payload, got {type(tensor).__name__}")
                tensor = tensor.detach().contiguous()
                nbytes = int(tensor.numel()) * int(tensor.element_size())

                if nbytes > self.bucket_size_bytes:
                    await flush_bucket()
                    payload = torch.empty(nbytes, dtype=torch.uint8, device=tensor.device)
                    copy_tensor_to_bytes(payload, tensor)
                    await self._send_bucket(rollout_handle, payload, (weight_entry(name, tensor, 0),))
                    continue

                bucket = self._get_bucket(tensor.device, self.bucket_size_bytes)
                aligned_offset = align_offset(offset, tensor.element_size())
                if entries and aligned_offset + nbytes > self.bucket_size_bytes:
                    await flush_bucket()
                    bucket = self._get_bucket(tensor.device, self.bucket_size_bytes)
                    aligned_offset = 0

                copy_tensor_to_bytes(bucket.narrow(0, aligned_offset, nbytes), tensor)
                entries.append(weight_entry(name, tensor, aligned_offset))
                offset = aligned_offset + nbytes

            await flush_bucket()
            await self._call(rollout_handle, "finish_weight_update", version)
        except Exception:
            await self._reset_if_supported(rollout_handle)
            raise
        finally:
            self.bucket = None
            self._release_cuda_ipc_refs()

    def _get_bucket(self, device: torch.device, nbytes: int) -> torch.Tensor:
        bucket = self.bucket
        if bucket is None or bucket.device != device or int(bucket.numel()) < int(nbytes):
            self.bucket = torch.empty(int(nbytes), dtype=torch.uint8, device=device)
        return self.bucket

    async def _send_bucket(
        self,
        rollout_handle: Any,
        bucket: torch.Tensor,
        entries: tuple[dict[str, Any], ...],
    ) -> None:
        if bucket.is_cuda:
            torch.cuda.synchronize(bucket.device)

        if self.transport == "shared_memory":
            await self._send_shared_memory_bucket(rollout_handle, bucket, entries)
            return
        if self.transport == "cuda_ipc":
            await self._send_cuda_ipc_bucket(rollout_handle, bucket, entries)
            return
        if not bucket.is_cuda:
            await self._send_shared_memory_bucket(rollout_handle, bucket, entries)
            return

        try:
            await self._send_cuda_ipc_bucket(rollout_handle, bucket, entries)
        except Exception:
            await self._reset_if_supported(rollout_handle)
            await self._send_shared_memory_bucket(rollout_handle, bucket, entries)

    async def _send_cuda_ipc_bucket(
        self,
        rollout_handle: Any,
        bucket: torch.Tensor,
        entries: tuple[dict[str, Any], ...],
    ) -> None:
        await self._call(rollout_handle, "load_weight_bucket", cuda_ipc_payload(bucket, entries))

    async def _send_shared_memory_bucket(
        self,
        rollout_handle: Any,
        bucket: torch.Tensor,
        entries: tuple[dict[str, Any], ...],
    ) -> None:
        payload, shm = shared_memory_payload(bucket, entries)
        try:
            await self._call(rollout_handle, "load_weight_bucket", payload)
        finally:
            shm.close()
            shm.unlink()

    async def _reset_if_supported(self, rollout_handle: Any) -> None:
        if hasattr(rollout_handle, "reset_weight_update"):
            await self._call(rollout_handle, "reset_weight_update")

    def _release_cuda_ipc_refs(self) -> None:
        trim_cpu_memory()
        if not torch.cuda.is_available():
            return
        torch.cuda.synchronize()
        torch.cuda.ipc_collect()
        torch.cuda.empty_cache()

    @staticmethod
    async def _aiter_weights(weight_stream: Any):
        if isinstance(weight_stream, dict):
            weight_stream = weight_stream.items()
        if hasattr(weight_stream, "__aiter__"):
            async for item in weight_stream:
                yield item
        else:
            for item in weight_stream:
                yield item

    @staticmethod
    async def _call(target: Any, method: str, *args: Any, **kwargs: Any) -> Any:
        fn = getattr(target, method)
        result = fn.remote(*args, **kwargs) if hasattr(fn, "remote") else fn(*args, **kwargs)
        if inspect.isawaitable(result):
            return await result
        if isinstance(result, ray.ObjectRef):
            return await asyncio.to_thread(ray.get, result)
        return result
