from __future__ import annotations

import asyncio
import gc
import inspect
import ipaddress
import os
import socket
import sys
from collections.abc import AsyncIterator
from typing import Any

import ray
import torch
import torch.distributed as dist
from vllm import SamplingParams
from vllm.engine.arg_utils import AsyncEngineArgs
try:
    from vllm.inputs import TokensPrompt
except ImportError:  # pragma: no cover - older vLLM compatibility
    TokensPrompt = None
try:
    from vllm.sampling_params import RequestOutputKind
except ImportError:  # pragma: no cover - older vLLM compatibility
    RequestOutputKind = None
from vllm.usage.usage_lib import UsageContext
from vllm.v1.engine.async_llm import AsyncLLM

from nanoverl.config import RolloutConfig
from nanoverl.data import TokenOutput
from nanoverl.sync.nccl_sync_engine import NCCLSyncEngine
from nanoverl.sync.weight_payload import (
    align_offset as _align_offset,
    bucket_from_weight_payload as _bucket_from_weight_payload,
    copy_tensor_to_bytes as _copy_tensor_to_bytes,
    cuda_ipc_payload as _cuda_ipc_payload,
    normalize_weight_transport as _normalize_weight_transport,
    shared_memory_payload as _shared_memory_payload,
    weight_entry as _weight_entry,
)


def _reserve_free_port(host: str) -> tuple[int, socket.socket]:
    try:
        family = socket.AF_INET6 if isinstance(ipaddress.ip_address(host), ipaddress.IPv6Address) else socket.AF_INET
    except ValueError:
        family = socket.AF_INET

    sock = socket.socket(family, socket.SOCK_STREAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    sock.bind((host, 0))
    return int(sock.getsockname()[1]), sock


def _cleanup_cuda_runtime() -> None:
    try:
        if dist.is_available() and dist.is_initialized():
            dist.destroy_process_group()
    except Exception:
        pass

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


async def _aiter_weights(weight_stream: Any) -> AsyncIterator[tuple[str, torch.Tensor]]:
    if isinstance(weight_stream, dict):
        weight_stream = weight_stream.items()

    if hasattr(weight_stream, "__aiter__"):
        async for item in weight_stream:
            yield item
    else:
        for item in weight_stream:
            yield item


def _chosen_token_logprobs(token_ids: list[int], sample_logprobs: Any) -> list[float] | None:
    if sample_logprobs is None:
        return None

    start_indices = getattr(sample_logprobs, "start_indices", None)
    end_indices = getattr(sample_logprobs, "end_indices", None)
    flat_token_ids = getattr(sample_logprobs, "token_ids", None)
    flat_logprobs = getattr(sample_logprobs, "logprobs", None)
    if (
        start_indices is not None
        and end_indices is not None
        and flat_token_ids is not None
        and flat_logprobs is not None
    ):
        if len(start_indices) < len(token_ids) or len(end_indices) < len(token_ids):
            return None
        values: list[float] = []
        for position, token_id in enumerate(token_ids):
            start = int(start_indices[position])
            end = int(end_indices[position])
            if end <= start:
                return None
            if end == start + 1:
                if int(flat_token_ids[start]) != int(token_id):
                    return None
                values.append(float(flat_logprobs[start]))
                continue
            for index in range(start, end):
                if int(flat_token_ids[index]) == int(token_id):
                    values.append(float(flat_logprobs[index]))
                    break
            else:
                return None
        return values

    values: list[float] = []
    for token_id, token_logprobs in zip(token_ids, sample_logprobs, strict=False):
        if not isinstance(token_logprobs, dict):
            return None
        entry = token_logprobs.get(token_id, token_logprobs.get(str(token_id)))
        if entry is None:
            return None
        values.append(float(getattr(entry, "logprob", entry)))
    return values


def _require_cuda_device() -> torch.device:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for vLLM CUDA IPC weight sync")
    return torch.device("cuda", torch.cuda.current_device())


def _clear_cuda_ipc_cache() -> None:
    gc.collect()
    if not torch.cuda.is_available():
        return
    torch.cuda.synchronize()
    torch.cuda.ipc_collect()
    torch.cuda.empty_cache()


def _maybe_clear_cuda_ipc_cache(payload: dict[str, Any]) -> None:
    if not bool(payload.get("defer_cache_clear", True)):
        _clear_cuda_ipc_cache()


def load_weight_bucket_on_vllm_worker(worker: Any, payload: dict[str, Any]) -> None:
    device = getattr(worker, "device", None)
    if device is None and torch.cuda.is_available():
        device = torch.device("cuda", torch.cuda.current_device())
    weights: list[tuple[str, torch.Tensor]] = []
    tensor = None
    bucket = None
    shm = None
    try:
        bucket, shm = _bucket_from_weight_payload(payload, device)
        for entry in payload["entries"]:
            offset = int(entry["offset"])
            nbytes = int(entry["nbytes"])
            dtype = entry["dtype"]
            shape = tuple(int(dim) for dim in entry["shape"])
            tensor = bucket.narrow(0, offset, nbytes).view(dtype).view(shape)
            if payload["transport"] == "shared_memory" and device is not None:
                tensor = tensor.to(device=device, non_blocking=False)
            elif payload["transport"] == "cuda_ipc":
                tensor = tensor.clone()
            weights.append((str(entry["name"]), tensor.contiguous()))
        worker.model_runner.model.load_weights(weights)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
    finally:
        del weights, tensor, bucket
        if shm is not None:
            shm.close()
        _maybe_clear_cuda_ipc_cache(payload)


def finish_weight_load_on_vllm_worker(worker: Any) -> None:
    _clear_cuda_ipc_cache()


def reset_weight_load_on_vllm_worker(worker: Any) -> None:
    _clear_cuda_ipc_cache()


def _vllm_common_options(
    config: RolloutConfig,
    *,
    nnodes: int,
    node_rank: int,
    master_addr: str,
    master_port: int,
) -> dict[str, Any]:
    options: dict[str, Any] = {
        "model": config.model_path,
        "served_model_name": config.model_path,
        "dtype": config.dtype,
        "max_model_len": int(config.max_model_len),
        "tensor_parallel_size": int(config.tensor_parallel_size),
        "max_num_seqs": int(config.max_num_seqs),
        "max_num_batched_tokens": int(config.max_num_batched_tokens),
        "enable_chunked_prefill": bool(config.enable_chunked_prefill),
        "enable_prefix_caching": bool(config.enable_prefix_caching),
        "logprobs_mode": str(config.logprobs_mode),
        "distributed_executor_backend": "mp",
        "nnodes": int(nnodes),
        "node_rank": int(node_rank),
        "master_addr": str(master_addr),
        "master_port": int(master_port),
        "seed": int(config.seed),
    }
    if config.gpu_memory_utilization is not None:
        options["gpu_memory_utilization"] = float(config.gpu_memory_utilization)
    return options


def _build_engine_args(
    config: RolloutConfig,
    *,
    nnodes: int,
    node_rank: int,
    master_addr: str,
    master_port: int,
) -> AsyncEngineArgs:
    kwargs = {
        **_vllm_common_options(
            config,
            nnodes=nnodes,
            node_rank=node_rank,
            master_addr=master_addr,
            master_port=master_port,
        ),
        "served_model_name": [config.model_path],
        "trust_remote_code": bool(config.trust_remote_code),
        "disable_log_stats": True,
        "enable_log_requests": False,
        "enable_sleep_mode": bool(config.free_cache_engine),
    }

    if int(nnodes) > 1:
        kwargs["enforce_eager"] = True
    else:
        for key in ("nnodes", "node_rank", "master_addr", "master_port"):
            kwargs.pop(key, None)

    return AsyncEngineArgs(**kwargs)


def _build_async_llm(
    config: RolloutConfig,
    *,
    nnodes: int,
    node_rank: int,
    master_addr: str,
    master_port: int,
) -> AsyncLLM:
    return AsyncLLM.from_engine_args(
        _build_engine_args(
            config,
            nnodes=nnodes,
            node_rank=node_rank,
            master_addr=master_addr,
            master_port=master_port,
        ),
        usage_context=UsageContext.OPENAI_API_SERVER,
    )


def _build_sync_engine(config: RolloutConfig) -> Any | None:
    mode = str(config.mode).lower().strip()

    if mode == "hybrid":
        return None

    if mode == "standalone":
        return NCCLSyncEngine(
            bucket_size_mb=int(config.sync_bucket_size_mb),
            group_name=config.sync_group_name,
            rebuild_group=bool(config.sync_rebuild_group),
        )

    raise ValueError(f"unsupported rollout.mode for VLLMServer sync: {config.mode}")


@ray.remote
class VLLMServer:
    """A complete single-node vLLM rollout server."""

    def __init__(
        self,
        config: RolloutConfig,
        server_id: str,
        node_rank: int,
        nnodes: int,
        visible_devices: str,
        server_rank: int = 0,
    ) -> None:
        self.config = config
        self.server_id = str(server_id)
        self.server_rank = int(server_rank)
        self.node_rank = int(node_rank)
        self.nnodes = int(nnodes)
        self.visible_devices = visible_devices
        if self.node_rank != 0 or self.nnodes != 1:
            raise ValueError(
                "VLLMServer now represents one complete single-node rollout server; "
                "start multiple VLLMServer actors for multi-node rollout data parallelism"
            )

        self.host = ray.util.get_node_ip_address().strip("[]")
        self.master_addr: str | None = self.host
        self.master_port: int | None = None
        self._master_sock: socket.socket | None = None
        self.master_port, self._master_sock = _reserve_free_port(self.host)

        self.model_version = -1
        self._started = False
        self._runtime_configured = False
        self._start_lock: asyncio.Lock | None = None

        self.engine: AsyncLLM | None = None
        self.error: BaseException | None = None
        self.sync_engine: NCCLSyncEngine | None = None
        self._weight_sync_bucket: torch.Tensor | None = None
        self._weights_awake_during_sleep = False

    def _master_endpoint(self) -> tuple[str, int]:
        assert self.master_addr is not None
        assert self.master_port is not None
        return self.master_addr, self.master_port

    def _leader_engine(self) -> AsyncLLM:
        assert self.engine is not None
        return self.engine

    def _configure_runtime(self) -> None:
        if self._runtime_configured:
            return

        runtime_defaults = {
            "NCCL_CUMEM_ENABLE": "0",
            "NCCL_IB_DISABLE": "1",
            "NCCL_NET_PLUGIN": "",
            "NCCL_SOCKET_FAMILY": "AF_INET",
            "NCCL_SOCKET_IFNAME": "eth0",
            "NCCL_TUNER_PLUGIN": "",
            "NCCL_FASTRAK_ENABLE_CONTROL_CHANNEL": "0",
            "VLLM_ALLREDUCE_USE_SYMM_MEM": "0",
            "VLLM_DISABLE_COMPILE_CACHE": "1",
            "VLLM_ALLOW_INSECURE_SERIALIZATION": "1",
        }
        for name, value in runtime_defaults.items():
            os.environ[name] = value

        if self.visible_devices:
            os.environ["CUDA_VISIBLE_DEVICES"] = self.visible_devices

        venv_bin = os.path.dirname(sys.executable)
        current_path = os.environ.get("PATH", "")
        if venv_bin and venv_bin not in current_path.split(os.pathsep):
            os.environ["PATH"] = os.pathsep.join(part for part in [venv_bin, current_path] if part)

        self._runtime_configured = True

    def _release_master_port(self) -> None:
        if self._master_sock is not None:
            self._master_sock.close()
            self._master_sock = None

    def _ensure_sync_engine(self) -> NCCLSyncEngine | None:
        if self.sync_engine is None:
            self.sync_engine = _build_sync_engine(self.config)
        return self.sync_engine

    def _set_master(self, master_addr: str | None, master_port: int | None) -> None:
        if master_addr is not None:
            self.master_addr = str(master_addr)
        if master_port is not None:
            self.master_port = int(master_port)

    async def _ensure_started(self) -> None:
        if not self._started:
            await self.start()

    async def _start_leader(self) -> None:
        if self._master_sock is None:
            self.master_port, self._master_sock = _reserve_free_port(self.master_addr or self.host)
        master_addr, master_port = self._master_endpoint()
        self._release_master_port()
        self.engine = _build_async_llm(
            self.config,
            nnodes=self.nnodes,
            node_rank=self.node_rank,
            master_addr=master_addr,
            master_port=master_port,
        )

    async def start(
        self,
        master_addr: str | None = None,
        master_port: int | None = None,
    ) -> dict[str, Any]:
        if self._start_lock is None:
            self._start_lock = asyncio.Lock()

        async with self._start_lock:
            self._set_master(master_addr, master_port)

            if self._started:
                return await self.get_bootstrap_info()

            self._configure_runtime()

            try:
                await self._start_leader()
                self.error = None
                self._started = True
            except BaseException as exc:
                self.error = exc
                await self._cleanup_runtime(finalize_sync=False)
                raise

        return await self.get_bootstrap_info()

    async def generate(
        self,
        request_id: str,
        prompt_ids: list[int],
        sampling_params: dict[str, Any],
    ) -> TokenOutput | list[TokenOutput]:
        await self._ensure_started()

        engine = self._leader_engine()
        sampling_params = dict(sampling_params)
        if RequestOutputKind is not None:
            sampling_params.setdefault("output_kind", RequestOutputKind.FINAL_ONLY)
        if sampling_params.get("logprobs") is not None:
            sampling_params.setdefault("flat_logprobs", True)
        prompt: Any = {"prompt_token_ids": list(prompt_ids)}
        if TokensPrompt is not None:
            prompt = TokensPrompt(**prompt)
        final_output = None
        async for output in engine.generate(
            prompt=prompt,
            sampling_params=SamplingParams(**sampling_params),
            request_id=str(request_id),
        ):
            final_output = output

        if final_output is None or not final_output.outputs:
            raise RuntimeError("AsyncLLM returned no output")

        outputs: list[TokenOutput] = []
        for sample in final_output.outputs:
            token_ids = list(getattr(sample, "token_ids", []) or [])
            log_probs = _chosen_token_logprobs(token_ids, getattr(sample, "logprobs", None))
            outputs.append(
                TokenOutput(
                    token_ids=token_ids,
                    log_probs=log_probs,
                    stop_reason=getattr(sample, "finish_reason", None) or getattr(sample, "stop_reason", None),
                    model_version=self.model_version,
                )
            )

        return outputs[0] if len(outputs) == 1 else outputs

    async def wait_for_requests_to_drain(self) -> None:
        engine = self._leader_engine()
        await engine.wait_for_requests_to_drain()

    async def abort_all_requests(self, reset_prefix_cache: bool = True) -> None:
        await self._ensure_started()

        engine = self._leader_engine()

        pause_generation = getattr(engine, "pause_generation", None)

        if pause_generation is not None:
            await pause_generation(
                wait_for_inflight_requests=False,
                clear_cache=reset_prefix_cache,
            )
            return

        output_processor = getattr(engine, "output_processor", None)
        request_states = dict(getattr(output_processor, "request_states", {}) or {})
        request_ids = list(request_states.keys())
        if not request_ids:
            return

        try:
            from vllm.v1.engine import FinishReason

            for req_state in request_states.values():
                request_output = req_state.make_request_output(
                    [],
                    pooling_output=None,
                    finish_reason=FinishReason.ABORT,
                    stop_reason=None,
                )
                req_state.queue.put(request_output)
        except Exception:
            pass

        if output_processor is not None and hasattr(output_processor, "abort_requests"):
            output_processor.abort_requests(request_ids)

        engine_core = getattr(engine, "engine_core", None)
        if engine_core is not None and hasattr(engine_core, "abort_requests_async"):
            await engine_core.abort_requests_async(request_ids)
        elif hasattr(engine, "abort"):
            for request_id in request_ids:
                result = engine.abort(request_id)
                if inspect.isawaitable(result):
                    await result

        await self.wait_for_requests_to_drain()

        if reset_prefix_cache or self.config.reset_prefix_cache_on_abort:
            await self.clear_cache()

    async def resume_generation(self) -> None:
        if self.engine is None:
            return

        resume_generation = getattr(self.engine, "resume_generation", None)
        if resume_generation is not None:
            await resume_generation()

    async def sleep(self) -> None:
        if not self._started:
            return

        if self.config.free_cache_engine:
            await self.wait_for_requests_to_drain()
            engine = self._leader_engine()
            is_sleeping = getattr(engine, "is_sleeping", None)
            if is_sleeping is not None and await is_sleeping():
                return
            await engine.sleep(level=int(self.config.sleep_level))
            self._weights_awake_during_sleep = False

    async def wake_up(self) -> None:
        await self._ensure_started()

        if self.config.free_cache_engine:
            engine = self._leader_engine()
            tags = ["kv_cache"] if self._weights_awake_during_sleep else None
            await engine.wake_up(tags=tags)
            self._weights_awake_during_sleep = False

        await self.clear_cache()

    async def clear_cache(self) -> None:
        await self._ensure_started()

        engine = self._leader_engine()

        await engine.reset_prefix_cache()

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    async def begin_weight_update(self, version: int) -> None:
        _ = version
        await self._ensure_started()

        engine = self._leader_engine()

        await self.wait_for_requests_to_drain()
        await engine.wake_up(tags=["weights"])
        self._weights_awake_during_sleep = True

    async def load_weight_bucket(self, payload: dict[str, Any]) -> None:
        await self._ensure_started()

        engine = self._leader_engine()

        try:
            await engine.collective_rpc(
                load_weight_bucket_on_vllm_worker,
                args=(payload,),
            )
        except Exception:
            await engine.collective_rpc(reset_weight_load_on_vllm_worker)
            raise

    async def finish_weight_update(self, version: int) -> None:
        await self._ensure_started()

        engine = self._leader_engine()
        await engine.collective_rpc(finish_weight_load_on_vllm_worker)
        await self.clear_cache()
        self.model_version = int(version)

    async def reset_weight_update(self) -> None:
        await self._ensure_started()

        engine = self._leader_engine()
        await engine.collective_rpc(reset_weight_load_on_vllm_worker)

    def _get_weight_sync_bucket(self, device: torch.device, nbytes: int) -> torch.Tensor:
        bucket = self._weight_sync_bucket
        if bucket is None or bucket.device != device or int(bucket.numel()) < int(nbytes):
            self._weight_sync_bucket = torch.empty(int(nbytes), dtype=torch.uint8, device=device)
        return self._weight_sync_bucket

    async def _reload_weights(self, weight_stream: Any) -> None:
        engine = self._leader_engine()
        assert engine is not None

        await engine.wake_up(tags=["weights"])
        self._weights_awake_during_sleep = True

        try:
            device = _require_cuda_device()
            bucket_size = int(self.config.weight_sync_bucket_mb) << 20
            bucket = self._get_weight_sync_bucket(device, bucket_size)
            entries: list[dict[str, Any]] = []
            offset = 0

            async def send_cuda_ipc_bucket(
                payload: torch.Tensor,
                payload_entries: tuple[dict[str, Any], ...],
            ) -> None:
                bucket_payload = _cuda_ipc_payload(payload, payload_entries)
                bucket_payload["defer_cache_clear"] = bool(self.config.weight_sync_defer_cache_clear)
                await engine.collective_rpc(
                    load_weight_bucket_on_vllm_worker,
                    args=(bucket_payload,),
                )

            async def send_shared_memory_bucket(
                payload: torch.Tensor,
                payload_entries: tuple[dict[str, Any], ...],
            ) -> None:
                bucket_payload, shm = _shared_memory_payload(payload, payload_entries)
                bucket_payload["defer_cache_clear"] = bool(self.config.weight_sync_defer_cache_clear)
                try:
                    await engine.collective_rpc(
                        load_weight_bucket_on_vllm_worker,
                        args=(bucket_payload,),
                    )
                finally:
                    shm.close()
                    shm.unlink()

            async def send_bucket(payload: torch.Tensor, payload_entries: tuple[dict[str, Any], ...]) -> None:
                torch.cuda.synchronize(device)
                transport = _normalize_weight_transport(self.config.weight_sync_transport)
                if transport == "shared_memory":
                    await send_shared_memory_bucket(payload, payload_entries)
                    return
                if transport == "cuda_ipc":
                    await send_cuda_ipc_bucket(payload, payload_entries)
                    return
                try:
                    await send_cuda_ipc_bucket(payload, payload_entries)
                except Exception:
                    await engine.collective_rpc(reset_weight_load_on_vllm_worker)
                    await send_shared_memory_bucket(payload, payload_entries)

            async def flush_bucket() -> None:
                nonlocal offset
                if not entries:
                    return
                await send_bucket(bucket, tuple(entries))
                entries.clear()
                offset = 0

            async for name, tensor in _aiter_weights(weight_stream):
                if not isinstance(tensor, torch.Tensor):
                    raise TypeError(f"expected tensor payload, got {type(tensor).__name__}")
                nbytes = int(tensor.numel()) * int(tensor.element_size())

                if nbytes > bucket_size:
                    await flush_bucket()
                    payload = self._get_weight_sync_bucket(device, nbytes).narrow(0, 0, nbytes)
                    _copy_tensor_to_bytes(payload, tensor)
                    await send_bucket(payload, (_weight_entry(name, tensor, 0),))
                    continue

                aligned_offset = _align_offset(offset, tensor.element_size())
                if aligned_offset + nbytes > bucket_size:
                    await flush_bucket()
                    aligned_offset = 0

                _copy_tensor_to_bytes(bucket.narrow(0, aligned_offset, nbytes), tensor)
                entries.append(_weight_entry(name, tensor, aligned_offset))
                offset = aligned_offset + nbytes

            await flush_bucket()

            await engine.collective_rpc(finish_weight_load_on_vllm_worker)
        except Exception:
            await engine.collective_rpc(reset_weight_load_on_vllm_worker)
            raise

    async def update_weights_from_sync_engine(self, version: int) -> None:
        self._configure_runtime()
        sync_engine = self._ensure_sync_engine()
        if sync_engine is None:
            raise RuntimeError("update_weights_from_sync_engine only supports rollout.mode='standalone'")

        await self.update_weights(sync_engine.receive_weights(), version)

    async def update_weights(self, weight_stream: Any, version: int) -> None:
        await self._ensure_started()

        await self.wait_for_requests_to_drain()
        await self._reload_weights(weight_stream)
        self.model_version = int(version)
        await self.clear_cache()

    def execute_sync_engine(self, method: str, *args, **kwargs):
        if method not in {"prepare", "init_process_group", "finalize"}:
            raise ValueError(f"unsupported sync engine method from manager: {method}")

        if method in {"prepare", "init_process_group"}:
            self._configure_runtime()

        sync_engine = self._ensure_sync_engine()
        if sync_engine is None:
            raise RuntimeError("this VLLMServer has no sync_engine; rollout.mode is probably hybrid")

        return getattr(sync_engine, method)(*args, **kwargs)

    async def _cleanup_runtime(self, *, finalize_sync: bool = False) -> None:
        self._release_master_port()

        if self.engine is not None:
            result = self.engine.shutdown()
            if inspect.isawaitable(result):
                await result
            self.engine = None

        if finalize_sync and self.sync_engine is not None:
            self.sync_engine.finalize()
            self.sync_engine = None

        _cleanup_cuda_runtime()

        self._started = False
        self._runtime_configured = False

    async def get_bootstrap_info(self) -> dict[str, Any]:
        return {
            "server_id": self.server_id,
            "server_rank": self.server_rank,
            "node_rank": self.node_rank,
            "master_addr": self.master_addr,
            "master_port": self.master_port,
            "model": self.config.model_path,
            "model_version": self.model_version,
        }

    async def get_info(self) -> dict[str, Any]:
        return {
            "server_id": self.server_id,
            "server_rank": self.server_rank,
            "node_rank": self.node_rank,
            "nnodes": self.nnodes,
            "model_version": self.model_version,
            "max_num_seqs": int(self.config.max_num_seqs),
            "max_num_batched_tokens": int(self.config.max_num_batched_tokens),
        }

    async def shutdown(self) -> dict[str, Any]:
        await self._cleanup_runtime(finalize_sync=True)
        return {
            "status": "ok",
            "server_id": self.server_id,
            "server_rank": self.server_rank,
            "node_rank": self.node_rank,
        }
