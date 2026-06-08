from __future__ import annotations

import asyncio
import ipaddress
import os
import time
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any, AsyncGenerator, Iterator, TypedDict

import ray.util.collective as collective
import torch
import zmq
from ray.util import get_node_ip_address as _get_node_ip_address

from nanoverl.sync.base_engine import BaseSyncEngine


class TensorMeta(TypedDict):
    name: str
    shape: torch.Size
    dtype: torch.dtype
    offset: int
    nbytes: int


@dataclass(frozen=True)
class MasterMetadata:
    zmq_ip: str
    zmq_port: int


def _is_valid_ipv6_address(address: str) -> bool:
    try:
        ipaddress.IPv6Address(address)
        return True
    except ValueError:
        return False


def _zmq_bind_host(ip_address: str) -> str:
    if _is_valid_ipv6_address(ip_address):
        return f"tcp://[{ip_address}]"
    return f"tcp://{ip_address}"


def _zmq_connect_address(metadata: MasterMetadata) -> str:
    if _is_valid_ipv6_address(metadata.zmq_ip):
        return f"tcp://[{metadata.zmq_ip}]:{metadata.zmq_port}"
    return f"tcp://{metadata.zmq_ip}:{metadata.zmq_port}"


class NCCLSyncEngine(BaseSyncEngine):
    """
    NCCL sync backend.

    Metadata goes through ZeroMQ PUB/SUB. Packed weight buckets go through NCCL broadcast.
    """

    METADATA_TOPIC = "bucket_metadata"
    SUBSCRIBE_TIMEOUT_ENV = "NANOVERL_ZMQ_SUBSCRIBE_TIMEOUT_S"

    def __init__(
        self,
        bucket_size_mb: int,
        group_name: str = "default",
        rebuild_group: bool = False,
    ) -> None:
        if bucket_size_mb <= 0:
            raise ValueError(f"bucket_size_mb must be positive, got {bucket_size_mb}")
        if not torch.cuda.is_available():
            raise RuntimeError("NCCLSyncEngine requires CUDA because backend='nccl'")

        self.bucket_size_mb = int(bucket_size_mb)
        self.bucket_size_bytes = self.bucket_size_mb * 1024 * 1024
        self.group_name = group_name
        self.rebuild_group = bool(rebuild_group)
        self.device = torch.device("cuda", torch.cuda.current_device())

        self.rank: int | None = None
        self.world_size: int | None = None
        self.send_buf: torch.Tensor | None = None
        self.recv_buf: torch.Tensor | None = None
        self.master_metadata: MasterMetadata | None = None
        self.metadata_socket: Any | None = None
        self.metadata_context: zmq.Context | None = None

        self._collective = collective
        self._group_initialized = False

    def prepare(self, create_master_metadata: bool = False) -> Any:
        self.send_buf = torch.zeros(self.bucket_size_bytes, dtype=torch.uint8, device=self.device)
        self.recv_buf = torch.zeros_like(self.send_buf)
        if create_master_metadata:
            self.master_metadata = self._start_metadata_publisher()
        return self.master_metadata

    @classmethod
    def build_topology(
        cls,
        trainer_world_size: int,
        rollout_world_size: int,
        master_metadata: Any = None,
    ) -> tuple[dict[str, list[Any]], dict[str, list[Any]]]:
        if trainer_world_size <= 0:
            raise ValueError(f"trainer_world_size must be positive, got {trainer_world_size}")
        if rollout_world_size <= 0:
            raise ValueError(f"rollout_world_size must be positive, got {rollout_world_size}")

        world_size = rollout_world_size + 1
        return (
            {
                "rank": [0] + [-1] * (trainer_world_size - 1),
                "world_size": [world_size] * trainer_world_size,
                "master_metadata": [master_metadata] + [None] * (trainer_world_size - 1),
            },
            {
                "rank": list(range(1, rollout_world_size + 1)),
                "world_size": [world_size] * rollout_world_size,
                "master_metadata": [master_metadata] * rollout_world_size,
            },
        )

    def init_process_group(
        self,
        rank: int,
        world_size: int,
        master_metadata: Any = None,
    ) -> None:
        self.rank = int(rank)
        self.world_size = int(world_size)
        if master_metadata is not None:
            self.master_metadata = master_metadata

        if self.rank < 0:
            return
        if self.rank == 0:
            if self.metadata_socket is None:
                raise RuntimeError("rank 0 must call prepare(create_master_metadata=True) before init_process_group")
        else:
            if self.master_metadata is None:
                raise RuntimeError("NCCLSyncEngine requires ZeroMQ master metadata")
            if self.metadata_socket is None:
                self._connect_metadata_subscriber(self.master_metadata)

        initialized = self._collective.is_group_initialized(self.group_name)
        if self.rebuild_group and initialized:
            self._collective.destroy_collective_group(self.group_name)
            initialized = False

        if not initialized:
            self._collective.init_collective_group(
                world_size=self.world_size,
                rank=self.rank,
                backend="nccl",
                group_name=self.group_name,
            )

        self._group_initialized = True
        self._warmup_device_communicator()
        if self.rank == 0:
            self._wait_for_metadata_subscribers(expected=max(0, self.world_size - 1))

    async def send_weights(
        self,
        weight_stream: Iterable[tuple[str, torch.Tensor]],
        rollout_handle: Any | None = None,
        version: int | None = None,
    ) -> None:
        rank = self._require_rank("send_weights")
        if rank < 0:
            for _ in weight_stream:
                pass
            return
        if rank != 0:
            raise RuntimeError("only trainer rank 0 can send weights")

        active_buf, spare_buf = self._require_buffers("send_weights")
        bucket_meta: dict[str, TensorMeta] = {}
        offset = 0
        pending: asyncio.Task[dict[str, Any]] | None = None

        for name, tensor in weight_stream:
            nbytes = self._tensor_nbytes(tensor)
            if nbytes > self.bucket_size_bytes:
                raise ValueError(f"tensor {name} is larger than bucket_size_bytes={self.bucket_size_bytes}")

            if bucket_meta and offset + nbytes > self.bucket_size_bytes:
                pending = await self._send_bucket_after_pending(active_buf, bucket_meta, pending)
                active_buf, spare_buf = spare_buf, active_buf
                bucket_meta, offset = {}, 0

            self._copy_tensor_to_bucket(active_buf, offset, tensor)
            bucket_meta[name] = {
                "name": name,
                "shape": tensor.shape,
                "dtype": tensor.dtype,
                "offset": offset,
                "nbytes": nbytes,
            }
            offset += nbytes

        if pending is not None:
            await pending
        torch.cuda.synchronize()

        await self._start_transfer_bucket(active_buf, {"bucket_meta": bucket_meta, "is_last": True})
        self.send_buf, self.recv_buf = active_buf, spare_buf

    async def receive_weights(self, version: int | None = None) -> AsyncGenerator[tuple[str, torch.Tensor], None]:
        _ = version
        rank = self._require_rank("receive_weights")
        if rank <= 0:
            raise RuntimeError("only rollout ranks can receive weights")

        spare_buf, active_buf = self._require_buffers("receive_weights")
        metadata = await self._start_transfer_bucket(active_buf, metadata=None)

        while True:
            is_last = bool(metadata["is_last"])
            next_transfer = None if is_last else self._start_transfer_bucket(spare_buf, metadata=None)

            for item in self._restore_tensors(active_buf, metadata):
                yield item

            if is_last:
                break

            assert next_transfer is not None
            metadata = await next_transfer
            active_buf, spare_buf = spare_buf, active_buf

        self.send_buf, self.recv_buf = spare_buf, active_buf

    def finalize(self) -> None:
        self.send_buf = None
        self.recv_buf = None
        self._close_metadata_channel()

        if self.rebuild_group and self._group_initialized:
            self._collective.destroy_collective_group(self.group_name)
            self._group_initialized = False
            self.rank = None
            self.world_size = None

    async def _send_bucket_after_pending(
        self,
        bucket: torch.Tensor,
        bucket_meta: dict[str, TensorMeta],
        pending: asyncio.Task[dict[str, Any]] | None,
    ) -> asyncio.Task[dict[str, Any]]:
        torch.cuda.synchronize()
        if pending is not None:
            await pending
        return self._start_transfer_bucket(bucket, {"bucket_meta": bucket_meta, "is_last": False})

    def _start_transfer_bucket(
        self,
        bucket: torch.Tensor,
        metadata: dict[str, Any] | None,
    ) -> asyncio.Task[dict[str, Any]]:
        return asyncio.create_task(self._transfer_bucket(bucket, metadata))

    async def _transfer_bucket(
        self,
        bucket: torch.Tensor,
        metadata: dict[str, Any] | None,
    ) -> dict[str, Any]:
        if bucket.device.type == "cpu":
            return self._transfer_bucket_blocking(bucket, metadata)
        return await asyncio.to_thread(self._transfer_bucket_blocking, bucket, metadata)

    def _transfer_bucket_blocking(
        self,
        bucket: torch.Tensor,
        metadata: dict[str, Any] | None,
    ) -> dict[str, Any]:
        metadata = self._transfer_metadata(metadata)
        self._collective.broadcast(bucket, src_rank=0, group_name=self.group_name)
        return metadata

    def _transfer_metadata(self, metadata: dict[str, Any] | None) -> dict[str, Any]:
        rank = self._require_rank("metadata broadcast")
        socket = self._require_metadata_socket()
        if rank == 0:
            if metadata is None:
                raise RuntimeError("rank 0 must provide metadata")
            socket.send_string(self.METADATA_TOPIC, flags=zmq.SNDMORE)
            socket.send_pyobj(metadata)
            return metadata
        socket.recv_string()
        return socket.recv_pyobj()

    def _start_metadata_publisher(self) -> MasterMetadata:
        self._close_metadata_channel()
        ip_address = str(_get_node_ip_address()).strip("[]")
        context = zmq.Context()
        socket = context.socket(zmq.XPUB)
        socket.setsockopt(zmq.SNDHWM, 0)
        socket.setsockopt(zmq.RCVHWM, 0)
        socket.setsockopt(zmq.LINGER, 0)
        if hasattr(zmq, "XPUB_VERBOSE"):
            socket.setsockopt(zmq.XPUB_VERBOSE, 1)
        port = socket.bind_to_random_port(_zmq_bind_host(ip_address))
        self.metadata_context = context
        self.metadata_socket = socket
        return MasterMetadata(zmq_ip=ip_address, zmq_port=int(port))

    def _connect_metadata_subscriber(self, metadata: MasterMetadata) -> None:
        self._close_metadata_channel()
        context = zmq.Context()
        socket = context.socket(zmq.SUB)
        socket.setsockopt(zmq.RCVHWM, 0)
        socket.setsockopt(zmq.LINGER, 0)
        socket.setsockopt_string(zmq.SUBSCRIBE, self.METADATA_TOPIC)
        socket.connect(_zmq_connect_address(metadata))
        self.metadata_context = context
        self.metadata_socket = socket

    def _wait_for_metadata_subscribers(self, expected: int) -> None:
        socket = self._require_metadata_socket()
        deadline = time.monotonic() + float(os.environ.get(self.SUBSCRIBE_TIMEOUT_ENV, "30"))
        seen = 0
        while seen < expected:
            remaining = deadline - time.monotonic()
            if remaining <= 0 or not socket.poll(int(remaining * 1000)):
                raise RuntimeError(f"timed out waiting for {expected} ZeroMQ metadata subscribers")
            frame = socket.recv()
            if frame == b"\x01" + self.METADATA_TOPIC.encode("utf-8"):
                seen += 1

    def _warmup_device_communicator(self) -> None:
        token = torch.ones(1, dtype=torch.uint8, device=self.device)
        self._collective.broadcast(token, src_rank=0, group_name=self.group_name)

    def _close_metadata_channel(self) -> None:
        if self.metadata_socket is not None:
            try:
                self.metadata_socket.close(0)
            finally:
                self.metadata_socket = None
        if self.metadata_context is not None:
            try:
                self.metadata_context.destroy(linger=0)
            finally:
                self.metadata_context = None
        self.master_metadata = None

    @staticmethod
    def _tensor_nbytes(tensor: torch.Tensor) -> int:
        return int(tensor.numel() * tensor.element_size())

    @staticmethod
    def _copy_tensor_to_bucket(bucket: torch.Tensor, offset: int, tensor: torch.Tensor) -> None:
        nbytes = NCCLSyncEngine._tensor_nbytes(tensor)
        byte_view = tensor.detach().to(device=bucket.device).contiguous().view(-1).view(torch.uint8)
        bucket[offset : offset + nbytes].copy_(byte_view)

    @staticmethod
    def _restore_tensors(
        bucket: torch.Tensor,
        metadata: dict[str, Any],
    ) -> Iterator[tuple[str, torch.Tensor]]:
        for meta in metadata["bucket_meta"].values():
            tensor_bytes = bucket[meta["offset"] : meta["offset"] + meta["nbytes"]]
            yield meta["name"], tensor_bytes.view(meta["dtype"]).view(meta["shape"])

    def _require_rank(self, role: str) -> int:
        if self.rank is None:
            raise RuntimeError(f"init_process_group must be called before {role}")
        return self.rank

    def _require_buffers(self, role: str) -> tuple[torch.Tensor, torch.Tensor]:
        if self.send_buf is None or self.recv_buf is None:
            raise RuntimeError(f"prepare must be called before {role}")
        return self.send_buf, self.recv_buf

    def _require_metadata_socket(self) -> Any:
        if self.metadata_socket is None:
            raise RuntimeError("NCCLSyncEngine metadata socket is not initialized")
        return self.metadata_socket
