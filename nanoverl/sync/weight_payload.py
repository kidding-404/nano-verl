from __future__ import annotations

from multiprocessing import shared_memory
from typing import Any

import torch
from torch.multiprocessing.reductions import reduce_tensor


def align_offset(offset: int, alignment: int) -> int:
    alignment = max(1, int(alignment))
    return ((int(offset) + alignment - 1) // alignment) * alignment


def normalize_weight_transport(transport: str) -> str:
    normalized = str(transport).lower().strip()
    if normalized not in {"auto", "cuda_ipc", "shared_memory"}:
        raise ValueError(f"unsupported rollout.weight_sync_transport: {transport}")
    return normalized


def weight_entry(name: str, tensor: torch.Tensor, offset: int) -> dict[str, Any]:
    return {
        "name": str(name),
        "dtype": tensor.dtype,
        "shape": tuple(tensor.shape),
        "offset": int(offset),
        "nbytes": int(tensor.numel()) * int(tensor.element_size()),
    }


def copy_tensor_to_bytes(dst: torch.Tensor, src: torch.Tensor) -> None:
    dst.view(src.dtype).view(src.shape).copy_(src.detach(), non_blocking=True)


def rebuild_ipc_tensor(handle: Any, device_id: int | None = None) -> torch.Tensor:
    rebuild, args = handle
    if device_id is not None:
        args = list(args)
        args[6] = int(device_id)
    return rebuild(*args)


def cuda_ipc_payload(bucket: torch.Tensor, entries: tuple[dict[str, Any], ...]) -> dict[str, Any]:
    if not bucket.is_cuda:
        raise RuntimeError("CUDA IPC weight payload requires a CUDA bucket")
    return {
        "transport": "cuda_ipc",
        "handle": reduce_tensor(bucket),
        "entries": entries,
    }


def shared_memory_payload(bucket: torch.Tensor, entries: tuple[dict[str, Any], ...]) -> tuple[dict[str, Any], Any]:
    size = int(bucket.numel())
    shm = shared_memory.SharedMemory(create=True, size=size)
    dst = torch.frombuffer(shm.buf[:size], dtype=torch.uint8)
    try:
        dst.copy_(bucket.detach().cpu())
    finally:
        del dst
    return {
        "transport": "shared_memory",
        "handle": {"name": shm.name, "size": size},
        "entries": entries,
    }, shm


def bucket_from_weight_payload(payload: dict[str, Any], device: torch.device | None) -> tuple[torch.Tensor, Any | None]:
    transport = str(payload["transport"])
    handle = payload["handle"]
    if transport == "cuda_ipc":
        device_id = getattr(device, "index", None)
        if device_id is None and torch.cuda.is_available():
            device_id = torch.cuda.current_device()
        return rebuild_ipc_tensor(handle, device_id), None
    if transport == "shared_memory":
        shm = shared_memory.SharedMemory(name=str(handle["name"]))
        size = int(handle["size"])
        return torch.frombuffer(shm.buf[:size], dtype=torch.uint8), shm
    raise ValueError(f"unsupported weight payload transport: {transport}")
