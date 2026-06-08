from __future__ import annotations

from collections.abc import Iterable
from typing import Any, AsyncGenerator

import torch


class BaseSyncEngine:
    """Base interface for weight sync backends."""

    def prepare(self) -> Any:
        return None

    @classmethod
    def build_topology(
        cls,
        trainer_world_size: int,
        rollout_world_size: int,
        master_metadata: Any = None,
    ) -> tuple[dict[str, list[Any]], dict[str, list[Any]]]:
        return {}, {}

    def init_process_group(
        self,
        rank: int = 0,
        world_size: int = 1,
        master_metadata: Any = None,
    ) -> None:
        return None

    async def send_weights(
        self,
        weight_stream: Iterable[tuple[str, torch.Tensor]],
        rollout_handle: Any | None = None,
        version: int | None = None,
    ) -> None:
        raise NotImplementedError(f"{type(self).__name__} does not implement send_weights")

    async def receive_weights(self, version: int | None = None) -> AsyncGenerator[tuple[str, torch.Tensor], None]:
        raise NotImplementedError(f"{type(self).__name__} does not implement receive_weights")
        yield

    def finalize(self) -> None:
        return None
