from __future__ import annotations

import asyncio
import inspect
import time
from functools import wraps
from typing import Any

import ray

from nanoverl.config import SyncConfig
from nanoverl.sync.base_engine import BaseSyncEngine
from nanoverl.sync.nccl_sync_engine import NCCLSyncEngine

_ENGINE_BY_BACKEND: dict[str, type[BaseSyncEngine]] = {
    "nccl": NCCLSyncEngine,
}


def auto_await(fn):
    @wraps(fn)
    def wrapper(*args, **kwargs):
        coro = fn(*args, **kwargs)
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            return asyncio.run(coro)
        return coro

    return wrapper


class SyncManager:
    """Orchestrates one trainer-to-rollout weight sync transaction."""

    def __init__(
        self,
        config: SyncConfig,
        rollout_mode: str,
        trainer_wg,
        servers: list[Any],
        rollout_servers: list[Any],
        request_aborters: list[Any] | None = None,
    ) -> None:
        self.config = config
        self.config.backend = self.config.backend.lower().strip()
        self.rollout_mode = rollout_mode.lower().strip()
        self.trainer_wg = trainer_wg
        self.servers = list(servers)
        self.rollout_servers = list(rollout_servers)
        self.request_aborters = list(request_aborters or [])
        self._node_rollout_servers: list[Any] | None = None
        self.last_timing: dict[str, float] = {}
        self.validate_mode_backend()
        self.engine_cls = self._engine_cls() if self.config.backend in _ENGINE_BY_BACKEND else None

    def _engine_cls(self) -> type[BaseSyncEngine]:
        try:
            return _ENGINE_BY_BACKEND[self.config.backend]
        except KeyError as exc:
            raise ValueError(f"unsupported sync backend: {self.config.backend}") from exc

    def validate_mode_backend(self) -> None:
        if (self.rollout_mode, self.config.backend) not in {("hybrid", "naive"), ("standalone", "nccl")}:
            raise ValueError(
                "valid sync pairs are hybrid+naive and standalone+nccl, "
                f"got {self.rollout_mode}+{self.config.backend}"
            )

    async def _resolve(self, value: Any) -> Any:
        if isinstance(value, ray.ObjectRef):
            if inspect.isawaitable(value):
                return await value
            return await asyncio.to_thread(ray.get, value)
        if inspect.isawaitable(value):
            return await value
        if isinstance(value, list):
            return await asyncio.gather(*[self._resolve(item) for item in value])
        if isinstance(value, tuple):
            return tuple(await asyncio.gather(*[self._resolve(item) for item in value]))
        if isinstance(value, dict):
            keys = list(value)
            resolved = await asyncio.gather(*[self._resolve(value[key]) for key in keys])
            return dict(zip(keys, resolved, strict=True))
        return value

    def _call(self, target: Any, method: str, *args: Any, **kwargs: Any) -> Any:
        fn = getattr(target, method)
        return fn.remote(*args, **kwargs) if hasattr(fn, "remote") else fn(*args, **kwargs)

    async def _call_many(
        self,
        targets: list[Any],
        method: str,
        *args: Any,
        optional: bool = False,
        **kwargs: Any,
    ) -> Any:
        if optional:
            targets = [target for target in targets if hasattr(target, method)]
            if not targets:
                return False

        result = await self._resolve([self._call(target, method, *args, **kwargs) for target in targets])
        return True if optional else result

    def _group_size(self, target: Any) -> int:
        workers = getattr(target, "workers", None)
        return len(target) if isinstance(target, (list, tuple)) else len(workers) if workers is not None else 1

    async def _get_node_rollout_servers(self) -> list[Any]:
        if self._node_rollout_servers is not None:
            return self._node_rollout_servers

        servers: list[Any] = []
        for target in self.rollout_servers:
            if hasattr(target, "get_servers"):
                servers.extend(await self._resolve(self._call(target, "get_servers")))
            else:
                servers.append(target)

        self._node_rollout_servers = servers
        return servers

    @staticmethod
    def _rank_kwargs(kwargs_by_name: dict[str, list[Any]], rank: int) -> dict[str, Any]:
        return {key: values[rank] for key, values in kwargs_by_name.items()}

    async def _call_ranked(
        self,
        target: Any,
        method: str,
        kwargs_by_name: dict[str, list[Any]],
        *args: Any,
    ) -> None:
        if isinstance(target, list):
            await self._resolve([
                self._call(item, method, *args, **self._rank_kwargs(kwargs_by_name, i))
                for i, item in enumerate(target)
            ])
            return
        await self._resolve(self._call(target, method, *args, **kwargs_by_name))

    async def _rollout_engine(self, method: str, *args: Any, **kwargs: Any) -> list[Any]:
        servers = await self._get_node_rollout_servers()
        return await self._call_many(servers, "execute_sync_engine", method, *args, **kwargs)

    @auto_await
    async def sleep_servers(self) -> None:
        if await self._call_many(self.servers, "sleep", optional=True):
            return
        await self._call_many(self.servers, "release", level=1, optional=True)

    @auto_await
    async def wake_up_servers(self) -> None:
        for method, kwargs in (
            ("wake_up", {}),
            ("wake", {}),
            ("resume", {"tags": ["weights", "kv_cache"]}),
        ):
            if await self._call_many(self.servers, method, optional=True, **kwargs):
                return

    @auto_await
    async def build_process_group(self) -> None:
        if self.rollout_mode != "standalone":
            return

        servers = await self._get_node_rollout_servers()
        master_metadata = await self._resolve(self._call(self.trainer_wg, "prepare"))
        await self._call_many(servers, "execute_sync_engine", "prepare")

        assert self.engine_cls is not None
        trainer_kwargs, rollout_kwargs = self.engine_cls.build_topology(
            trainer_world_size=self._group_size(self.trainer_wg),
            rollout_world_size=len(servers),
            master_metadata=master_metadata,
        )
        await asyncio.gather(
            self._call_ranked(self.trainer_wg, "init_process_group", trainer_kwargs),
            self._call_ranked(
                servers,
                "execute_sync_engine",
                rollout_kwargs,
                "init_process_group",
            ),
        )

    async def _finalize(self) -> None:
        if hasattr(self.trainer_wg, "finalize"):
            await self._resolve(self._call(self.trainer_wg, "finalize"))
        if await self._get_node_rollout_servers():
            await self._rollout_engine("finalize")

    async def _sleep_rollout_servers(self, servers: list[Any]) -> None:
        await self._call_many(servers, "sleep", optional=True)

    async def _wake_rollout_servers(self, servers: list[Any]) -> None:
        await self._call_many(servers, "wake_up", optional=True)

    async def _hybrid_update(self, version: int) -> None:
        await self.sleep_servers()
        try:
            await self._resolve(self._call(self.trainer_wg, "update_weights", version, rollout_handles=self.servers))
        finally:
            await self.wake_up_servers()

    async def _standalone_update(self, version: int) -> None:
        timing: dict[str, float] = {}
        servers = await self._get_node_rollout_servers()
        started = time.perf_counter()
        if self.request_aborters:
            await self._call_many(self.request_aborters, "abort_all_requests", optional=True)
        else:
            await self._call_many(servers, "abort_all_requests", optional=True)
        timing["sync_abort"] = time.perf_counter() - started
        started = time.perf_counter()
        await self._sleep_rollout_servers(servers)
        timing["sync_sleep"] = time.perf_counter() - started
        started = time.perf_counter()
        await self.build_process_group()
        timing["sync_build_pg"] = time.perf_counter() - started

        try:
            started = time.perf_counter()
            trainer_task = self._resolve(self._call(self.trainer_wg, "update_weights", version))
            rollout_task = self._call_many(servers, "update_weights_from_sync_engine", version)
            await asyncio.gather(trainer_task, rollout_task)
            timing["sync_transfer"] = time.perf_counter() - started
        finally:
            started = time.perf_counter()
            await self._finalize()
            timing["sync_finalize"] = time.perf_counter() - started

        started = time.perf_counter()
        await self._wake_rollout_servers(servers)
        await self._call_many(servers, "resume_generation", optional=True)
        timing["sync_wake_resume"] = time.perf_counter() - started
        self.last_timing = timing

    @auto_await
    async def update_weights(self, version: int) -> None:
        self.last_timing = {}
        self.validate_mode_backend()
        if self.rollout_mode == "hybrid":
            await self._hybrid_update(version)
            return
        await self._standalone_update(version)
