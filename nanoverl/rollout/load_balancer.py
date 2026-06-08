from __future__ import annotations

import asyncio
from collections import OrderedDict
from typing import Any

import ray

from nanoverl.config import RolloutConfig
from nanoverl.data import TokenOutput


@ray.remote
class LoadBalancer:
    """Route rollout requests across complete VLLMServer actors."""

    def __init__(
        self,
        config: RolloutConfig,
        servers: dict[str, Any],
        sticky_cache_size: int = 10000,
    ) -> None:
        if not servers:
            raise ValueError("LoadBalancer requires at least one VLLMServer")

        self.config = config
        self.servers = dict(servers)
        self.sticky_cache_size = int(sticky_cache_size)

        self._request_to_server: OrderedDict[str, str] = OrderedDict()
        self._inflight_requests: dict[str, int] = {
            server_id: 0 for server_id in self.servers
        }
        self._inflight_tokens: dict[str, int] = {
            server_id: 0 for server_id in self.servers
        }
        self._generation_id = 0
        self._aborted_generations: set[int] = set()

    def _max_num_seqs(self) -> int:
        return int(getattr(self.config, "max_num_seqs", 1))

    def _max_num_batched_tokens(self) -> int:
        return int(getattr(self.config, "max_num_batched_tokens", 1 << 60))

    def _estimate_token_budget(
        self,
        prompt_ids: list[int],
        sampling_params: dict[str, Any],
    ) -> int:
        _ = sampling_params
        return int(len(prompt_ids))

    def _remember_sticky_route(self, request_id: str, server_id: str) -> None:
        if request_id in self._request_to_server:
            self._request_to_server.move_to_end(request_id)

        self._request_to_server[request_id] = server_id

        while len(self._request_to_server) > self.sticky_cache_size:
            self._request_to_server.popitem(last=False)

    def _has_capacity(self, server_id: str, token_budget: int) -> bool:
        _ = token_budget
        return self._inflight_requests[server_id] < self._max_num_seqs()

    def _least_loaded_server(self, token_budget: int) -> str:
        candidates = [
            server_id
            for server_id in self.servers
            if self._has_capacity(server_id, token_budget)
        ]

        if not candidates:
            raise RuntimeError("no rollout VLLMServer has capacity for this request")

        return min(
            candidates,
            key=lambda server_id: (
                self._inflight_requests[server_id],
                self._inflight_tokens[server_id],
                server_id,
            ),
        )

    def _acquire_server(
        self,
        request_id: str,
        token_budget: int,
    ) -> tuple[str, Any]:
        request_id = str(request_id)

        server_id = self._request_to_server.get(request_id)
        if server_id is not None and self._has_capacity(server_id, token_budget):
            self._request_to_server.move_to_end(request_id)
        else:
            server_id = self._least_loaded_server(token_budget)
            self._remember_sticky_route(request_id, server_id)

        self._inflight_requests[server_id] += 1
        self._inflight_tokens[server_id] += token_budget

        return server_id, self.servers[server_id]

    async def _wait_for_server(
        self,
        request_id: str,
        token_budget: int,
        generation_id: int,
    ) -> tuple[str, Any] | None:
        if token_budget > self._max_num_batched_tokens():
            raise RuntimeError(
                f"request token budget {token_budget} exceeds rollout.max_num_batched_tokens "
                f"{self._max_num_batched_tokens()}"
        )
        while True:
            if generation_id in self._aborted_generations:
                return None
            try:
                server_id, server = self._acquire_server(request_id, token_budget)
                if generation_id in self._aborted_generations:
                    self._release_server(server_id, token_budget)
                    return None
                return server_id, server
            except RuntimeError:
                await asyncio.sleep(0.05)

    def _release_server(self, server_id: str, token_budget: int) -> None:
        self._inflight_requests[server_id] = max(0, self._inflight_requests[server_id] - 1)
        self._inflight_tokens[server_id] = max(0, self._inflight_tokens[server_id] - token_budget)

    async def _broadcast(self, method: str, *args: Any, **kwargs: Any) -> list[Any]:
        return await asyncio.gather(
            *[
                getattr(server, method).remote(*args, **kwargs)
                for server in self.servers.values()
            ]
        )

    async def generate(
        self,
        request_id: str,
        prompt_ids: list[int],
        sampling_params: dict[str, Any],
        generation_id: int | None = None,
    ) -> TokenOutput | list[TokenOutput]:
        generation_id = self._generation_id if generation_id is None else int(generation_id)
        token_budget = self._estimate_token_budget(prompt_ids, sampling_params)
        route = await self._wait_for_server(request_id, token_budget, generation_id)
        if route is None:
            return TokenOutput(token_ids=[], log_probs=[], stop_reason="aborted")
        server_id, server = route

        try:
            output = await server.generate.remote(
                str(request_id),
                prompt_ids,
                sampling_params,
            )
            return output
        finally:
            self._release_server(server_id, token_budget)

    async def begin_generation(self) -> int:
        self._generation_id += 1
        return self._generation_id

    async def sleep_all(self) -> None:
        await self._broadcast("sleep")

    async def wake_up_all(self) -> None:
        await self._broadcast("wake_up")

    async def clear_cache_all(self) -> None:
        await self._broadcast("clear_cache")

    async def abort_all_requests(self, reset_prefix_cache: bool = True) -> None:
        self._aborted_generations.add(self._generation_id)
        await self._broadcast("abort_all_requests", reset_prefix_cache=reset_prefix_cache)

        for server_id in self._inflight_requests:
            self._inflight_requests[server_id] = 0
            self._inflight_tokens[server_id] = 0

    async def resume_generation(self) -> None:
        await self._broadcast("resume_generation")

    def get_loads(self) -> dict[str, int]:
        return dict(self._inflight_requests)

    async def get_server_infos(self) -> dict[str, Any]:
        infos = await asyncio.gather(
            *[
                server.get_info.remote()
                for server in self.servers.values()
            ]
        )

        return {
            server_id: info
            for server_id, info in zip(self.servers.keys(), infos, strict=True)
        }
