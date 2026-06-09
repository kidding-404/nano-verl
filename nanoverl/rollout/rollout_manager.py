from __future__ import annotations

import asyncio
import inspect
import os
from functools import wraps
from typing import Any

import ray
from ray.util.scheduling_strategies import NodeAffinitySchedulingStrategy

from nanoverl.config import RolloutConfig, SyncConfig, SystemConfig
from nanoverl.rollout.load_balancer import LoadBalancer
from nanoverl.sync.sync_manager import SyncManager


VLLM_SERVER_ENV_DEFAULTS = {
    "RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES": "1",
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


async def _resolve(value: Any) -> Any:
    if isinstance(value, ray.ObjectRef):
        if inspect.isawaitable(value):
            return await value
        return await asyncio.to_thread(ray.get, value)
    if inspect.isawaitable(value):
        return await value
    if isinstance(value, list):
        return await asyncio.gather(*[_resolve(item) for item in value])
    if isinstance(value, tuple):
        return tuple(await asyncio.gather(*[_resolve(item) for item in value]))
    if isinstance(value, dict):
        keys = list(value)
        resolved = await asyncio.gather(*[_resolve(value[key]) for key in keys])
        return dict(zip(keys, resolved, strict=True))
    return value


def _remote_call(target: Any, method: str, *args: Any, **kwargs: Any) -> Any:
    fn = getattr(target, method)
    return fn.remote(*args, **kwargs) if hasattr(fn, "remote") else fn(*args, **kwargs)


def _apply_vllm_server_env_defaults() -> None:
    for name, default in VLLM_SERVER_ENV_DEFAULTS.items():
        os.environ[name] = str(default)


ServerLayout = tuple[str, str | None, float]


def _resource_gpu_nodes(nodes: int, gpus_per_node: int) -> list[tuple[str, dict[str, Any]]]:
    selected: list[tuple[str, dict[str, Any]]] = []
    gpu_counts: list[str] = []
    for node in ray.nodes():
        if not node.get("Alive", False):
            continue
        resources = node.get("Resources") or {}
        gpu_count = int(float(resources.get("GPU", 0)))
        node_id = node.get("NodeID")
        if node_id is None:
            continue
        if gpu_count > 0:
            gpu_counts.append(f"{node_id}:{gpu_count}")
        if gpu_count >= gpus_per_node:
            selected.append((str(node_id), node))
    if len(selected) < nodes:
        available = ", ".join(gpu_counts) or "none"
        raise ValueError(
            f"Ray has {len(selected)} GPU node(s) with at least {gpus_per_node} GPU(s), "
            f"but resources.nodes requires {nodes}; available GPU nodes: {available}"
        )
    return selected[:nodes]


def _split_visible_devices(visible_devices: str) -> list[str]:
    return [device.strip() for device in str(visible_devices).split(",") if device.strip()]


def _server_layouts_from_devices(
    config: RolloutConfig,
    devices: list[str],
    node_id: str | None,
    num_gpus: float,
) -> list[ServerLayout]:
    tp_size = int(config.tensor_parallel_size)
    if not devices:
        raise RuntimeError("Cannot start rollout because no GPU devices were selected")
    if len(devices) % tp_size != 0:
        raise ValueError(
            f"rollout node {node_id} has {len(devices)} visible GPU device(s), "
            "which must be divisible by rollout.tensor_parallel_size"
        )
    return [
        (",".join(devices[start : start + tp_size]), node_id, num_gpus)
        for start in range(0, len(devices), tp_size)
    ]


def _colocated_server_layouts(
    config: RolloutConfig,
    visible_devices_by_node: list[str],
    node_ids: list[str] | None,
) -> list[ServerLayout]:
    node_ids = node_ids or [None] * len(visible_devices_by_node)
    layouts: list[ServerLayout] = []
    for visible_devices, node_id in zip(visible_devices_by_node, node_ids, strict=True):
        devices = _split_visible_devices(visible_devices)
        layouts.extend(_server_layouts_from_devices(config, devices, node_id, num_gpus=0.0))
    return layouts


def _standalone_server_layouts(config: SystemConfig, actor_mgr: Any | None = None) -> list[ServerLayout]:
    _ = actor_mgr
    resources = config.resources
    rollout_cfg = config.rollout
    node_entries = _resource_gpu_nodes(resources.nodes, resources.gpus_per_node)
    num_gpus = float(rollout_cfg.tensor_parallel_size)
    layouts: list[ServerLayout] = []
    for node_id, _node in node_entries[: resources.rollout_nodes]:
        for _ in range(resources.rollout_servers_per_node):
            layouts.append(("", node_id, num_gpus))
    if len(layouts) != resources.rollout_world_size:
        raise RuntimeError(
            f"expected {resources.rollout_world_size} rollout server(s), launched {len(layouts)}"
        )
    return layouts


def create_vllm_server(
    *,
    config: RolloutConfig,
    server_id: str,
    server_rank: int,
    visible_devices: str,
    node_id: str | None = None,
    num_gpus: float = 0.0,
) -> Any:
    _apply_vllm_server_env_defaults()
    from nanoverl.rollout.vllm_server import VLLMServer

    options: dict[str, Any] = {}
    if config.ray_num_cpus_per_server > 0:
        options["num_cpus"] = float(config.ray_num_cpus_per_server)
    if num_gpus > 0:
        options["num_gpus"] = float(num_gpus)

    if node_id is not None:
        options["scheduling_strategy"] = NodeAffinitySchedulingStrategy(
            node_id=node_id,
            soft=False,
        )

    return VLLMServer.options(**options).remote(
        config=config,
        server_id=server_id,
        server_rank=server_rank,
        node_rank=0,
        nnodes=1,
        visible_devices=visible_devices,
    )


def _as_list(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, (str, bytes, dict)):
        return [value]
    if isinstance(value, list):
        return value
    if isinstance(value, tuple):
        return list(value)
    if hasattr(value, "__len__") and hasattr(value, "__getitem__"):
        try:
            return [value[index] for index in range(len(value))]
        except TypeError:
            pass
    return [value]


def _is_single_prompt(value: Any) -> bool:
    ndim = getattr(value, "ndim", None)
    if ndim is None and hasattr(value, "dim"):
        ndim = value.dim()
    if ndim is not None:
        return int(ndim) == 1
    return isinstance(value, (list, tuple)) and all(isinstance(item, int) for item in value)


def _to_python_list(value: Any) -> list[Any]:
    return value.tolist() if hasattr(value, "tolist") else list(value)


class RolloutManager:
    """Trainer-facing rollout entrypoint."""

    def __init__(
        self,
        config: RolloutConfig,
        load_balancer: Any,
        vllm_servers: list[Any] | None = None,
    ) -> None:
        self.config = config
        self.load_balancer = load_balancer
        self.vllm_servers = list(vllm_servers or [])
        self.actor_mgr: Any | None = None
        self.sync_cfg: SyncConfig | None = None
        self.sync_mgr: SyncManager | None = None

    @classmethod
    def launch(
        cls,
        config: SystemConfig | RolloutConfig,
        tokenizer: Any | None = None,
        run_id: str | None = None,
        actor_mgr: Any | None = None,
    ) -> "RolloutManager":
        _ = tokenizer
        if not isinstance(config, SystemConfig):
            raise TypeError("RolloutManager.launch requires SystemConfig so resources can be used")
        rollout_cfg = config.rollout
        if not ray.is_initialized():
            init_kwargs: dict[str, Any] = {
                "ignore_reinit_error": True,
                "namespace": rollout_cfg.ray_namespace,
                "log_to_driver": True,
            }
            if rollout_cfg.ray_address:
                init_kwargs["address"] = rollout_cfg.ray_address
            else:
                init_kwargs["num_gpus"] = int(config.resources.gpus_per_node)
            ray.init(**init_kwargs)

        if rollout_cfg.mode.lower().strip() == "hybrid":
            if actor_mgr is None:
                raise ValueError("rollout.mode='hybrid' requires an initialized ActorManager")
            visible_devices_by_node, node_ids = actor_mgr.get_colocated_rollout_layout()
            print(
                f"[rollout] hybrid visible_devices_by_node={visible_devices_by_node}",
                flush=True,
            )
            server_layouts = _colocated_server_layouts(rollout_cfg, visible_devices_by_node, node_ids)
        else:
            server_layouts = _standalone_server_layouts(config, actor_mgr)
        if len(server_layouts) != config.resources.rollout_world_size:
            raise RuntimeError(
                f"expected {config.resources.rollout_world_size} rollout server(s), "
                f"launched {len(server_layouts)}"
            )

        servers = {
            str(server_rank): create_vllm_server(
                config=rollout_cfg,
                server_id=f"{run_id or 'rollout'}-{server_rank}",
                server_rank=server_rank,
                visible_devices=visible_devices,
                node_id=node_id,
                num_gpus=num_gpus,
            )
            for server_rank, (visible_devices, node_id, num_gpus) in enumerate(server_layouts)
        }
        load_balancer = LoadBalancer.remote(config=rollout_cfg, servers=servers)
        return cls(
            config=rollout_cfg,
            load_balancer=load_balancer,
            vllm_servers=list(servers.values()),
        )

    def wait_until_ready(self, timeout: int = 600, interval: float = 1.0) -> None:
        _ = interval
        if not self.vllm_servers:
            return
        ray.get([server.start.remote() for server in self.vllm_servers], timeout=timeout)

    def bind_actor_manager(self, actor_mgr: Any, sync_cfg: SyncConfig) -> None:
        self.actor_mgr = actor_mgr
        self.sync_cfg = sync_cfg
        self.sync_mgr = SyncManager(
            config=sync_cfg,
            rollout_mode=self.config.mode,
            trainer_wg=actor_mgr,
            servers=self.vllm_servers,
            rollout_servers=self.vllm_servers,
            request_aborters=[self.load_balancer],
        )

    def on_actor_state_changed(self, step: int) -> None:
        if self.sync_mgr is None:
            return
        self.sync_mgr.update_weights(step)

    def shutdown(self) -> None:
        for server in self.vllm_servers:
            try:
                ray.get(server.shutdown.remote(), timeout=120)
            except Exception:
                pass
            try:
                ray.kill(server, no_restart=True)
            except Exception:
                pass
        try:
            ray.kill(self.load_balancer, no_restart=True)
        except Exception:
            pass
        self.vllm_servers = []

    def _split_batch(self, batch: Any) -> list[dict[str, Any]]:
        """Return request dicts with request_id, prompt_ids, and sampling_params."""
        if isinstance(batch, list):
            return [dict(item) for item in batch]

        if hasattr(batch, "to_dicts"):
            return [dict(item) for item in batch.to_dicts()]

        if not isinstance(batch, dict):
            raise TypeError("batch must be a list of dicts, a dict of columns, or define to_dicts()")

        prompt_ids = batch.get("prompt_ids")
        if prompt_ids is None:
            prompt_ids = batch.get("prompts")
        if prompt_ids is None:
            raise KeyError("batch must contain prompt_ids or prompts")

        prompt_ids = [prompt_ids] if _is_single_prompt(prompt_ids) else _as_list(prompt_ids)
        batch_size = len(prompt_ids)
        request_ids = _as_list(batch.get("request_ids", batch.get("request_id")))
        sampling_params = batch.get("sampling_params", {})

        if isinstance(sampling_params, dict):
            sampling_params = [sampling_params] * batch_size
        else:
            sampling_params = _as_list(sampling_params)

        if not request_ids:
            request_ids = [f"rollout-{index}" for index in range(batch_size)]
        elif len(request_ids) == 1 and batch_size > 1:
            request_ids = [f"{request_ids[0]}-{index}" for index in range(batch_size)]

        if len(request_ids) != batch_size:
            raise ValueError("request_ids length must match prompt_ids length")
        if len(sampling_params) != batch_size:
            raise ValueError("sampling_params length must match prompt_ids length")

        return [
            {
                "request_id": str(request_ids[index]),
                "prompt_ids": _to_python_list(prompt_ids[index]),
                "sampling_params": dict(sampling_params[index]),
            }
            for index in range(batch_size)
        ]

    @auto_await
    async def generate_sequences(self, batch: Any) -> list[Any]:
        generation_id = await _resolve(_remote_call(self.load_balancer, "begin_generation"))
        requests = self._split_batch(batch)
        batch_size = max(1, int(getattr(self.config, "batch_size", len(requests) or 1)))
        outputs: list[Any] = []
        for start in range(0, len(requests), batch_size):
            chunk = requests[start : start + batch_size]
            calls = [
                _remote_call(
                    self.load_balancer,
                    "generate",
                    item["request_id"],
                    item["prompt_ids"],
                    item["sampling_params"],
                    generation_id,
                )
                for item in chunk
            ]
            for output in await _resolve(calls):
                if isinstance(output, list):
                    outputs.extend(output)
                else:
                    outputs.append(output)
        return outputs

    @auto_await
    async def sleep(self) -> None:
        await _resolve(_remote_call(self.load_balancer, "sleep_all"))

    @auto_await
    async def wake_up(self) -> None:
        await _resolve(_remote_call(self.load_balancer, "wake_up_all"))

    @auto_await
    async def clear_cache(self) -> None:
        await _resolve(_remote_call(self.load_balancer, "clear_cache_all"))

    @auto_await
    async def abort_all_requests(self) -> None:
        reset_prefix_cache = bool(getattr(self.config, "reset_prefix_cache_on_abort", True))
        await _resolve(
            _remote_call(
                self.load_balancer,
                "abort_all_requests",
                reset_prefix_cache=reset_prefix_cache,
            )
        )

    @auto_await
    async def resume_generation(self) -> None:
        await _resolve(_remote_call(self.load_balancer, "resume_generation"))

    @auto_await
    async def get_metrics(self) -> dict[str, Any]:
        loads, server_infos = await _resolve(
            [
                _remote_call(self.load_balancer, "get_loads"),
                _remote_call(self.load_balancer, "get_server_infos"),
            ]
        )

        servers = {
            server_id: (info.__dict__ if hasattr(info, "__dict__") else dict(info))
            for server_id, info in server_infos.items()
        }
        versions = {
            server_id: int(info.get("model_version", -1))
            for server_id, info in servers.items()
        }

        return {
            "num_servers": len(servers),
            "inflight_requests": sum(int(value) for value in loads.values()),
            "loads": loads,
            "server_versions": versions,
            "servers": servers,
        }
