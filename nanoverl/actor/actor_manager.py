from __future__ import annotations

import os
from typing import Any

import ray
from ray.util.scheduling_strategies import NodeAffinitySchedulingStrategy

from nanoverl.actor.fsdp_worker import FSDPActorWorker
from nanoverl.config import SystemConfig
from nanoverl.data import DataProto


FSDP_WORKER_ENV_DEFAULTS = {
    "NCCL_CUMEM_ENABLE": "0",
    "NCCL_IB_DISABLE": "1",
    "NCCL_NET_PLUGIN": "",
    "NCCL_SOCKET_FAMILY": "AF_INET",
    "NCCL_SOCKET_IFNAME": "eth0",
    "NCCL_TUNER_PLUGIN": "",
    "NCCL_FASTRAK_ENABLE_CONTROL_CHANNEL": "0",
}


def _apply_env_defaults(defaults: dict[str, str]) -> None:
    for name, default in defaults.items():
        os.environ[name] = str(default)


def apply_fsdp_worker_env_defaults() -> None:
    _apply_env_defaults(FSDP_WORKER_ENV_DEFAULTS)


def apply_ray_worker_env_defaults() -> None:
    apply_fsdp_worker_env_defaults()


def _ensure_ray_initialized(address: str | None, namespace: str, num_gpus: int | None = None) -> None:
    if ray.is_initialized():
        return
    apply_ray_worker_env_defaults()
    init_kwargs: dict[str, Any] = {
        "ignore_reinit_error": True,
        "namespace": namespace,
        "log_to_driver": True,
    }
    if address:
        init_kwargs["address"] = address
    elif num_gpus is not None:
        init_kwargs["num_gpus"] = int(num_gpus)
    ray.init(**init_kwargs)


def _resource_gpu_node_ids(nodes: int, gpus_per_node: int) -> list[str]:
    node_ids: list[str] = []
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
            node_ids.append(str(node_id))
    if len(node_ids) < nodes:
        available = ", ".join(gpu_counts) or "none"
        raise ValueError(
            f"Ray has {len(node_ids)} GPU node(s) with at least {gpus_per_node} GPU(s), "
            f"but resources.nodes requires {nodes}; available GPU nodes: {available}"
        )
    return node_ids[:nodes]


class ActorManager:
    def __init__(
        self,
        workers: list[Any],
        dp_size: int | None = None,
        local_ranks: list[int] | None = None,
        local_world_sizes: list[int] | None = None,
    ) -> None:
        self.workers = list(workers)
        self.dp_size = int(dp_size if dp_size is not None else len(self.workers))
        self.local_ranks = list(local_ranks or [0 for _ in self.workers])
        self.local_world_sizes = list(local_world_sizes or [1 for _ in self.workers])

    @classmethod
    def launch(
        cls,
        config: SystemConfig,
        tokenizer: Any,
        backend_cfg: dict | None = None,
    ) -> "ActorManager":
        _ensure_ray_initialized(
            config.rollout.ray_address,
            config.rollout.ray_namespace,
            num_gpus=config.resources.gpus_per_node,
        )
        resources = config.resources
        node_ids = _resource_gpu_node_ids(resources.nodes, resources.gpus_per_node)
        actor_node_ids = node_ids[: resources.actor_nodes]
        worker_options: dict[str, Any] = {
            "num_cpus": float(config.actor.ray_num_cpus_per_worker),
            "num_gpus": 1.0,
        }
        remote_worker_cls = ray.remote(FSDPActorWorker)
        workers: list[Any] = []
        local_ranks: list[int] = []
        local_world_sizes: list[int] = []
        local_world_size = int(resources.actor_gpus_per_node)
        for node_id in actor_node_ids:
            for local_rank in range(local_world_size):
                options = dict(worker_options)
                options["scheduling_strategy"] = NodeAffinitySchedulingStrategy(
                    node_id=node_id,
                    soft=False,
                )
                workers.append(
                    remote_worker_cls.options(**options).remote(
                        tokenizer=tokenizer,
                        backend_cfg=backend_cfg,
                    )
                )
                local_ranks.append(local_rank)
                local_world_sizes.append(local_world_size)
        if len(workers) != resources.actor_world_size:
            raise RuntimeError(
                f"expected {resources.actor_world_size} actor worker(s), launched {len(workers)}"
            )
        return cls(
            workers,
            dp_size=resources.actor_world_size,
            local_ranks=local_ranks,
            local_world_sizes=local_world_sizes,
        )

    def init_model(self, config: Any) -> None:
        if not self.workers:
            return
        master_addr, master_port = ray.get(self.workers[0].get_master_addr_port.remote())
        futures = []
        for rank, worker in enumerate(self.workers):
            futures.append(
                worker.init_model.remote(
                    config=config,
                    rank=rank,
                    world_size=len(self.workers),
                    master_addr=master_addr,
                    master_port=master_port,
                    local_rank=self.local_ranks[rank],
                    local_world_size=self.local_world_sizes[rank],
                )
            )
        ray.get(futures)

    def _call_all(self, method_name: str, *args: Any, **kwargs: Any) -> list[Any]:
        if not self.workers:
            return []
        return ray.get([getattr(worker, method_name).remote(*args, **kwargs) for worker in self.workers])

    def _split_data(self, data: DataProto) -> list[DataProto]:
        if not self.workers:
            return []
        world_size = len(self.workers)
        if len(data) == 0:
            raise ValueError("Cannot dispatch an empty DataProto to actor workers")
        if len(data) % world_size != 0:
            raise ValueError(
                f"Actor batch size must be divisible by worker count: batch={len(data)} workers={world_size}"
            )
        shard_size = len(data) // world_size
        return [data[start : start + shard_size] for start in range(0, len(data), shard_size)]

    def _call_sharded_with_shards(
        self,
        method_name: str,
        data: DataProto,
        *args: Any,
        **kwargs: Any,
    ) -> tuple[list[DataProto], list[Any]]:
        shards = self._split_data(data)
        outputs = ray.get([
            getattr(worker, method_name).remote(shard, *args, **kwargs)
            for worker, shard in zip(self.workers, shards, strict=True)
        ])
        return shards, outputs

    def _call_sharded(self, method_name: str, data: DataProto, *args: Any, **kwargs: Any) -> list[Any]:
        _, outputs = self._call_sharded_with_shards(method_name, data, *args, **kwargs)
        return outputs

    def _compute(self, method_name: str, data: DataProto, *args: Any, **kwargs: Any) -> DataProto:
        return DataProto.concat(self._call_sharded(method_name, data, *args, **kwargs))

    @staticmethod
    def _micro_batch_kwargs(micro_batch_size: int | None) -> dict[str, int]:
        return {} if micro_batch_size is None else {"micro_batch_size": int(micro_batch_size)}

    def compute_log_prob(self, data: DataProto, micro_batch_size: int | None = None) -> DataProto:
        return self._compute("compute_log_prob", data, **self._micro_batch_kwargs(micro_batch_size))

    def compute_ref_log_prob(self, data: DataProto, micro_batch_size: int | None = None) -> DataProto:
        return self._compute("compute_ref_log_prob", data, **self._micro_batch_kwargs(micro_batch_size))

    def update_policy(self, data: DataProto, micro_batch_size: int | None = None) -> dict[str, float]:
        if "response_mask" in data.batch and "global_response_tokens" not in data.meta_info:
            data.meta_info["global_response_tokens"] = float(data.batch["response_mask"].sum().item())
        shards, outputs = self._call_sharded_with_shards(
            "update_policy",
            data,
            **self._micro_batch_kwargs(micro_batch_size),
        )
        weights = [self._metric_weight(shard) for shard in shards]
        return self._reduce_metrics(outputs, weights)

    def get_colocated_rollout_layout(self) -> tuple[list[str], list[str]]:
        infos = self._call_all("get_ray_runtime_info")
        node_devices: dict[str, list[str]] = {}
        for info in infos:
            node_id = str(info["node_id"])
            devices = [item for item in str(info.get("visible_devices", "")).split(",") if item != ""]
            if not devices:
                raise RuntimeError("Cannot colocate rollout because Ray did not assign visible GPU devices")
            node_devices.setdefault(node_id, [])
            for device in devices:
                if device not in node_devices[node_id]:
                    node_devices[node_id].append(device)
        node_ids = list(node_devices)
        return [",".join(node_devices[node_id]) for node_id in node_ids], node_ids

    def prepare(self) -> Any:
        if not self.workers:
            return None
        return self.workers[0].prepare.remote()

    @staticmethod
    def _rank_arg(value: Any, rank: int) -> Any:
        return value[rank] if isinstance(value, list) else value

    def init_process_group(self, **kwargs: Any) -> list[Any]:
        return [
            worker.init_process_group.remote(
                **{key: self._rank_arg(value, rank) for key, value in kwargs.items()}
            )
            for rank, worker in enumerate(self.workers)
        ]

    def update_weights(self, version: int, rollout_handles: list[Any] | None = None) -> list[Any]:
        if rollout_handles is not None:
            return [worker.update_rollout_weights.remote(version, rollout_handles) for worker in self.workers]
        return [worker.update_weights.remote(version) for worker in self.workers]

    def finalize(self) -> list[Any]:
        return [worker.finalize.remote() for worker in self.workers]

    def save_checkpoint(self, path: str, step: int) -> None:
        self._call_all("save_checkpoint", path, step)

    def load_checkpoint(self, path: str) -> int:
        outputs = self._call_all("load_checkpoint", path)
        if not outputs:
            return 0
        return int(max(outputs))

    def shutdown(self) -> None:
        if not self.workers:
            return
        try:
            futures = [worker.shutdown.remote() for worker in self.workers]
            done, _ = ray.wait(futures, num_returns=len(futures), timeout=10)
            if done:
                ray.get(done)
        except Exception:
            pass
        finally:
            for worker in self.workers:
                try:
                    ray.kill(worker, no_restart=True)
                except Exception:
                    pass
            self.workers = []

    def _metric_weight(self, data: DataProto) -> float:
        if "response_mask" not in data.batch:
            return float(max(len(data), 1))
        return float(max(data.batch["response_mask"].sum().item(), 1.0))

    def _reduce_metrics(self, metrics_list: list[dict[str, float]], weights: list[float] | None = None) -> dict[str, float]:
        if not metrics_list:
            return {}
        weights = weights or [1.0 for _ in metrics_list]
        weight_sum = sum(weights) or float(len(metrics_list))
        metrics: dict[str, float] = {}
        keys = set().union(*(output.keys() for output in metrics_list))
        for key in keys:
            if key == "grad_norm":
                metrics[key] = float(sum(output.get(key, 0.0) for output in metrics_list) / len(metrics_list))
            else:
                metrics[key] = float(
                    sum(output.get(key, 0.0) * weight for output, weight in zip(metrics_list, weights, strict=True))
                    / weight_sum
                )
        return metrics
