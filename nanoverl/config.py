from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


_DTYPE_CHOICES = {"float32", "fp32", "float16", "fp16", "bfloat16", "bf16"}
_REQUIRED_SECTIONS = ("trainer", "algorithm", "data", "model", "actor", "rollout")
_OPTIONAL_SECTIONS = ("weight_sync", "logging", "checkpoint", "reward")
_CONFIG_ALIAS_PATHS = {
    "trainer": ("experiment", "loop"),
    "algorithm": ("experiment", "algorithm"),
    "reward": ("experiment", "reward"),
    "logging": ("experiment", "logging"),
    "checkpoint": ("experiment", "checkpoint"),
    "model": ("system", "model"),
    "actor": ("system", "actor"),
    "rollout": ("system", "rollout"),
    "weight_sync": ("system", "sync"),
    "resources": ("system", "resources"),
}


def _require_positive(name: str, value: int | float) -> None:
    if value <= 0:
        raise ValueError(f"{name} must be positive, got {value}")


def _require_positive_int(name: str, value: int) -> None:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be a positive integer, got {value!r}")
    _require_positive(name, value)


def _require_non_empty(name: str, value: str) -> None:
    if not value:
        raise ValueError(f"{name} must be non-empty")


def _normalize_choice(name: str, value: str, choices: set[str]) -> str:
    normalized = str(value).lower().strip()
    if normalized not in choices:
        expected = ", ".join(sorted(choices))
        raise ValueError(f"{name} must be one of {expected}, got {value}")
    return normalized


def _as_mapping(value: Any, path: str) -> dict[str, Any]:
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise TypeError(f"{path} must be a mapping")
    return dict(value)


def _require_sections(config: dict[str, Any], *sections: str) -> None:
    missing = [section for section in sections if section not in config]
    if missing:
        raise ValueError(f"Config is missing required section(s): {', '.join(missing)}")


def _reject_keys(config: dict[str, Any], path: str, keys: set[str]) -> None:
    present = sorted(set(config) & keys)
    if present:
        joined = ", ".join(f"{path}.{key}" for key in present)
        raise ValueError(f"{joined} has been removed; configure GPU resources under resources instead")


def _section_maps(config: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {section: _as_mapping(config.get(section), section) for section in (*_REQUIRED_SECTIONS, *_OPTIONAL_SECTIONS)}


def _move_alias(config: dict[str, Any], target: str, alias: str, default: Any) -> None:
    config[target] = config.pop(alias, config.get(target, default))


def _deep_merge_config(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for key, value in override.items():
        if isinstance(merged.get(key), dict) and isinstance(value, dict):
            merged[key] = _deep_merge_config(merged[key], value)
        else:
            merged[key] = value
    return merged


def _flatten_config_layers(config: dict[str, Any]) -> dict[str, Any]:
    if "base" not in config and "advanced" not in config:
        return dict(config)

    unexpected = sorted(set(config) - {"base", "advanced"})
    if unexpected:
        raise ValueError(
            "Layered config root only supports base and advanced sections; "
            f"move root section(s) under base or advanced: {', '.join(unexpected)}"
        )

    base = _as_mapping(config.get("base"), "base")
    advanced = _as_mapping(config.get("advanced"), "advanced")
    return _deep_merge_config(base, advanced)


def _validate_all(*sections: Any) -> None:
    for section in sections:
        validate = getattr(section, "validate", None)
        if validate is not None:
            validate()


def validate_loop_sync_boundary(experiment: "ExperimentConfig", system: "SystemConfig") -> None:
    loop_mode = str(experiment.loop.mode).lower().strip()
    expected: dict[str, tuple[str, str]] = {
        "sync": ("hybrid", "naive"),
        "one_step_off": ("standalone", "nccl"),
    }
    if loop_mode not in expected:
        return
    rollout_mode, sync_backend = expected[loop_mode]
    actual = (system.rollout.mode, system.sync.backend)
    if actual != (rollout_mode, sync_backend):
        raise ValueError(
            f"trainer.mode={loop_mode!r} requires "
            f"rollout.mode={rollout_mode!r} and weight_sync.backend={sync_backend!r}, "
            f"got rollout.mode={actual[0]!r} and weight_sync.backend={actual[1]!r}"
        )


@dataclass
class TrainerConfig:
    total_steps: int = 100
    seed: int = 42
    log_freq: int = 1
    mode: str = "sync"

    @property
    def max_steps(self) -> int:
        return self.total_steps

    @max_steps.setter
    def max_steps(self, value: int) -> None:
        self.total_steps = int(value)

    def validate(self) -> None:
        _require_positive("trainer.total_steps", self.total_steps)
        _require_positive("trainer.log_freq", self.log_freq)
        self.mode = _normalize_choice("trainer.mode", self.mode, {"sync", "one_step_off"})


@dataclass
class RolloutCorrectionConfig:
    bypass_mode: bool = False

    def validate(self) -> None:
        if not isinstance(self.bypass_mode, bool):
            raise TypeError("algorithm.rollout_correction.bypass_mode must be a bool")


@dataclass
class AlgorithmConfig:
    name: str = "grpo"
    num_generations: int = 4
    clip_low: float = 0.2
    clip_high: float = 0.2
    clip_ratio_c: float = 3.0
    beta: float = 0.0
    entropy_coef: float = 0.0
    temperature: float = 0.8
    top_p: float = 0.95
    max_new_tokens: int = 64
    epsilon: float = 1e-6
    rollout_correction: RolloutCorrectionConfig = field(default_factory=RolloutCorrectionConfig)

    @classmethod
    def from_dict(cls, config: dict[str, Any]) -> "AlgorithmConfig":
        algorithm = dict(config)
        rollout_correction = RolloutCorrectionConfig(
            **_as_mapping(
                algorithm.pop("rollout_correction", None),
                "algorithm.rollout_correction",
            )
        )
        return cls(**algorithm, rollout_correction=rollout_correction)

    def validate(self) -> None:
        self.name = _normalize_choice("algorithm.adv_estimator", self.name, {"grpo"})
        _require_positive("rollout.n", self.num_generations)
        _require_positive("data.max_response_length", self.max_new_tokens)
        if self.clip_ratio_c <= 1.0:
            raise ValueError(f"algorithm.clip_ratio_c must be > 1.0, got {self.clip_ratio_c}")
        self.rollout_correction.validate()


@dataclass
class CustomRewardFunctionConfig:
    path: str | None = None
    name: str = "compute_score"


@dataclass
class RewardConfig:
    custom_reward_function: CustomRewardFunctionConfig = field(default_factory=CustomRewardFunctionConfig)
    kwargs: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, config: dict[str, Any]) -> "RewardConfig":
        kwargs = config.get("kwargs", {})
        if kwargs is None:
            kwargs = {}
        if not isinstance(kwargs, dict):
            raise TypeError("reward.kwargs must be a mapping")
        return cls(
            custom_reward_function=CustomRewardFunctionConfig(
                **_as_mapping(
                    config.get("custom_reward_function"),
                    "reward.custom_reward_function",
                )
            ),
            kwargs=dict(kwargs),
        )


@dataclass
class LoggingConfig:
    backend: str = "console"
    output_dir: str = "outputs"
    project_name: str = "nanoverl"
    experiment_name: str = "run"
    api_key: str | None = None

    def validate(self) -> None:
        self.backend = _normalize_choice(
            "logging.logger",
            self.backend or "console",
            {"console", "swanlab", "wandb"},
        )


@dataclass
class CheckpointConfig:
    dir: str = "checkpoints/default"
    save_freq: int = 25
    resume: bool = False

    def validate(self) -> None:
        if self.save_freq < 0:
            raise ValueError(f"checkpoint.save_freq must be non-negative, got {self.save_freq}")
        _require_non_empty("checkpoint.dir", self.dir)


@dataclass
class ExperimentConfig:
    loop: TrainerConfig = field(default_factory=TrainerConfig)
    algorithm: AlgorithmConfig = field(default_factory=AlgorithmConfig)
    reward: RewardConfig = field(default_factory=RewardConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    checkpoint: CheckpointConfig = field(default_factory=CheckpointConfig)

    def validate(self) -> None:
        _validate_all(self.loop, self.algorithm, self.reward, self.logging, self.checkpoint)


@dataclass
class ModelConfig:
    path: str = "model/Qwen2.5-0.5B-Instruct"
    trust_remote_code: bool = False
    enable_gradient_checkpointing: bool = True
    use_remove_padding: bool = False
    max_length: int = 1024

    def validate(self) -> None:
        _require_non_empty("model.path", self.path)


@dataclass
class RoleResourcesConfig:
    nodes: int = 1
    gpus_per_node: int = 1

    @classmethod
    def from_dict(cls, config: dict[str, Any] | None, path: str) -> "RoleResourcesConfig":
        return cls(**_as_mapping(config, path))

    def validate(self, path: str) -> None:
        _require_positive_int(f"{path}.nodes", self.nodes)
        _require_positive_int(f"{path}.gpus_per_node", self.gpus_per_node)


@dataclass
class ResourcesConfig:
    nodes: int = 1
    gpus_per_node: int = 1
    actor: RoleResourcesConfig | None = None
    rollout: RoleResourcesConfig | None = None
    actor_nodes: int = field(init=False, default=1)
    actor_gpus_per_node: int = field(init=False, default=1)
    actor_world_size: int = field(init=False, default=1)
    rollout_nodes: int = field(init=False, default=1)
    rollout_gpus_per_node: int = field(init=False, default=1)
    rollout_servers_per_node: int = field(init=False, default=1)
    rollout_world_size: int = field(init=False, default=1)

    @classmethod
    def from_dict(cls, config: dict[str, Any]) -> "ResourcesConfig":
        resources = dict(config)
        if "actor" in resources:
            resources["actor"] = RoleResourcesConfig.from_dict(resources["actor"], "resources.actor")
        if "rollout" in resources:
            resources["rollout"] = RoleResourcesConfig.from_dict(resources["rollout"], "resources.rollout")
        return cls(**resources)

    def validate_base(self) -> None:
        _require_positive_int("resources.nodes", self.nodes)
        _require_positive_int("resources.gpus_per_node", self.gpus_per_node)

    def resolve(self, rollout_mode: str, tensor_parallel_size: int) -> None:
        self.validate_base()
        _require_positive_int("rollout.tensor_parallel_size", tensor_parallel_size)
        mode = _normalize_choice("rollout.mode", rollout_mode, {"hybrid", "standalone"})

        if mode == "hybrid":
            if self.actor is not None or self.rollout is not None:
                raise ValueError(
                    "resources.actor and resources.rollout are not allowed when rollout.mode='hybrid'; "
                    "hybrid uses all GPUs from resources"
                )
            self.actor_nodes = self.nodes
            self.actor_gpus_per_node = self.gpus_per_node
            self.rollout_nodes = self.nodes
            self.rollout_gpus_per_node = self.gpus_per_node
        else:
            if self.actor is None or self.rollout is None:
                raise ValueError(
                    "rollout.mode='standalone' requires resources.actor and resources.rollout"
                )
            self.actor.validate("resources.actor")
            self.rollout.validate("resources.rollout")
            if self.actor.nodes != self.nodes or self.rollout.nodes != self.nodes:
                raise ValueError(
                    "standalone resources require resources.actor.nodes == "
                    "resources.rollout.nodes == resources.nodes"
                )
            total_gpus = self.actor.gpus_per_node + self.rollout.gpus_per_node
            if total_gpus > self.gpus_per_node:
                raise ValueError(
                    "standalone resources require resources.actor.gpus_per_node + "
                    "resources.rollout.gpus_per_node <= resources.gpus_per_node; "
                    f"got {total_gpus} > {self.gpus_per_node}"
                )
            self.actor_nodes = self.actor.nodes
            self.actor_gpus_per_node = self.actor.gpus_per_node
            self.rollout_nodes = self.rollout.nodes
            self.rollout_gpus_per_node = self.rollout.gpus_per_node

        if self.rollout_gpus_per_node % tensor_parallel_size != 0:
            raise ValueError(
                "rollout per-node GPU count must be divisible by rollout.tensor_parallel_size; "
                f"got {self.rollout_gpus_per_node} and {tensor_parallel_size}"
            )

        self.actor_world_size = self.actor_nodes * self.actor_gpus_per_node
        self.rollout_servers_per_node = self.rollout_gpus_per_node // tensor_parallel_size
        self.rollout_world_size = self.rollout_nodes * self.rollout_servers_per_node


@dataclass
class FSDPConfig:
    optimizer_offload: bool = False
    model_dtype: str | None = None
    mix_precision: str = "bfloat16"
    reshard_after_forward: bool = True
    fsdp_size: int = -1

    def validate(self) -> None:
        if self.fsdp_size == 0 or self.fsdp_size < -1:
            raise ValueError(f"actor.fsdp.fsdp_size must be -1 or positive, got {self.fsdp_size}")
        if self.model_dtype is not None:
            self.model_dtype = _normalize_choice("actor.fsdp.model_dtype", self.model_dtype, _DTYPE_CHOICES)
        self.mix_precision = _normalize_choice("actor.fsdp.mix_precision", self.mix_precision, _DTYPE_CHOICES)


@dataclass
class ActorConfig:
    lr: float = 1e-5
    weight_decay: float = 0.01
    max_grad_norm: float = 1.0
    use_fsdp: bool = True
    distributed_init_timeout_seconds: int = 1800
    ray_num_cpus_per_worker: float = 1.0
    ppo_mini_batch_size: int = 256
    ppo_micro_batch_size_per_gpu: int = 16
    use_dynamic_bsz: bool = False
    ppo_max_token_len_per_gpu: int | None = None
    clear_cache_after_update: bool = False
    skip_zero_response_update: bool = True
    fsdp_config: FSDPConfig = field(default_factory=FSDPConfig)

    @classmethod
    def from_dict(cls, config: dict[str, Any]) -> "ActorConfig":
        actor = dict(config)
        optim = _as_mapping(actor.pop("optim", None), "actor.optim")
        actor.update(optim)
        if "fsdp" in actor:
            actor["fsdp_config"] = actor.pop("fsdp")
        fsdp_config = _as_mapping(actor.get("fsdp_config"), "actor.fsdp")
        actor["fsdp_config"] = FSDPConfig(**fsdp_config)
        return cls(**actor)

    def validate(self) -> None:
        _require_positive("actor.optim.lr", self.lr)
        _require_positive("actor.max_grad_norm", self.max_grad_norm)
        _require_positive("actor.distributed_init_timeout_seconds", self.distributed_init_timeout_seconds)
        _require_positive("actor.ray_num_cpus_per_worker", self.ray_num_cpus_per_worker)
        _require_positive("actor.ppo_mini_batch_size", self.ppo_mini_batch_size)
        _require_positive("actor.ppo_micro_batch_size_per_gpu", self.ppo_micro_batch_size_per_gpu)
        if self.ppo_max_token_len_per_gpu is not None:
            _require_positive("actor.ppo_max_token_len_per_gpu", self.ppo_max_token_len_per_gpu)
        self.fsdp_config.validate()


@dataclass
class RolloutConfig:
    backend: str = "vllm"
    mode: str = "hybrid"
    tensor_parallel_size: int = 1
    seed: int = 0
    engine_kwargs: dict[str, Any] = field(default_factory=dict)
    temperature: float = 0.8
    top_p: float = 0.95
    max_new_tokens: int = 64
    batch_size: int = 32
    max_num_seqs: int = 256
    max_num_batched_tokens: int = 8192
    max_model_len: int = 8192
    enable_chunked_prefill: bool = True
    enable_prefix_caching: bool = True
    logprobs_mode: str = "processed_logprobs"
    free_cache_engine: bool = True
    reset_prefix_cache_on_abort: bool = True
    log_prob_micro_batch_size_per_gpu: int = 16
    log_prob_use_dynamic_bsz: bool = False
    log_prob_max_token_len_per_gpu: int | None = None
    gpu_memory_utilization: float | None = None
    sleep_level: int = 2
    ray_address: str | None = None
    ray_namespace: str = "nanoverl"
    ray_num_cpus_per_server: float = 1.0
    weight_sync_bucket_mb: int = 16
    weight_sync_transport: str = "auto"
    weight_sync_defer_cache_clear: bool = True
    model_path: str = ""
    dtype: str = "bfloat16"
    trust_remote_code: bool = False
    sync_bucket_size_mb: int = 16
    sync_group_name: str = "default"
    sync_rebuild_group: bool = False

    def validate(self) -> None:
        self.backend = _normalize_choice("rollout.name", self.backend, {"vllm"})
        self.mode = _normalize_choice("rollout.mode", self.mode, {"hybrid", "standalone"})
        _require_positive_int("rollout.tensor_parallel_size", self.tensor_parallel_size)
        self.seed = int(self.seed)
        if self.seed < 0:
            raise ValueError(f"rollout.seed must be non-negative, got {self.seed}")
        _require_positive("data.max_response_length", self.max_new_tokens)
        _require_positive("rollout.batch_size", self.batch_size)
        _require_positive("rollout.max_num_seqs", self.max_num_seqs)
        _require_positive("rollout.max_num_batched_tokens", self.max_num_batched_tokens)
        _require_positive("rollout.max_model_len", self.max_model_len)
        _require_non_empty("rollout.logprobs_mode", self.logprobs_mode)
        _require_positive("rollout.log_prob_micro_batch_size_per_gpu", self.log_prob_micro_batch_size_per_gpu)
        if self.log_prob_max_token_len_per_gpu is not None:
            _require_positive("rollout.log_prob_max_token_len_per_gpu", self.log_prob_max_token_len_per_gpu)
        _require_positive("rollout.ray_num_cpus_per_server", self.ray_num_cpus_per_server)
        if self.sleep_level not in {1, 2}:
            raise ValueError(f"rollout.sleep_level must be 1 or 2, got {self.sleep_level}")
        if self.gpu_memory_utilization is not None and not 0 < self.gpu_memory_utilization <= 1:
            raise ValueError(
                "rollout.gpu_memory_utilization must be in (0, 1], "
                f"got {self.gpu_memory_utilization}"
            )
        _require_non_empty("rollout.ray_namespace", self.ray_namespace)
        _require_positive("weight_sync.bucket_size_mb", self.weight_sync_bucket_mb)
        _require_positive("weight_sync.bucket_size_mb", self.sync_bucket_size_mb)
        _require_non_empty("weight_sync.group_name", self.sync_group_name)
        self.weight_sync_transport = _normalize_choice(
            "weight_sync.transport",
            self.weight_sync_transport,
            {"auto", "cuda_ipc", "shared_memory"},
        )
        self.dtype = _normalize_choice("rollout.dtype", self.dtype, _DTYPE_CHOICES)


@dataclass
class SyncConfig:
    backend: str = "naive"
    bucket_size_mb: int = 16
    group_name: str = "default"
    rebuild_group: bool = False
    transport: str = "auto"

    def validate(self) -> None:
        self.backend = _normalize_choice("weight_sync.backend", self.backend, {"naive", "nccl"})
        self.transport = _normalize_choice(
            "weight_sync.transport",
            self.transport,
            {"auto", "cuda_ipc", "shared_memory"},
        )
        _require_positive("weight_sync.bucket_size_mb", self.bucket_size_mb)
        _require_non_empty("weight_sync.group_name", self.group_name)


@dataclass
class SystemConfig:
    model: ModelConfig = field(default_factory=ModelConfig)
    resources: ResourcesConfig = field(default_factory=ResourcesConfig)
    actor: ActorConfig = field(default_factory=ActorConfig)
    rollout: RolloutConfig = field(default_factory=RolloutConfig)
    sync: SyncConfig = field(default_factory=SyncConfig)

    def validate(self) -> None:
        _validate_all(self.model, self.actor, self.rollout, self.sync)
        self.resources.resolve(self.rollout.mode, self.rollout.tensor_parallel_size)


@dataclass
class DataConfig:
    train_files: str = "data/gsm8k/train.parquet"
    val_files: str | None = "data/gsm8k/test.parquet"
    train_max_samples: int | None = None
    prompt_key: str = "question"
    prompt_template: str = "{question}"
    apply_chat_template_kwargs: dict[str, Any] = field(default_factory=dict)
    train_batch_size: int = 2
    shuffle: bool = True
    random_sample: bool = False
    seed: int = 42
    num_workers: int = 0
    max_prompt_length: int = 256
    max_response_length: int = 64

    def validate(self) -> None:
        _require_non_empty("data.train_files", self.train_files)
        _require_non_empty("data.prompt_template", self.prompt_template)
        _require_non_empty("data.prompt_key", self.prompt_key)
        self.apply_chat_template_kwargs = _as_mapping(
            self.apply_chat_template_kwargs,
            "data.apply_chat_template_kwargs",
        )
        _require_positive("data.train_batch_size", self.train_batch_size)
        _require_positive("data.max_prompt_length", self.max_prompt_length)
        _require_positive("data.max_response_length", self.max_response_length)
        if self.train_max_samples is not None:
            _require_positive("data.train_max_samples", self.train_max_samples)
        if f"{{{self.prompt_key}}}" not in self.prompt_template and "{question}" not in self.prompt_template:
            raise ValueError("data.prompt_template must contain data.prompt_key or {question}")


@dataclass
class Config:
    experiment: ExperimentConfig = field(default_factory=ExperimentConfig)
    system: SystemConfig = field(default_factory=SystemConfig)
    data: DataConfig = field(default_factory=DataConfig)

    def __getattr__(self, name: str) -> Any:
        if name not in _CONFIG_ALIAS_PATHS:
            raise AttributeError(name)
        value: Any = self
        for part in _CONFIG_ALIAS_PATHS[name]:
            value = getattr(value, part)
        return value

    @classmethod
    def from_dict(cls, config: dict[str, Any]) -> "Config":
        if not isinstance(config, dict):
            raise TypeError("Config root must be a mapping")
        config = _flatten_config_layers(config)
        _require_sections(config, *_REQUIRED_SECTIONS, "resources")
        if "distributed" in config:
            raise ValueError("distributed has been removed; configure GPU resources under resources instead")

        sections = _section_maps(config)
        trainer, algorithm, data, model, actor, rollout = (sections[name] for name in _REQUIRED_SECTIONS)
        weight_sync, logging, checkpoint, reward = (sections[name] for name in _OPTIONAL_SECTIONS)

        resources_cfg = ResourcesConfig.from_dict(_as_mapping(config.get("resources"), "resources"))
        _reject_keys(actor, "actor", {"world_size", "dp_size", "ray_num_gpus_per_worker"})
        _reject_keys(rollout, "rollout", {"ray_num_gpus_per_server", "tensor_model_parallel_size"})

        data_cfg = DataConfig(**data)
        _move_alias(algorithm, "name", "adv_estimator", "grpo")
        _move_alias(algorithm, "clip_low", "clip_ratio_low", 0.2)
        _move_alias(algorithm, "clip_high", "clip_ratio_high", 0.2)
        algorithm.update(
            num_generations=rollout.get("n", 4),
            temperature=rollout.get("temperature", 0.8),
            top_p=rollout.get("top_p", 0.95),
            max_new_tokens=data_cfg.max_response_length,
        )
        _move_alias(logging, "backend", "logger", "console")
        experiment_cfg = ExperimentConfig(
            loop=TrainerConfig(**trainer),
            algorithm=AlgorithmConfig.from_dict(algorithm),
            reward=RewardConfig.from_dict(reward),
            logging=LoggingConfig(**logging),
            checkpoint=CheckpointConfig(**checkpoint),
        )

        model.setdefault("max_length", max(1024, data_cfg.max_prompt_length + data_cfg.max_response_length))
        model_cfg = ModelConfig(**model)

        _move_alias(rollout, "backend", "name", "vllm")
        engine_kwargs = _as_mapping(rollout.get("engine_kwargs"), "rollout.engine_kwargs")
        vllm_kwargs = _as_mapping(engine_kwargs.get("vllm"), "rollout.engine_kwargs.vllm")
        if "seed" not in rollout and "seed" in vllm_kwargs:
            rollout["seed"] = vllm_kwargs["seed"]
        rollout.pop("n", None)
        rollout["max_new_tokens"] = data_cfg.max_response_length
        rollout.setdefault("max_num_batched_tokens", data_cfg.max_prompt_length + data_cfg.max_response_length)
        rollout.setdefault("max_model_len", model_cfg.max_length)
        rollout.setdefault("model_path", model_cfg.path)
        rollout.setdefault("trust_remote_code", model_cfg.trust_remote_code)

        rollout.setdefault("weight_sync_bucket_mb", weight_sync.get("bucket_size_mb", 16))
        rollout.setdefault("weight_sync_transport", weight_sync.get("transport", "auto"))
        rollout.setdefault("sync_bucket_size_mb", weight_sync.get("bucket_size_mb", 16))
        rollout.setdefault("sync_group_name", weight_sync.get("group_name", "default"))
        rollout.setdefault("sync_rebuild_group", weight_sync.get("rebuild_group", False))

        root = cls(
            experiment=experiment_cfg,
            system=SystemConfig(
                model=model_cfg,
                resources=resources_cfg,
                actor=ActorConfig.from_dict(actor),
                rollout=RolloutConfig(**rollout),
                sync=SyncConfig(**weight_sync),
            ),
            data=data_cfg,
        )
        root.validate()
        return root

    def validate(self) -> None:
        _validate_all(self.experiment, self.system, self.data)
        validate_loop_sync_boundary(self.experiment, self.system)

    def asdict(self) -> dict[str, Any]:
        return asdict(self)
