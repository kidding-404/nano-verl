from __future__ import annotations

import argparse
import hashlib
import importlib.util
import os
import secrets
import sys
from pathlib import Path
from typing import Any

import yaml

from nanoverl.config import Config, DataConfig, ExperimentConfig, SystemConfig
from nanoverl.logging import ExperimentLogger

DEFAULT_ROLLOUT_WAIT_TIMEOUT = 600
_FORCE_SERIAL_REWARD_ATTR = "__nanoverl_force_serial__"
_REWARD_PARALLEL_BACKEND_ATTR = "__nanoverl_parallel_backend__"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run nano-verl training")
    parser.add_argument("--config", type=str, required=True, help="YAML config path")
    return parser


def load_yaml_config(path: str | Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as file:
        config = yaml.safe_load(file) or {}
    if not isinstance(config, dict):
        raise TypeError("Config root must be a mapping")
    return _expand_env_vars(config)


def _expand_env_vars(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _expand_env_vars(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_expand_env_vars(item) for item in value]
    if isinstance(value, str):
        return os.path.expandvars(value)
    return value


def build_logger(experiment_cfg: ExperimentConfig, run_config: dict[str, Any]) -> ExperimentLogger:
    return ExperimentLogger(
        backend=experiment_cfg.logging.backend,
        output_dir=experiment_cfg.logging.output_dir,
        project_name=experiment_cfg.logging.project_name,
        experiment_name=experiment_cfg.logging.experiment_name,
        run_config=run_config,
        api_key=experiment_cfg.logging.api_key,
    )


def load_reward_fn(
    experiment_cfg: ExperimentConfig,
    tokenizer: Any,
    logger: ExperimentLogger | None = None,
) -> Any:
    from nanoverl.reward import RewardManager
    from nanoverl.reward import compute_score as default_compute_score

    custom_cfg = experiment_cfg.reward.custom_reward_function
    score_fn = None
    if custom_cfg.path:
        module_path = Path(custom_cfg.path)
        module_name = f"_nanoverl_custom_reward_{hashlib.sha1(str(module_path).encode()).hexdigest()[:12]}"
        spec = importlib.util.spec_from_file_location(module_name, module_path)
        if spec is None or spec.loader is None:
            raise ImportError(f"Cannot load reward module from {module_path}")
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        spec.loader.exec_module(module)
        score_fn = getattr(module, custom_cfg.name, None)
        if score_fn is None:
            raise AttributeError(f"Reward module {module_path} has no function {custom_cfg.name}")
    reward_fn = score_fn or default_compute_score
    reward_kwargs = dict(experiment_cfg.reward.kwargs)
    overlong_buffer_cfg = reward_kwargs.get("overlong_buffer_cfg")
    num_workers = int(reward_kwargs.get("num_workers", 1))
    parallel_backend = str(reward_kwargs.get("parallel_backend", "thread"))
    parallel_backend = str(getattr(reward_fn, _REWARD_PARALLEL_BACKEND_ATTR, parallel_backend))
    if bool(getattr(reward_fn, _FORCE_SERIAL_REWARD_ATTR, False)):
        num_workers = 1
        parallel_backend = "thread"
    return RewardManager(
        tokenizer=tokenizer,
        reward_fn=reward_fn,
        num_examine=int(reward_kwargs.get("num_examine", 0)),
        num_workers=num_workers,
        parallel_backend=parallel_backend,
        logger=logger,
        max_resp_len=int(reward_kwargs.get("max_resp_len", experiment_cfg.algorithm.max_new_tokens)),
        overlong_buffer_cfg=overlong_buffer_cfg,
    )


def _validate_assets(data_cfg: DataConfig, system_cfg: SystemConfig) -> None:
    """训练入口只校验资源是否存在，不负责下载。"""
    missing_paths: list[str] = []
    for required_path in [data_cfg.train_files, data_cfg.val_files]:
        if required_path and not Path(required_path).exists():
            missing_paths.append(required_path)
    if missing_paths:
        joined = ", ".join(missing_paths)
        raise FileNotFoundError(
            f"Missing required assets: {joined}. "
            "Run `uv run python scripts/download_assets.py` before training."
        )


def _load_tokenizer(system_cfg: SystemConfig) -> Any:
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        system_cfg.model.path,
        trust_remote_code=system_cfg.model.trust_remote_code,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    return tokenizer


def build_trainer(
    experiment_cfg: ExperimentConfig,
    system_cfg: SystemConfig,
    data_cfg: DataConfig,
    logger: ExperimentLogger,
    tokenizer: Any | None = None,
    reward_fn: Any | None = None,
    rollout_wait_timeout: int = DEFAULT_ROLLOUT_WAIT_TIMEOUT,
) -> Any:
    from nanoverl.actor.actor_manager import ActorManager
    from nanoverl.datasets.parquet_dataset import DataLoader
    from nanoverl.rollout.rollout_manager import RolloutManager
    from nanoverl.trainer.trainer import Trainer

    tokenizer = tokenizer or _load_tokenizer(system_cfg)
    train_dataloader = DataLoader(
        dataset_source=data_cfg.train_files,
        tokenizer=tokenizer,
        prompt_template=data_cfg.prompt_template,
        max_prompt_length=data_cfg.max_prompt_length,
        batch_size=data_cfg.train_batch_size,
        prompt_key=data_cfg.prompt_key,
        apply_chat_template_kwargs=data_cfg.apply_chat_template_kwargs,
        limit=data_cfg.train_max_samples,
        shuffle=data_cfg.shuffle,
        num_workers=data_cfg.num_workers,
        random_sample=data_cfg.random_sample,
        seed=data_cfg.seed,
    ).build()
    actor_mgr = None
    rollout_mgr = None
    try:
        actor_mgr = ActorManager.launch(
            config=system_cfg,
            tokenizer=tokenizer,
            backend_cfg={
                "clip_low": experiment_cfg.algorithm.clip_low,
                "clip_high": experiment_cfg.algorithm.clip_high,
                "clip_ratio_c": experiment_cfg.algorithm.clip_ratio_c,
                "beta": experiment_cfg.algorithm.beta,
                "entropy_coef": experiment_cfg.algorithm.entropy_coef,
            },
        )
        actor_mgr.init_model(system_cfg)
        rollout_mgr = RolloutManager.launch(
            system_cfg,
            tokenizer=tokenizer,
            run_id=secrets.token_hex(8),
            actor_mgr=actor_mgr,
        )
        rollout_mgr.wait_until_ready(timeout=rollout_wait_timeout, interval=1.0)
        rollout_mgr.bind_actor_manager(actor_mgr, system_cfg.sync)
        trainer = Trainer(
            experiment_cfg=experiment_cfg,
            system_cfg=system_cfg,
            data_cfg=data_cfg,
            tokenizer=tokenizer,
            reward_fn=reward_fn,
        )
        trainer.configure_runtime(actor_mgr, rollout_mgr, logger, train_dataloader, sync_mgr=rollout_mgr.sync_mgr)
        return trainer
    except Exception:
        if rollout_mgr is not None:
            rollout_mgr.shutdown()
        if actor_mgr is not None:
            actor_mgr.shutdown()
        raise


def run_training(config_path: str | Path) -> dict[str, Any]:
    root_cfg = Config.from_dict(load_yaml_config(config_path))
    experiment_cfg = root_cfg.experiment
    system_cfg = root_cfg.system
    data_cfg = root_cfg.data
    _validate_assets(data_cfg, system_cfg)
    logger = build_logger(experiment_cfg, root_cfg.asdict())
    print("[launch] mode=single_controller_ray", flush=True)
    trainer: Any | None = None
    try:
        tokenizer = _load_tokenizer(system_cfg)
        reward_fn = load_reward_fn(experiment_cfg, tokenizer, logger=logger)
        trainer = build_trainer(
            experiment_cfg,
            system_cfg,
            data_cfg,
            logger,
            tokenizer=tokenizer,
            reward_fn=reward_fn,
        )
        trainer.fit()
        return {"status": "ok", "steps": experiment_cfg.loop.max_steps}
    finally:
        if trainer is not None:
            if trainer.rollout_mgr is not None:
                trainer.rollout_mgr.shutdown()
            if trainer.actor_mgr is not None:
                trainer.actor_mgr.shutdown()
        else:
            logger.close()


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    run_training(Path(args.config).resolve())


if __name__ == "__main__":
    main()
