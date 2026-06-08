from __future__ import annotations

import asyncio
import inspect
import json
import random
import threading
import time
from concurrent.futures import Future
from dataclasses import is_dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch

from nanoverl import algorithms
from nanoverl.config import DataConfig, ExperimentConfig, SystemConfig, validate_loop_sync_boundary
from nanoverl.data import DataProto


class Trainer:
    def __init__(
        self,
        experiment_cfg: ExperimentConfig,
        system_cfg: SystemConfig,
        data_cfg: DataConfig,
        tokenizer: Any | None = None,
        reward_fn: Callable[[DataProto], DataProto] | None = None,
    ) -> None:
        self.experiment_cfg = experiment_cfg
        self.system_cfg = system_cfg
        self.data_cfg = data_cfg
        self.tokenizer = tokenizer
        self.reward_fn = reward_fn
        self.actor_mgr: Any | None = None
        self.rollout_mgr: Any | None = None
        self.sync_mgr: Any | None = None
        self.logger: Any | None = None
        self.train_dataloader: Any | None = None
        self.global_step = 0

    def configure_runtime(
        self,
        actor_mgr: Any,
        rollout_mgr: Any,
        logger: Any,
        train_dataloader: Any,
        sync_mgr: Any | None = None,
    ) -> None:
        self.actor_mgr = actor_mgr
        self.rollout_mgr = rollout_mgr
        self.sync_mgr = sync_mgr
        self.logger = logger
        self.train_dataloader = train_dataloader

    def _set_seed(self) -> None:
        seed = self.experiment_cfg.loop.seed
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    def _ensure_ready(self) -> None:
        if any(obj is None for obj in [self.actor_mgr, self.rollout_mgr, self.logger, self.train_dataloader]):
            raise RuntimeError("Trainer runtime is not fully configured")
        if self.reward_fn is None:
            raise RuntimeError("reward_fn is required")

    def prepare_gen_batch(self, batch: DataProto) -> DataProto:
        prepared = algorithms.prepare_gen_batch(batch, self.experiment_cfg)
        metadata = {
            key: value
            for key, value in batch.non_tensor_batch.items()
            if key not in prepared.non_tensor_batch
        }
        if metadata:
            prepared = prepared.union(DataProto.from_dict(None, metadata))
        return prepared

    def repeat_for_rollout(self, batch: DataProto, n: int, step: int | None = None) -> DataProto:
        original_size = len(batch)
        repeated = batch.repeat(n, interleave=True)
        group_ids = torch.arange(original_size, dtype=torch.long).repeat_interleave(n)
        request_step = self.global_step if step is None else step
        request_ids = [f"step{request_step:06d}_g{gid}_k{kid}" for gid in range(original_size) for kid in range(n)]
        return repeated.union(DataProto.from_dict({"group_ids": group_ids}, {"request_id": request_ids}))

    def build_update_batch(self, batch: DataProto, rollout_output: DataProto) -> DataProto:
        return algorithms.build_update_batch(batch, rollout_output, self.experiment_cfg)

    def _sampling_params(self) -> dict[str, Any]:
        params = {
            "temperature": float(self.experiment_cfg.algorithm.temperature),
            "top_p": float(self.experiment_cfg.algorithm.top_p),
            "top_k": -1,
            "repetition_penalty": 1.0,
            "max_tokens": int(self.experiment_cfg.algorithm.max_new_tokens),
            "detokenize": False,
            "skip_special_tokens": False,
        }
        if self._use_rollout_log_probs():
            params["logprobs"] = 1
        return params

    def _use_rollout_log_probs(self) -> bool:
        return self._bypass_old_log_prob_recompute()

    def _bypass_old_log_prob_recompute(self) -> bool:
        return (
            self.experiment_cfg.loop.mode == "one_step_off"
            and self.experiment_cfg.algorithm.rollout_correction.bypass_mode
        )

    @staticmethod
    def _normalize_stop_reason(reason: Any) -> str | None:
        if reason is None:
            return None
        text = str(reason).lower().strip()
        return "aborted" if text in {"abort", "aborted"} else str(reason)

    @classmethod
    def _is_aborted_stop_reason(cls, reason: Any) -> bool:
        return cls._normalize_stop_reason(reason) == "aborted"

    def _actor_update_mini_batch_size(self) -> int:
        return int(self.system_cfg.actor.ppo_mini_batch_size) * max(
            int(self.experiment_cfg.algorithm.num_generations),
            1,
        )

    @staticmethod
    def _trim_prompt_ids(prompt_ids: torch.Tensor, prompt_attention_mask: torch.Tensor | None) -> list[list[int]]:
        if prompt_attention_mask is None:
            return [list(row) for row in prompt_ids.tolist()]
        trimmed: list[list[int]] = []
        for row, mask in zip(prompt_ids, prompt_attention_mask, strict=True):
            prompt_len = int(mask.sum().item())
            trimmed.append(row[:prompt_len].tolist())
        return trimmed

    def _rollout_request_batch(self, batch: DataProto) -> dict[str, Any]:
        request_ids = batch.non_tensor_batch.get("request_id")
        if request_ids is None:
            request_ids = [f"step{self.global_step:06d}_r{idx}" for idx in range(len(batch))]
        prompt_ids = self._trim_prompt_ids(
            batch.batch["prompt_ids"],
            batch.batch.get("prompt_attention_mask"),
        )
        sampling_params = [self._sampling_params() for _ in range(len(batch))]
        return self._coalesce_rollout_requests(batch, list(request_ids), prompt_ids, sampling_params)

    def _coalesce_rollout_requests(
        self,
        batch: DataProto,
        request_ids: list[Any],
        prompt_ids: list[list[int]],
        sampling_params: list[dict[str, Any]],
    ) -> dict[str, Any]:
        n = int(self.experiment_cfg.algorithm.num_generations)
        group_ids = batch.batch.get("group_ids")
        if n <= 1 or group_ids is None or len(batch) != len(prompt_ids):
            return {
                "request_ids": [str(request_id) for request_id in request_ids],
                "prompt_ids": prompt_ids,
                "sampling_params": sampling_params,
            }

        groups = group_ids.detach().cpu().tolist()
        coalesced_request_ids: list[str] = []
        coalesced_prompt_ids: list[list[int]] = []
        coalesced_sampling_params: list[dict[str, Any]] = []
        index = 0
        while index < len(prompt_ids):
            group_id = groups[index]
            next_index = index + 1
            while next_index < len(prompt_ids) and groups[next_index] == group_id:
                next_index += 1

            group_prompt_ids = prompt_ids[index:next_index]
            group_sampling_params = sampling_params[index:next_index]
            if (
                len(group_prompt_ids) > 1
                and all(value == group_prompt_ids[0] for value in group_prompt_ids)
                and all(value == group_sampling_params[0] for value in group_sampling_params)
            ):
                params = dict(group_sampling_params[0])
                params["n"] = len(group_prompt_ids)
                coalesced_request_ids.append(str(request_ids[index]))
                coalesced_prompt_ids.append(group_prompt_ids[0])
                coalesced_sampling_params.append(params)
            else:
                coalesced_request_ids.extend(str(request_id) for request_id in request_ids[index:next_index])
                coalesced_prompt_ids.extend(group_prompt_ids)
                coalesced_sampling_params.extend(group_sampling_params)
            index = next_index

        return {
            "request_ids": coalesced_request_ids,
            "prompt_ids": coalesced_prompt_ids,
            "sampling_params": coalesced_sampling_params,
        }

    def _as_mapping(self, output: Any) -> dict[str, Any]:
        if isinstance(output, dict):
            return output
        if is_dataclass(output):
            return output.__dict__
        if hasattr(output, "__dict__"):
            return output.__dict__
        raise TypeError(f"Unsupported rollout output type: {type(output).__name__}")

    def _decode_responses(self, responses: list[list[int]]) -> list[str]:
        if self.tokenizer is None:
            return ["" for _ in responses]
        return [
            str(text)
            for text in self.tokenizer.batch_decode(
                responses,
                skip_special_tokens=True,
            )
        ]

    def _rollout_output_to_proto(
        self,
        outputs: Any,
        policy_version: int | None = None,
        include_log_probs: bool = True,
    ) -> DataProto:
        if isinstance(outputs, DataProto):
            return outputs
        items = [self._as_mapping(item) for item in outputs]
        stop_reasons = [self._normalize_stop_reason(item.get("stop_reason")) for item in items]
        response_ids = [
            []
            if self._is_aborted_stop_reason(reason)
            else list(item.get("token_ids", item.get("responses", [])) or [])
            for item, reason in zip(items, stop_reasons, strict=True)
        ]
        response_tensors = [torch.tensor(ids, dtype=torch.long) for ids in response_ids]
        if response_tensors:
            responses = torch.nn.utils.rnn.pad_sequence(
                response_tensors,
                batch_first=True,
                padding_value=0,
            )
            response_mask = torch.nn.utils.rnn.pad_sequence(
                [torch.ones_like(tensor, dtype=torch.long) for tensor in response_tensors],
                batch_first=True,
                padding_value=0,
            )
        else:
            responses = torch.empty(0, 0, dtype=torch.long)
            response_mask = torch.empty(0, 0, dtype=torch.long)

        tensors: dict[str, torch.Tensor] = {
            "responses": responses,
            "response_mask": response_mask,
        }
        log_probs = [
            [] if self._is_aborted_stop_reason(reason) and item.get("log_probs") is not None else item.get("log_probs")
            for item, reason in zip(items, stop_reasons, strict=True)
        ]
        if include_log_probs and log_probs and all(value is not None for value in log_probs):
            log_prob_tensors = [torch.tensor(value, dtype=torch.float32) for value in log_probs]
            tensors["rollout_log_probs"] = torch.nn.utils.rnn.pad_sequence(
                log_prob_tensors,
                batch_first=True,
                padding_value=0.0,
            )

        versions = [
            version if version is not None else policy_version
            for version in (item.get("model_version") for item in items)
        ]
        non_tensors = {
            "stop_reason": stop_reasons,
            "policy_version": [int(version) if version is not None else -1 for version in versions],
            "response_text": self._decode_responses(response_ids),
        }
        return DataProto.from_dict(tensors, non_tensors)

    def _iter_batches_by_size(self, batch: DataProto, batch_size: int | None) -> list[DataProto]:
        if batch_size is None or batch_size <= 0 or len(batch) <= batch_size:
            return [batch]
        return [batch[start : start + batch_size] for start in range(0, len(batch), batch_size)]

    def _merge_weighted_metrics(
        self,
        metrics_list: list[tuple[dict[str, float], float]],
    ) -> dict[str, float]:
        if not metrics_list:
            return {}
        totals: dict[str, float] = {}
        weight_sum = sum(weight for _, weight in metrics_list) or float(len(metrics_list))
        for metrics, weight in metrics_list:
            for key, value in metrics.items():
                if key == "grad_norm":
                    totals[key] = float(value)
                else:
                    totals[key] = totals.get(key, 0.0) + float(value) * weight
        for key in list(totals):
            if key != "grad_norm":
                totals[key] /= weight_sum
        return totals

    @staticmethod
    def _zero_response_sample(batch: DataProto, count: int) -> DataProto:
        pad = batch[-1].repeat(count, interleave=True)
        for key in (
            "responses",
            "response_mask",
            "advantages",
            "returns",
            "old_log_probs",
            "rollout_log_probs",
            "ref_log_probs",
            "token_level_rewards",
            "entropy",
        ):
            if key in pad.batch:
                pad.batch[key] = torch.zeros_like(pad.batch[key])
        return pad

    def _pad_actor_update_batch(self, batch: DataProto, divisor: int) -> DataProto:
        if divisor <= 1 or len(batch) == 0:
            return batch
        pad_size = (-len(batch)) % divisor
        if pad_size == 0:
            return batch
        return DataProto.concat([batch, self._zero_response_sample(batch, pad_size)])

    def _actor_update_batch(self, batch: DataProto) -> tuple[DataProto | None, int]:
        if not bool(getattr(self.system_cfg.actor, "skip_zero_response_update", True)):
            return batch, 0
        response_mask = batch.batch.get("response_mask")
        if response_mask is None:
            return batch, 0
        valid_mask = response_mask.sum(dim=1) > 0
        valid_indices = valid_mask.nonzero(as_tuple=False).flatten().cpu().tolist()
        skipped = len(batch) - len(valid_indices)
        if not valid_indices:
            return None, skipped
        actor_batch = batch[valid_indices]
        if self.experiment_cfg.algorithm.beta == 0.0 and self.experiment_cfg.algorithm.entropy_coef == 0.0:
            advantages = actor_batch.batch.get("advantages")
            if advantages is not None:
                keep_mask = advantages.float().abs().sum(dim=1) > 0
                keep_indices = keep_mask.nonzero(as_tuple=False).flatten().cpu().tolist()
                if 0 < len(keep_indices) < len(actor_batch):
                    actor_batch = actor_batch[keep_indices]
        actor_batch = self._pad_actor_update_batch(actor_batch, self.system_cfg.distributed.dp_size)
        return actor_batch, skipped

    def _actor_update_is_noop(self, batch: DataProto) -> bool:
        if self.experiment_cfg.algorithm.beta != 0.0 or self.experiment_cfg.algorithm.entropy_coef != 0.0:
            return False
        advantages = batch.batch.get("advantages")
        response_mask = batch.batch.get("response_mask")
        if advantages is None or response_mask is None:
            return False
        valid_advantages = advantages.float()[response_mask.bool()]
        if valid_advantages.numel() == 0:
            return True
        return bool(torch.count_nonzero(valid_advantages).item() == 0)

    def _compute_reward(self, batch: DataProto) -> DataProto:
        return self.reward_fn(batch)

    def compute_advantage(self, batch: DataProto) -> DataProto:
        return algorithms.compute_advantage(batch, self.experiment_cfg)

    def balance_batch(self, batch: DataProto) -> DataProto:
        if self.system_cfg.distributed.dp_size <= 1:
            return batch
        total_len = batch.batch["prompt_attention_mask"].sum(dim=1)
        if "response_mask" in batch.batch:
            total_len = total_len + batch.batch["response_mask"].sum(dim=1)
        partitions = self._balanced_partitions(total_len.tolist(), self.system_cfg.distributed.dp_size)
        order = [index for partition in partitions for index in partition]
        return batch[order]

    @staticmethod
    def _balanced_partitions(lengths: list[int | float], num_partitions: int) -> list[list[int]]:
        if num_partitions <= 1:
            return [list(range(len(lengths)))]
        base_size, extra = divmod(len(lengths), num_partitions)
        capacities = [base_size + (1 if idx < extra else 0) for idx in range(num_partitions)]
        partitions = [[] for _ in range(num_partitions)]
        loads = [0.0 for _ in range(num_partitions)]
        for index in sorted(range(len(lengths)), key=lambda idx: (-float(lengths[idx]), idx)):
            candidates = [idx for idx, capacity in enumerate(capacities) if len(partitions[idx]) < capacity]
            target = min(candidates, key=lambda idx: (loads[idx], len(partitions[idx]), idx))
            partitions[target].append(index)
            loads[target] += float(lengths[index])
        return partitions

    def pad_batch(self, batch: DataProto, divisor: int) -> tuple[DataProto, int]:
        if divisor <= 1 or len(batch) == 0:
            return batch, 0
        pad_size = (-len(batch)) % divisor
        if pad_size == 0:
            return batch, 0
        padded = DataProto.concat([batch, batch[-1].repeat(pad_size, interleave=True)])
        return padded, pad_size

    def unpad_batch(self, batch: DataProto, pad_size: int) -> DataProto:
        if pad_size <= 0:
            return batch
        return batch[: len(batch) - pad_size]

    def prepare_step(self, batch: DataProto, step: int | None = None) -> DataProto:
        repeated = self.repeat_for_rollout(batch, self.experiment_cfg.algorithm.num_generations, step=step)
        return self.prepare_gen_batch(repeated)

    def _maybe_call_rollout(self, method_name: str) -> None:
        method = getattr(self.rollout_mgr, method_name, None)
        if method is not None:
            method()

    @staticmethod
    async def _await_if_needed(value: Any) -> Any:
        if inspect.isawaitable(value):
            return await value
        return value

    async def _maybe_call_rollout_async(self, method_name: str) -> None:
        method = getattr(self.rollout_mgr, method_name, None)
        if method is not None:
            await self._await_if_needed(method())

    def _manage_rollout_lifecycle_per_phase(self) -> bool:
        return not (
            self.experiment_cfg.loop.mode == "one_step_off"
            and self.system_cfg.rollout.mode.lower().strip() == "standalone"
        )

    def _run_rollout_phase(self, batch: DataProto, policy_version: int | None = None) -> DataProto:
        manage_lifecycle = self._manage_rollout_lifecycle_per_phase()
        if manage_lifecycle:
            self._maybe_call_rollout("wake_up")
        try:
            outputs = self.rollout_mgr.generate_sequences(self._rollout_request_batch(batch))
            return self._rollout_output_to_proto(
                outputs,
                policy_version=policy_version,
                include_log_probs=self._use_rollout_log_probs(),
            )
        finally:
            if manage_lifecycle:
                self._maybe_call_rollout("sleep")

    async def _run_rollout_phase_async(self, batch: DataProto, policy_version: int | None = None) -> DataProto:
        manage_lifecycle = self._manage_rollout_lifecycle_per_phase()
        if manage_lifecycle:
            await self._maybe_call_rollout_async("wake_up")
        try:
            outputs = self.rollout_mgr.generate_sequences(self._rollout_request_batch(batch))
            outputs = await self._await_if_needed(outputs)
            return self._rollout_output_to_proto(
                outputs,
                policy_version=policy_version,
                include_log_probs=self._use_rollout_log_probs(),
            )
        finally:
            if manage_lifecycle:
                await self._maybe_call_rollout_async("sleep")

    def _run_rollout_phase_in_thread(self, batch: DataProto, policy_version: int | None = None) -> DataProto:
        return asyncio.run(self._run_rollout_phase_async(batch, policy_version=policy_version))

    def _run_rollout_phase_background(
        self,
        batch: DataProto,
        policy_version: int | None = None,
    ) -> Future[DataProto]:
        future: Future[DataProto] = Future()

        def run() -> None:
            try:
                result = self._run_rollout_phase_in_thread(batch, policy_version=policy_version)
            except BaseException as exc:
                future.set_exception(exc)
            else:
                future.set_result(result)

        threading.Thread(target=run, name="nanoverl-rollout-prefetch", daemon=True).start()
        return future

    async def _run_rollout_phase_background_async(
        self,
        batch: DataProto,
        policy_version: int | None = None,
    ) -> DataProto:
        future = self._run_rollout_phase_background(batch, policy_version=policy_version)
        while not future.done():
            await asyncio.sleep(0.01)
        return future.result()

    def _next_batch(self, data_iter: Any) -> tuple[Any, dict[str, Any]]:
        try:
            return data_iter, next(data_iter)
        except StopIteration:
            data_iter = iter(self.train_dataloader)
            return data_iter, next(data_iter)

    def _next_prepared_batch(self, data_iter: Any, rollout_step: int) -> tuple[Any, DataProto]:
        data_iter, batch_dict = self._next_batch(data_iter)
        batch = DataProto.from_mixed_dict(batch_dict, meta_info={"step": rollout_step})
        return data_iter, self.prepare_step(batch, step=rollout_step)

    def _run_train_phase(self, batch: DataProto) -> tuple[DataProto, dict[str, float]]:
        train_timing: dict[str, float] = {}
        started = time.perf_counter()
        batch = self._compute_reward(batch)
        train_timing["train_reward"] = time.perf_counter() - started
        need_ref_log_prob = algorithms.need_ref_log_prob(self.experiment_cfg)
        kl_metrics: dict[str, float] = {}
        if not need_ref_log_prob:
            started = time.perf_counter()
            batch = self.compute_advantage(batch)
            train_timing["train_advantage"] = time.perf_counter() - started
        started = time.perf_counter()
        batch = self.balance_batch(batch)
        batch, pad_size = self.pad_batch(batch, self.system_cfg.distributed.dp_size)
        train_timing["train_balance_pad"] = time.perf_counter() - started
        started = time.perf_counter()
        actor_update_noop = False if need_ref_log_prob else self._actor_update_is_noop(batch)
        if self._bypass_old_log_prob_recompute() and "rollout_log_probs" in batch.batch:
            batch = batch.union(DataProto.from_dict({"old_log_probs": batch.batch["rollout_log_probs"]}))
        elif actor_update_noop:
            pass
        else:
            batch = batch.union(
                self.actor_mgr.compute_log_prob(
                    batch,
                    micro_batch_size=self.system_cfg.rollout.log_prob_micro_batch_size_per_gpu,
                )
            )
        if need_ref_log_prob:
            batch = batch.union(
                self.actor_mgr.compute_ref_log_prob(
                    batch,
                    micro_batch_size=self.system_cfg.rollout.log_prob_micro_batch_size_per_gpu,
                )
            )
            batch, kl_metrics = algorithms.apply_kl_penalty(batch, self.experiment_cfg.algorithm.beta)
        train_timing["train_old_log_prob"] = time.perf_counter() - started
        if need_ref_log_prob:
            started = time.perf_counter()
            batch = self.unpad_batch(batch, pad_size)
            batch = self.compute_advantage(batch)
            batch, pad_size = self.pad_batch(batch, self.system_cfg.distributed.dp_size)
            train_timing["train_advantage"] = time.perf_counter() - started
        started = time.perf_counter()
        metrics_list: list[tuple[dict[str, float], float]] = []
        skipped_zero_response_samples = 0
        skipped_noop_actor_updates = 0
        for mini_batch in self._iter_batches_by_size(batch, self._actor_update_mini_batch_size()):
            actor_batch, skipped = self._actor_update_batch(mini_batch)
            skipped_zero_response_samples += skipped
            if actor_batch is None:
                continue
            if self._actor_update_is_noop(actor_batch):
                skipped_noop_actor_updates += len(actor_batch)
                continue
            metrics = self.actor_mgr.update_policy(
                actor_batch,
                micro_batch_size=self.system_cfg.actor.ppo_micro_batch_size_per_gpu,
            )
            weight = float(actor_batch.batch["response_mask"].sum().item())
            metrics_list.append((metrics, max(weight, 1.0)))
        train_timing["train_actor_update"] = time.perf_counter() - started
        update_metrics = self._merge_weighted_metrics(metrics_list)
        update_metrics.update(kl_metrics)
        update_metrics["skipped_zero_response_samples"] = float(skipped_zero_response_samples)
        update_metrics["skipped_noop_actor_updates"] = float(skipped_noop_actor_updates)
        if skipped_noop_actor_updates and not metrics_list:
            update_metrics.update(
                {
                    "loss": 0.0,
                    "entropy": 0.0,
                    "approx_kl": 0.0,
                    "clipfrac": 0.0,
                    "grad_norm": 0.0,
                }
            )
        update_metrics.update({f"timing_s/{key}": value for key, value in train_timing.items()})
        batch = self.unpad_batch(batch, pad_size)
        return batch, update_metrics

    def _build_step_metrics(
        self,
        batch: DataProto,
        update_metrics: dict[str, float],
        step: int,
        timing: dict[str, float] | None = None,
    ) -> dict[str, float]:
        rewards = batch.batch["rewards"].float()
        response_mask = batch.batch["response_mask"].float()
        response_len = response_mask.sum(dim=1)
        prompt_mask = batch.batch.get("prompt_attention_mask")
        prompt_len = prompt_mask.float().sum(dim=1) if prompt_mask is not None else None
        total_response_tokens = float(response_mask.sum().item())
        total_tokens = total_response_tokens + (float(prompt_len.sum().item()) if prompt_len is not None else 0.0)
        timing = timing or {}
        step_time = float(timing.get("step", 0.0))
        metrics = {
            "perf/step_s": step_time,
            "perf/wall_step_s": float(timing.get("wall_step", step_time)),
            "perf/rollout_s": float(timing.get("rollout", 0.0)),
            "perf/train_s": float(timing.get("train", 0.0)),
            "perf/sync_s": float(timing.get("sync", 0.0)),
            "perf/rollout_wait_s": float(timing.get("rollout_wait", 0.0)),
            "perf/tokens_per_s": total_tokens / step_time if step_time > 0 else 0.0,
            "reward/mean": float(rewards.mean().item()),
            "reward/std": float(rewards.std(unbiased=False).item()),
            "reward/min": float(rewards.min().item()),
            "reward/max": float(rewards.max().item()),
            "response/len_mean": float(response_len.mean().item()),
            "response/len_std": float(response_len.std(unbiased=False).item()),
            "response/len_min": float(response_len.min().item()),
            "response/len_max": float(response_len.max().item()),
            "policy/entropy": float(update_metrics.get("entropy", 0.0)),
            "policy/kl": float(update_metrics.get("approx_kl", update_metrics.get("reward_kl_penalty", 0.0))),
            "policy/clipfrac": float(update_metrics.get("clipfrac", 0.0)),
            "policy/grad_norm": float(update_metrics.get("grad_norm", 0.0)),
        }
        return metrics

    def _build_trajectory_rows(self, batch: DataProto, step: int) -> list[dict[str, Any]]:
        rewards = batch.batch.get("rewards")
        response_mask = batch.batch.get("response_mask")
        reward_values = rewards.float().cpu().tolist() if rewards is not None else [0.0] * len(batch)
        response_lengths = (
            response_mask.sum(dim=1).cpu().tolist() if response_mask is not None else [0] * len(batch)
        )
        reward_details = batch.non_tensor_batch.get("reward_details", [{} for _ in range(len(batch))])
        request_ids = batch.non_tensor_batch.get("request_id", [None] * len(batch))
        prompt_texts = batch.non_tensor_batch.get("prompt_text", [""] * len(batch))
        response_texts = batch.non_tensor_batch.get("response_text", [""] * len(batch))
        ground_truths = batch.non_tensor_batch.get("ground_truth", [None] * len(batch))
        sample_ids = batch.non_tensor_batch.get("sample_id", [None] * len(batch))
        stop_reasons = batch.non_tensor_batch.get("stop_reason", [None] * len(batch))
        policy_versions = batch.non_tensor_batch.get("policy_version", [None] * len(batch))
        questions = batch.non_tensor_batch.get("question", [None] * len(batch))
        group_ids = batch.batch.get("group_ids")
        group_values = group_ids.cpu().tolist() if group_ids is not None else [None] * len(batch)
        rows: list[dict[str, Any]] = []
        for idx in range(len(batch)):
            rows.append(
                {
                    "step": step,
                    "request_id": request_ids[idx],
                    "sample_id": sample_ids[idx],
                    "group_id": group_values[idx],
                    "question": questions[idx],
                    "prompt": prompt_texts[idx],
                    "response": response_texts[idx],
                    "ground_truth": ground_truths[idx],
                    "reward": reward_values[idx],
                    "response_length": int(response_lengths[idx]),
                    "stop_reason": stop_reasons[idx],
                    "policy_version": policy_versions[idx],
                    "reward_detail": reward_details[idx],
                }
            )
        return rows

    def _maybe_log_trajectories(self, batch: DataProto, step: int, max_steps: int) -> None:
        if self.logger is None:
            return
        freq = self.experiment_cfg.checkpoint.save_freq
        if step != 1 and step != max_steps and (freq <= 0 or step % freq != 0):
            return
        self.logger.log_trajectories(step, self._build_trajectory_rows(batch, step))

    def save_checkpoint(self, step: int) -> None:
        checkpoint_dir = Path(self.experiment_cfg.checkpoint.dir)
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        step_dir = checkpoint_dir / f"global_step_{step:06d}"
        actor_dir = step_dir / "actor"
        actor_dir.mkdir(parents=True, exist_ok=True)
        self.actor_mgr.save_checkpoint(str(actor_dir), step)
        if self.tokenizer is not None:
            self.tokenizer.save_pretrained(actor_dir)
        trainer_state = {
            "step": step,
            "path": str(actor_dir.relative_to(checkpoint_dir)),
            "format": "huggingface",
        }
        (step_dir / "trainer_state.json").write_text(
            json.dumps(trainer_state, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        (checkpoint_dir / "latest.json").write_text(
            json.dumps(trainer_state, ensure_ascii=False),
            encoding="utf-8",
        )
        (checkpoint_dir / "latest_checkpointed_iteration.txt").write_text(str(step), encoding="utf-8")
        print(f"[trainer] checkpoint_saved step={step} dir={actor_dir}", flush=True)

    def load_checkpoint(self) -> int:
        latest = Path(self.experiment_cfg.checkpoint.dir) / "latest.json"
        if not latest.exists():
            return 0
        payload = json.loads(latest.read_text(encoding="utf-8"))
        path = Path(self.experiment_cfg.checkpoint.dir) / payload["path"]
        return self.actor_mgr.load_checkpoint(str(path))

    def sync_rollout_weights(self, step: int) -> None:
        if self.sync_mgr is not None:
            self.sync_mgr.update_weights(step)
            return
        legacy_hook = getattr(self.rollout_mgr, "on_actor_state_changed", None)
        if legacy_hook is not None:
            legacy_hook(step)

    async def sync_rollout_weights_async(self, step: int) -> None:
        if self.sync_mgr is not None:
            await self._await_if_needed(self.sync_mgr.update_weights(step))
            return
        legacy_hook = getattr(self.rollout_mgr, "on_actor_state_changed", None)
        if legacy_hook is not None:
            await self._await_if_needed(legacy_hook(step))

    def _append_sync_timing(self, timing: dict[str, float]) -> None:
        sync_timing = getattr(self.sync_mgr, "last_timing", None)
        if isinstance(sync_timing, dict):
            timing.update({str(key): float(value) for key, value in sync_timing.items()})

    def _setup_fit(self) -> None:
        self._ensure_ready()
        validate_loop_sync_boundary(self.experiment_cfg, self.system_cfg)
        self._set_seed()
        if self.experiment_cfg.checkpoint.resume:
            self.global_step = self.load_checkpoint()
            if self.global_step:
                self.sync_rollout_weights(self.global_step)

    async def _setup_fit_async(self) -> None:
        self._ensure_ready()
        validate_loop_sync_boundary(self.experiment_cfg, self.system_cfg)
        self._set_seed()
        if self.experiment_cfg.checkpoint.resume:
            self.global_step = self.load_checkpoint()
            if self.global_step:
                await self.sync_rollout_weights_async(self.global_step)

    def _finish_step(
        self,
        trained_batch: DataProto,
        update_metrics: dict[str, float],
        step: int,
        max_steps: int,
        timing: dict[str, float] | None = None,
    ) -> None:
        metrics = self._build_step_metrics(trained_batch, update_metrics, step, timing=timing)
        if step % self.experiment_cfg.loop.log_freq == 0:
            self.logger.log_metrics(step, metrics)
        self._maybe_log_trajectories(trained_batch, step, max_steps)
        checkpoint_freq = self.experiment_cfg.checkpoint.save_freq
        if checkpoint_freq > 0 and (step % checkpoint_freq == 0 or step == max_steps):
            self.save_checkpoint(step)

    def _fit_sync(self) -> None:
        data_iter = iter(self.train_dataloader)
        max_steps = self.experiment_cfg.loop.max_steps
        while self.global_step < max_steps:
            step = self.global_step + 1
            step_started = time.perf_counter()
            timing: dict[str, float] = {}
            self.global_step = step
            print(f"[trainer] step={step} phase=prepare", flush=True)
            data_iter, prepared = self._next_prepared_batch(data_iter, rollout_step=step)
            print(f"[trainer] step={step} phase=rollout", flush=True)
            started = time.perf_counter()
            rollout_output = self._run_rollout_phase(prepared, policy_version=step - 1)
            timing["rollout"] = time.perf_counter() - started
            update_batch = self.build_update_batch(prepared, rollout_output)
            print(f"[trainer] step={step} phase=train", flush=True)
            started = time.perf_counter()
            trained_batch, update_metrics = self._run_train_phase(update_batch)
            timing["train"] = time.perf_counter() - started
            print(f"[trainer] step={step} phase=sync", flush=True)
            started = time.perf_counter()
            self.sync_rollout_weights(step)
            timing["sync"] = time.perf_counter() - started
            self._append_sync_timing(timing)
            timing["step"] = time.perf_counter() - step_started
            self._finish_step(trained_batch, update_metrics, step, max_steps, timing=timing)

    async def _fit_one_step_off_async(self) -> None:
        data_iter = iter(self.train_dataloader)
        max_steps = self.experiment_cfg.loop.max_steps
        if self.global_step >= max_steps:
            return
        print(f"[trainer] step={self.global_step} phase=initial_sync", flush=True)
        await self.sync_rollout_weights_async(self.global_step)
        data_iter, current_prepared = self._next_prepared_batch(data_iter, rollout_step=self.global_step)
        print(f"[trainer] step={self.global_step} phase=initial_rollout", flush=True)
        rollout_started = time.perf_counter()
        current_output = await self._run_rollout_phase_async(current_prepared, policy_version=self.global_step)
        current_rollout_time = time.perf_counter() - rollout_started
        current_batch = self.build_update_batch(current_prepared, current_output)
        rollout_step = self.global_step + 1
        while self.global_step < max_steps:
            step = self.global_step + 1
            step_started = time.perf_counter()
            timing: dict[str, float] = {"rollout": current_rollout_time}
            next_prepared: DataProto | None = None
            next_task: asyncio.Task[DataProto] | None = None
            next_rollout_started = 0.0
            if step > 1:
                print(f"[trainer] step={step} phase=sync", flush=True)
                started = time.perf_counter()
                await self.sync_rollout_weights_async(self.global_step)
                timing["sync"] = timing.get("sync", 0.0) + time.perf_counter() - started
                self._append_sync_timing(timing)
            if step < max_steps:
                data_iter, next_prepared = self._next_prepared_batch(data_iter, rollout_step=rollout_step)
                rollout_step += 1
                print(
                    f"[trainer] step={step} phase=prefetch_rollout policy_version={self.global_step}",
                    flush=True,
                )
                next_rollout_started = time.perf_counter()
                next_task = asyncio.create_task(
                    self._run_rollout_phase_background_async(next_prepared, policy_version=self.global_step)
                )
                await asyncio.sleep(0)
            print(f"[trainer] step={step} phase=train", flush=True)
            started = time.perf_counter()
            trained_batch, update_metrics = self._run_train_phase(current_batch)
            timing["train"] = time.perf_counter() - started
            self.global_step = step
            current_rollout_time = 0.0
            if next_task is not None:
                print(f"[trainer] step={step} phase=wait_prefetch", flush=True)
                assert next_prepared is not None
                wait_started = time.perf_counter()
                next_output = await next_task
                timing["rollout_wait"] = time.perf_counter() - wait_started
                current_rollout_time = time.perf_counter() - next_rollout_started
                current_batch = self.build_update_batch(next_prepared, next_output)
            else:
                print(f"[trainer] step={step} phase=sync", flush=True)
                started = time.perf_counter()
                await self.sync_rollout_weights_async(step)
                timing["sync"] = timing.get("sync", 0.0) + time.perf_counter() - started
                self._append_sync_timing(timing)
            timing["wall_step"] = time.perf_counter() - step_started
            timing["step"] = timing.get("rollout", 0.0) + timing.get("train", 0.0) + timing.get("sync", 0.0)
            self._finish_step(trained_batch, update_metrics, step, max_steps, timing=timing)

    async def _fit_one_step_off_entry_async(self) -> None:
        await self._setup_fit_async()
        await self._fit_one_step_off_async()

    def fit(self) -> None:
        try:
            if self.experiment_cfg.loop.mode == "one_step_off":
                asyncio.run(self._fit_one_step_off_entry_async())
            else:
                self._setup_fit()
                self._fit_sync()
        finally:
            if self.logger is not None:
                self.logger.close()
