from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from typing import Any, Callable

import torch

from nanoverl.data import DataProto



def _compute_reward_task(task: tuple[Callable[..., float], Any, str, Any, dict[str, Any]]) -> float:
    reward_fn, data_source, solution_str, ground_truth, extra_info = task
    return float(
        reward_fn(
            data_source=str(data_source),
            solution_str=solution_str,
            ground_truth=ground_truth,
            extra_info=extra_info,
        )
    )


class RewardManager:
    def __init__(
        self,
        tokenizer: Any,
        reward_fn: Callable[..., float],
        num_examine: int,
        num_workers: int = 1,
        parallel_backend: str = "thread",
        logger: Any | None = None,
    ) -> None:
        self.tokenizer = tokenizer
        self.reward_fn = reward_fn
        self.num_examine = max(int(num_examine), 0)
        self.num_workers = max(int(num_workers), 1)
        self.parallel_backend = str(parallel_backend).lower().strip()
        if self.parallel_backend not in {"thread", "process"}:
            raise ValueError(f"Unsupported reward parallel backend: {parallel_backend}")
        self.logger = logger

    def _decode_solution_strs(self, batch: DataProto) -> list[str]:
        response_texts = batch.non_tensor_batch.get("response_text")
        if response_texts is not None:
            return [str(text) for text in response_texts]
        if "responses" not in batch.batch:
            raise KeyError("RewardManager requires response_text or responses in the batch")
        responses = batch.batch["responses"].detach().cpu()
        return [str(text) for text in self.tokenizer.batch_decode(responses, skip_special_tokens=True)]

    def _sample_value(
        self,
        batch: DataProto,
        key: str,
        idx: int,
        default: Any = None,
    ) -> Any:
        values = batch.non_tensor_batch.get(key)
        if values is None:
            return default
        return values[idx]

    def _build_extra_info(self, batch: DataProto, idx: int) -> dict[str, Any]:
        excluded = {"data_source", "ground_truth", "response_text"}
        extra_info = {
            key: values[idx]
            for key, values in batch.non_tensor_batch.items()
            if key not in excluded
        }
        if batch.meta_info:
            extra_info["meta_info"] = dict(batch.meta_info)
        return extra_info

    def _compute_one_reward(
        self,
        data_source: Any,
        solution_str: str,
        ground_truth: Any,
        extra_info: dict[str, Any],
    ) -> float:
        return _compute_reward_task((self.reward_fn, data_source, solution_str, ground_truth, extra_info))

    def _compute_base_rewards(self, reward_inputs: list[tuple[Any, str, Any, dict[str, Any]]]) -> list[float]:
        if self.num_workers <= 1 or len(reward_inputs) <= 1:
            return [self._compute_one_reward(*item) for item in reward_inputs]

        worker_count = min(self.num_workers, len(reward_inputs))
        if self.parallel_backend == "process":
            tasks = [(self.reward_fn, *item) for item in reward_inputs]
            with ProcessPoolExecutor(max_workers=worker_count) as executor:
                return list(executor.map(_compute_reward_task, tasks))

        with ThreadPoolExecutor(max_workers=worker_count) as executor:
            return list(executor.map(lambda item: self._compute_one_reward(*item), reward_inputs))

    def _log_samples(
        self,
        batch: DataProto,
        solution_strs: list[str],
        rewards: list[float],
    ) -> None:
        if self.num_examine <= 0 or self.logger is None:
            return
        samples: list[list[Any]] = []
        for idx in range(min(self.num_examine, len(solution_strs))):
            prompt = str(self._sample_value(batch, "prompt_text", idx, self._sample_value(batch, "question", idx, "")))
            solution = solution_strs[idx]
            samples.append([prompt, solution, float(rewards[idx])])
        self.logger.log_samples(int(batch.meta_info.get("step", 0)), samples)

    def __call__(self, batch: DataProto) -> DataProto:
        solution_strs = self._decode_solution_strs(batch)
        default_sources = ["unknown"] * len(solution_strs)
        data_sources = batch.non_tensor_batch.get("data_source", default_sources)
        ground_truths = batch.non_tensor_batch.get("ground_truth", [None] * len(solution_strs))
        reward_inputs = [
            (
                data_sources[idx],
                solution_str,
                ground_truths[idx],
                self._build_extra_info(batch, idx),
            )
            for idx, solution_str in enumerate(solution_strs)
        ]
        base_rewards = self._compute_base_rewards(reward_inputs)
        rewards: list[float] = []
        details: list[dict[str, Any]] = []
        for base_reward in base_rewards:
            reward = float(base_reward)
            rewards.append(reward)
            details.append({"reward": reward})
        self._log_samples(batch, solution_strs, rewards)
        reward_proto = DataProto.from_dict(
            {"rewards": torch.tensor(rewards, dtype=torch.float32)},
            {"reward_details": details, "response_text": solution_strs},
        )
        return batch.union(reward_proto)
