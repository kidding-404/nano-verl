from __future__ import annotations

import torch

from nanoverl.config import ExperimentConfig
from nanoverl.data import DataProto

SAFETY_BOUND = 20.0


def prepare_gen_batch(batch: DataProto, experiment_cfg: ExperimentConfig) -> DataProto:
    return batch.select(
        batch_keys=["prompt_ids", "prompt_attention_mask", "group_ids"],
        non_tensor_keys=["question", "ground_truth", "data_source", "prompt_text", "request_id"],
    )


def build_update_batch(batch: DataProto, rollout_output: DataProto, experiment_cfg: ExperimentConfig) -> DataProto:
    _ = experiment_cfg
    return batch.union(rollout_output)


def compute_grpo_outcome_advantage(
    token_level_rewards: torch.Tensor,
    response_mask: torch.Tensor,
    index: torch.Tensor,
    epsilon: float = 1e-6,
) -> tuple[torch.Tensor, torch.Tensor]:
    scores = token_level_rewards.sum(dim=-1) if token_level_rewards.dim() > 1 else token_level_rewards
    advantages = torch.zeros_like(scores)
    returns = scores.clone()
    unique_index = torch.unique(index)
    for group_id in unique_index.tolist():
        group_mask = index == group_id
        group_rewards = scores[group_mask]
        centered = group_rewards - group_rewards.mean()
        if group_rewards.numel() > 1:
            centered = centered / (group_rewards.std(unbiased=True) + epsilon)
        advantages[group_mask] = centered
    advantages = advantages.unsqueeze(-1) * response_mask.float()
    returns = returns.unsqueeze(-1) * response_mask.float()
    return advantages, returns


def compute_advantage(batch: DataProto, experiment_cfg: ExperimentConfig) -> DataProto:
    group_ids = batch.batch["group_ids"]
    rewards = batch.batch.get("token_level_rewards", batch.batch["rewards"])
    response_mask = batch.batch["response_mask"]
    advantages, returns = compute_grpo_outcome_advantage(
        token_level_rewards=rewards,
        response_mask=response_mask,
        index=group_ids,
        epsilon=experiment_cfg.algorithm.epsilon,
    )
    return batch.union(DataProto.from_dict({"advantages": advantages, "returns": returns}))


def need_ref_log_prob(experiment_cfg: ExperimentConfig) -> bool:
    return experiment_cfg.algorithm.beta > 0


def kl_penalty(log_prob: torch.Tensor, ref_log_prob: torch.Tensor, kl_penalty_type: str = "kl") -> torch.Tensor:
    if kl_penalty_type in {"kl", "k1"}:
        return log_prob - ref_log_prob
    if kl_penalty_type == "abs":
        return (log_prob - ref_log_prob).abs()
    if kl_penalty_type in {"mse", "k2"}:
        return 0.5 * (log_prob - ref_log_prob).square()
    if kl_penalty_type in {"low_var_kl", "k3"}:
        kl = torch.clamp(ref_log_prob - log_prob, min=-20.0, max=20.0)
        return torch.clamp(torch.exp(kl) - kl - 1.0, min=-10.0, max=10.0)
    raise NotImplementedError(f"Unsupported KL penalty type: {kl_penalty_type}")


def _token_level_scores(rewards: torch.Tensor, response_mask: torch.Tensor) -> torch.Tensor:
    scores = torch.zeros_like(response_mask, dtype=torch.float32)
    lengths = response_mask.long().sum(dim=1)
    valid = lengths > 0
    if valid.any():
        rows = torch.arange(response_mask.shape[0], device=response_mask.device)[valid]
        scores[rows, lengths[valid] - 1] = rewards.float().to(response_mask.device)[valid]
    return scores


def apply_kl_penalty(batch: DataProto, beta: float, kl_penalty_type: str = "kl") -> tuple[DataProto, dict[str, float]]:
    response_mask = batch.batch["response_mask"]
    mask = response_mask.float()
    kld = kl_penalty(batch.batch["old_log_probs"], batch.batch["ref_log_probs"], kl_penalty_type) * mask
    token_level_rewards = _token_level_scores(batch.batch["rewards"], response_mask) - float(beta) * kld
    current_kl = _masked_mean(kld, response_mask, axis=-1).mean()
    return batch.union(DataProto.from_dict({"token_level_rewards": token_level_rewards})), {
        "reward_kl_penalty": float(current_kl.item()),
        "reward_kl_penalty_coeff": float(beta),
    }


def _masked_sum(values: torch.Tensor, mask: torch.Tensor, axis: int | None = None) -> torch.Tensor:
    masked = values.float() * mask.float()
    return masked.sum() if axis is None else masked.sum(dim=axis)


def _masked_mean(values: torch.Tensor, mask: torch.Tensor, axis: int | None = None) -> torch.Tensor:
    mask_f = mask.float()
    numerator = values.float() * mask_f
    if axis is None:
        return numerator.sum() / mask_f.sum().clamp_min(1.0)
    return numerator.sum(dim=axis) / mask_f.sum(dim=axis).clamp_min(1.0)


def compute_offpolicy_metrics(
    old_log_prob: torch.Tensor,
    rollout_log_prob: torch.Tensor,
    response_mask: torch.Tensor,
) -> dict[str, float]:
    """Compute lightweight rollout-vs-training policy diagnostics."""
    if not response_mask.bool().any():
        return {}

    mean_log_prob_training = _masked_mean(old_log_prob, response_mask, axis=-1)
    mean_log_prob_rollout = _masked_mean(rollout_log_prob, response_mask, axis=-1)
    log_ratio = old_log_prob - rollout_log_prob
    log_ppl_diff = mean_log_prob_rollout - mean_log_prob_training

    log_ratio_safe = torch.clamp(log_ratio, min=-SAFETY_BOUND, max=SAFETY_BOUND)
    log_ratio_sum = _masked_sum(log_ratio, response_mask, axis=-1)
    log_ratio_sum_safe = torch.clamp(log_ratio_sum, min=-SAFETY_BOUND, max=SAFETY_BOUND)

    return {
        "training_ppl": float(torch.exp(-mean_log_prob_training).mean().item()),
        "training_log_ppl": float((-mean_log_prob_training).mean().item()),
        "kl": float(_masked_mean(rollout_log_prob - old_log_prob, response_mask).item()),
        "k3_kl": float((_masked_mean(torch.exp(log_ratio) - log_ratio - 1, response_mask)).item()),
        "rollout_ppl": float(torch.exp(-mean_log_prob_rollout).mean().item()),
        "rollout_log_ppl": float((-mean_log_prob_rollout).mean().item()),
        "log_ppl_diff": float(log_ppl_diff.mean().item()),
        "log_ppl_abs_diff": float(log_ppl_diff.abs().mean().item()),
        "log_ppl_diff_max": float(log_ppl_diff.max().item()),
        "log_ppl_diff_min": float(log_ppl_diff.min().item()),
        "ppl_ratio": float(torch.exp(log_ppl_diff).mean().item()),
        "chi2_token": float((_masked_mean(torch.exp(log_ratio_safe).square(), response_mask) - 1.0).item()),
        "chi2_seq": float((torch.exp(2.0 * log_ratio_sum_safe).mean() - 1.0).item()),
    }


def compute_policy_loss(
    old_log_prob: torch.Tensor,
    log_prob: torch.Tensor,
    advantages: torch.Tensor,
    response_mask: torch.Tensor,
    clip_low: float = 0.2,
    clip_high: float = 0.2,
    clip_ratio_c: float = 3.0,
) -> tuple[torch.Tensor, dict[str, float]]:
    mask = response_mask.float()
    ratio = torch.exp((log_prob - old_log_prob).clamp(min=-20.0, max=20.0))
    clipped_ratio = torch.clamp(ratio, 1.0 - clip_low, 1.0 + clip_high)
    pg_losses1 = -advantages * ratio
    pg_losses2 = -advantages * clipped_ratio
    clipped_pg_losses = torch.maximum(pg_losses1, pg_losses2)
    lower_bound_losses = -advantages * clip_ratio_c
    dual_clipped_losses = torch.min(lower_bound_losses, clipped_pg_losses)
    pg_losses = torch.where(advantages < 0, dual_clipped_losses, clipped_pg_losses)
    masked_loss = pg_losses * mask
    denom = mask.sum().clamp_min(1.0)
    loss = masked_loss.sum() / denom
    clipfrac = torch.gt(pg_losses2, pg_losses1).float().mul(mask).sum() / denom
    clipfrac_lower = torch.gt(clipped_pg_losses, lower_bound_losses).float().mul(advantages < 0).mul(mask).sum() / denom
    approx_kl = ((old_log_prob - log_prob) * mask).sum() / denom
    return loss, {
        "clipfrac": float(clipfrac.item()),
        "clipfrac_lower": float(clipfrac_lower.item()),
        "approx_kl": float(approx_kl.item()),
    }
