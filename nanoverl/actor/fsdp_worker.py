from __future__ import annotations

import ctypes
import gc
import importlib.util
import os
import socket
from contextlib import nullcontext
from datetime import timedelta
from pathlib import Path
from typing import Any

import ray
import torch
import torch.distributed as dist
from torch.distributed.checkpoint.state_dict import StateDictOptions, get_model_state_dict, set_model_state_dict
from torch.distributed.device_mesh import DeviceMesh, init_device_mesh
from torch.distributed.fsdp import MixedPrecisionPolicy, fully_shard
from torch.distributed.tensor import DTensor
from torch.nn.utils import clip_grad_norm_
from torch.nn.utils.rnn import pad_sequence
from torch.optim import AdamW
from transformers import AutoConfig, AutoModelForCausalLM

from nanoverl.algorithms import compute_policy_loss
from nanoverl.config import SystemConfig
from nanoverl.data import DataProto
from nanoverl.sync.naive_sync_engine import NaiveSyncEngine
from nanoverl.sync.nccl_sync_engine import NCCLSyncEngine

_DEFAULT_DYNAMIC_MICRO_BATCH_MAX_TOKENS = 4096
_MICRO_BATCH_INDICES_KEY = "_nanoverl_micro_batch_indices"

try:
    from flash_attn.ops.triton.cross_entropy import cross_entropy_loss
except ImportError:  # pragma: no cover
    cross_entropy_loss = None


def resolve_dtype(dtype_name: str) -> torch.dtype:
    mapping = {
        "float32": torch.float32,
        "fp32": torch.float32,
        "float16": torch.float16,
        "fp16": torch.float16,
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
    }
    if dtype_name not in mapping:
        raise ValueError(f"Unsupported dtype: {dtype_name}")
    return mapping[dtype_name]


def is_flash_attn_available() -> bool:
    return importlib.util.find_spec("flash_attn") is not None


def trim_cpu_memory() -> None:
    gc.collect()
    if os.name != "posix":
        return
    try:
        malloc_trim = ctypes.CDLL("libc.so.6").malloc_trim
    except Exception:
        return
    malloc_trim(0)


def _prepare_prompt_tensors(
    prompt_ids: torch.Tensor,
    prompt_attention_mask: torch.Tensor,
    pad_token_id: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    prompt_ids = prompt_ids.to(device)
    prompt_attention_mask = prompt_attention_mask.to(device)
    prompt_ids = prompt_ids.masked_fill(prompt_attention_mask == 0, pad_token_id)
    return prompt_ids, prompt_attention_mask


def _target_log_probs_from_logits(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    if logits.numel() == 0:
        return torch.empty_like(labels, dtype=torch.float32)

    if cross_entropy_loss is not None and logits.is_cuda:
        flat_logits = logits.reshape(-1, logits.shape[-1])
        flat_labels = labels.reshape(-1)
        output = cross_entropy_loss(flat_logits, flat_labels, inplace_backward=True)
        losses = output[0] if isinstance(output, tuple) else output
        return (-losses).view_as(labels).float()

    logits_f = logits.float()
    label_logits = torch.gather(logits_f, dim=-1, index=labels.unsqueeze(-1)).squeeze(-1)
    logsumexp = torch.stack([torch.logsumexp(row, dim=-1) for row in logits_f])
    return label_logits - logsumexp


def _entropy_from_logits(logits: torch.Tensor) -> torch.Tensor:
    probs = torch.nn.functional.softmax(logits, dim=-1)
    return torch.logsumexp(logits, dim=-1) - torch.sum(probs * logits, dim=-1)


def _entropy_from_logits_with_chunking(logits: torch.Tensor, chunk_size: int = 2048) -> torch.Tensor:
    entropy = torch.zeros(logits.shape[0], device=logits.device)
    for start in range(0, logits.shape[0], chunk_size):
        logits_chunk = logits[start : start + chunk_size].float()
        probs_chunk = torch.nn.functional.softmax(logits_chunk, dim=-1)
        entropy_chunk = torch.logsumexp(logits_chunk, dim=-1) - torch.sum(probs_chunk * logits_chunk, dim=-1)
        entropy[start : start + chunk_size] = entropy_chunk
    return entropy


def _transformer_layer_class_names(model: torch.nn.Module) -> set[str]:
    layer_names = getattr(model, "_no_split_modules", None) or []
    if isinstance(layer_names, str):
        layer_names = [layer_names]
    return {str(name) for name in layer_names}


def _iter_transformer_layer_modules(model: torch.nn.Module) -> list[torch.nn.Module]:
    layer_name_set = _transformer_layer_class_names(model)
    if not layer_name_set:
        return []
    return [
        module
        for module in model.modules()
        if module is not model and module.__class__.__name__ in layer_name_set
    ]


def _detach_state_tensor(tensor: torch.Tensor, *, device: torch.device, offload_to_cpu: bool) -> torch.Tensor:
    if isinstance(tensor, DTensor):
        tensor = tensor.full_tensor()
    tensor = tensor.detach()
    if offload_to_cpu:
        return tensor.cpu().contiguous()
    return tensor.to(device=device).contiguous()


def _sync_weight_tensor(tensor: torch.Tensor, *, device: torch.device, offload_to_cpu: bool) -> torch.Tensor:
    tensor = _detach_state_tensor(tensor, device=device, offload_to_cpu=offload_to_cpu)
    if not offload_to_cpu and tensor.is_floating_point():
        tensor = tensor.to(dtype=torch.bfloat16)
    return tensor.contiguous()


def _scatter_response_stats(
    shift_logits: torch.Tensor,
    shift_labels: torch.Tensor,
    sample_indices: list[int],
    prompt_lengths: list[int],
    response_lengths: list[int],
    resp_log_probs: torch.Tensor,
    resp_entropy: torch.Tensor,
    *,
    calculate_entropy: bool,
) -> torch.Tensor:
    response_logit_slices: list[torch.Tensor] = []
    response_label_slices: list[torch.Tensor] = []
    response_sample_indices: list[int] = []
    for row_idx, sample_idx in enumerate(sample_indices):
        prompt_len = prompt_lengths[sample_idx]
        response_len = response_lengths[sample_idx]
        if response_len == 0:
            continue
        start = prompt_len - 1
        end = start + response_len
        response_logit_slices.append(shift_logits[row_idx, start:end])
        response_label_slices.append(shift_labels[row_idx, start:end])
        response_sample_indices.append(sample_idx)

    if not response_logit_slices:
        return shift_logits.sum() * 0.0

    flat_logits = torch.cat(response_logit_slices, dim=0)
    flat_labels = torch.cat(response_label_slices, dim=0)
    flat_log_probs = _target_log_probs_from_logits(flat_logits, flat_labels)
    flat_entropy = _entropy_from_logits_with_chunking(flat_logits) if calculate_entropy else None
    offset = 0
    for sample_idx in response_sample_indices:
        response_len = response_lengths[sample_idx]
        next_offset = offset + response_len
        resp_log_probs[sample_idx, :response_len] = flat_log_probs[offset:next_offset]
        if flat_entropy is not None:
            resp_entropy[sample_idx, :response_len] = flat_entropy[offset:next_offset]
        offset = next_offset
    return flat_log_probs.sum() * 0.0


def _scatter_packed_response_stats(
    flat_logits: torch.Tensor,
    flat_ids: torch.Tensor,
    sequence_offsets: list[int],
    prompt_lengths: list[int],
    response_lengths: list[int],
    resp_log_probs: torch.Tensor,
    resp_entropy: torch.Tensor,
    *,
    calculate_entropy: bool,
) -> torch.Tensor:
    response_logit_slices: list[torch.Tensor] = []
    response_label_slices: list[torch.Tensor] = []
    response_sample_indices: list[int] = []
    for sample_idx, sequence_offset in enumerate(sequence_offsets):
        prompt_len = prompt_lengths[sample_idx]
        response_len = response_lengths[sample_idx]
        if response_len == 0:
            continue
        start = sequence_offset + prompt_len - 1
        end = start + response_len
        response_logit_slices.append(flat_logits[start:end])
        response_label_slices.append(flat_ids[start + 1 : end + 1])
        response_sample_indices.append(sample_idx)

    if not response_logit_slices:
        return flat_logits.sum() * 0.0

    response_logits = torch.cat(response_logit_slices, dim=0)
    response_labels = torch.cat(response_label_slices, dim=0)
    flat_log_probs = _target_log_probs_from_logits(response_logits, response_labels)
    flat_entropy = _entropy_from_logits_with_chunking(response_logits) if calculate_entropy else None
    offset = 0
    for sample_idx in response_sample_indices:
        response_len = response_lengths[sample_idx]
        next_offset = offset + response_len
        resp_log_probs[sample_idx, :response_len] = flat_log_probs[offset:next_offset]
        if flat_entropy is not None:
            resp_entropy[sample_idx, :response_len] = flat_entropy[offset:next_offset]
        offset = next_offset
    return flat_log_probs.sum() * 0.0


def compute_response_stats(
    model: Any,
    prompt_ids: torch.Tensor,
    prompt_attention_mask: torch.Tensor,
    responses: torch.Tensor,
    response_mask: torch.Tensor,
    pad_token_id: int,
    device: torch.device,
    calculate_entropy: bool = False,
    use_remove_padding: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    prompt_ids, prompt_attention_mask = _prepare_prompt_tensors(
        prompt_ids, prompt_attention_mask, pad_token_id, device
    )
    responses = responses.to(device)
    response_mask = response_mask.to(device)
    merged_ids: list[torch.Tensor] = []
    merged_mask: list[torch.Tensor] = []
    prompt_lengths: list[int] = []
    response_lengths: list[int] = []
    for idx in range(prompt_ids.shape[0]):
        prompt_len = int(prompt_attention_mask[idx].sum().item())
        response_len = int(response_mask[idx].sum().item())
        prompt_lengths.append(prompt_len)
        response_lengths.append(response_len)
        merged = torch.cat([prompt_ids[idx, :prompt_len], responses[idx, :response_len]], dim=0)
        merged_ids.append(merged)
        merged_mask.append(torch.ones_like(merged))
    max_response = responses.shape[1]
    resp_log_probs = torch.zeros((responses.shape[0], max_response), dtype=torch.float32, device=device)
    resp_entropy = torch.zeros_like(resp_log_probs)

    if use_remove_padding and len(merged_ids) == 1:
        sequence_offsets: list[int] = []
        offset = 0
        for merged in merged_ids:
            sequence_offsets.append(offset)
            offset += int(merged.numel())
        flat_ids_1d = torch.cat(merged_ids, dim=0)
        position_ids = torch.cat(
            [
                torch.arange(int(merged.numel()), dtype=torch.long, device=device)
                for merged in merged_ids
            ],
            dim=0,
        ).unsqueeze(0)
        outputs = model(
            input_ids=flat_ids_1d.unsqueeze(0),
            attention_mask=None,
            position_ids=position_ids,
        )
        flat_logits = outputs.logits.squeeze(0)
        zero_anchor = _scatter_packed_response_stats(
            flat_logits,
            flat_ids_1d,
            sequence_offsets,
            prompt_lengths,
            response_lengths,
            resp_log_probs,
            resp_entropy,
            calculate_entropy=calculate_entropy,
        )
    else:
        padded_ids = pad_sequence(merged_ids, batch_first=True, padding_value=pad_token_id)
        padded_mask = pad_sequence(merged_mask, batch_first=True, padding_value=0)
        outputs = model(input_ids=padded_ids, attention_mask=padded_mask)
        shift_logits = outputs.logits[:, :-1]
        shift_labels = padded_ids[:, 1:]
        zero_anchor = _scatter_response_stats(
            shift_logits,
            shift_labels,
            list(range(len(merged_ids))),
            prompt_lengths,
            response_lengths,
            resp_log_probs,
            resp_entropy,
            calculate_entropy=calculate_entropy,
        )
    return resp_log_probs + zero_anchor, resp_entropy + zero_anchor


class FSDPActorWorker:
    def __init__(self, tokenizer: Any, backend_cfg: dict | None = None) -> None:
        self.tokenizer = tokenizer
        self.backend_cfg = backend_cfg or {}
        self.system_cfg: SystemConfig | None = None
        self.model: Any = None
        self.reference_model: Any = None
        self.optimizer: AdamW | None = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.rank = 0
        self.world_size = 1
        self.local_rank = 0
        self.use_fsdp = False
        self.pad_token_id = tokenizer.pad_token_id or tokenizer.eos_token_id or 0
        self.flash_attn_active = False
        self.sync_engine: NCCLSyncEngine | None = None
        self.sync_rank: int | None = None

    def _build_model_load_kwargs(self, dtype: torch.dtype) -> dict[str, Any]:
        assert self.system_cfg is not None
        if not torch.cuda.is_available():
            raise RuntimeError("Training requires CUDA for actor models")
        attn_implementation = "flash_attention_2"
        if not is_flash_attn_available():
            raise RuntimeError("flash_attn is required because nanoverl now uses flash_attention_2 for actor models")
        return {
            "dtype": dtype,
            "trust_remote_code": self.system_cfg.model.trust_remote_code,
            "attn_implementation": attn_implementation,
        }

    def _load_hf_model(self, dtype: torch.dtype) -> Any:
        assert self.system_cfg is not None
        load_kwargs = self._build_model_load_kwargs(dtype)
        with self._model_init_context():
            model = AutoModelForCausalLM.from_pretrained(
                self.system_cfg.model.path,
                **load_kwargs,
            )
        self.flash_attn_active = load_kwargs["attn_implementation"] == "flash_attention_2"
        if self.rank == 0:
            print(f"[actor] {load_kwargs['attn_implementation']} enabled for training", flush=True)
        return model

    def _model_init_context(self) -> Any:
        assert self.system_cfg is not None
        if not self.use_fsdp or self.rank == 0:
            return nullcontext()
        model_config = AutoConfig.from_pretrained(
            self.system_cfg.model.path,
            trust_remote_code=self.system_cfg.model.trust_remote_code,
        )
        if bool(getattr(model_config, "tie_word_embeddings", False)):
            # Match verl: tied word embeddings are not safe to construct on meta tensors.
            return nullcontext()
        from accelerate import init_empty_weights

        return init_empty_weights()

    def _actor_model_load_dtype(self) -> torch.dtype:
        assert self.system_cfg is not None
        fsdp_config = self.system_cfg.actor.fsdp_config
        return resolve_dtype(fsdp_config.model_dtype or fsdp_config.mix_precision)

    def _actor_mixed_precision_dtype(self) -> torch.dtype:
        assert self.system_cfg is not None
        return resolve_dtype(self.system_cfg.actor.fsdp_config.mix_precision)

    def _fsdp2_mixed_precision(self, model: torch.nn.Module) -> Any:
        assert self.system_cfg is not None
        param_dtype = self._actor_mixed_precision_dtype()
        return MixedPrecisionPolicy(
            param_dtype=param_dtype,
            reduce_dtype=torch.float32,
            cast_forward_inputs=True,
        )

    def _fsdp2_device_mesh(self) -> DeviceMesh | None:
        assert self.system_cfg is not None
        fsdp_size = int(self.system_cfg.actor.fsdp_config.fsdp_size)
        if fsdp_size <= 0 or fsdp_size >= self.world_size:
            return None
        if self.world_size % fsdp_size != 0:
            raise ValueError(
                "actor.fsdp.fsdp_size must divide WORLD_SIZE when set, "
                f"got fsdp_size={fsdp_size}, world_size={self.world_size}"
            )
        return init_device_mesh(
            self.device.type,
            (self.world_size // fsdp_size, fsdp_size),
            mesh_dim_names=("replicate", "shard"),
        )

    def _apply_fsdp2(self, model: torch.nn.Module) -> torch.nn.Module:
        assert self.system_cfg is not None
        shard_kwargs: dict[str, Any] = {
            "mesh": self._fsdp2_device_mesh(),
            "reshard_after_forward": bool(self.system_cfg.actor.fsdp_config.reshard_after_forward),
            "mp_policy": self._fsdp2_mixed_precision(model),
        }

        for layer in _iter_transformer_layer_modules(model):
            fully_shard(layer, **shard_kwargs)
        return fully_shard(model, **shard_kwargs)

    def _load_fsdp2_initial_state(self, full_state: dict[str, torch.Tensor]) -> None:
        state_to_load = full_state if self.rank == 0 else {}
        set_model_state_dict(
            self.model,
            state_to_load,
            options=StateDictOptions(
                full_state_dict=True,
                strict=True,
                broadcast_from_rank0=True,
            ),
        )

    def _forward_autocast_dtype(self) -> torch.dtype | None:
        if self.system_cfg is None or self.device.type != "cuda":
            return None
        dtype = self._actor_mixed_precision_dtype()
        if dtype in {torch.float16, torch.bfloat16}:
            return dtype
        return None

    def _init_distributed(self) -> None:
        self.rank = int(os.environ.get("RANK", "0"))
        self.world_size = int(os.environ.get("WORLD_SIZE", "1"))
        self.local_rank = int(os.environ.get("LOCAL_RANK", "0"))
        if self.system_cfg is not None and self.system_cfg.actor.use_fsdp and self.world_size == 1:
            os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
            os.environ.setdefault("MASTER_PORT", "29500")
        if self.system_cfg is not None and self.system_cfg.actor.use_fsdp and not dist.is_initialized():
            backend = "nccl" if torch.cuda.is_available() else "gloo"
            timeout = timedelta(seconds=int(self.system_cfg.actor.distributed_init_timeout_seconds))
            dist.init_process_group(backend=backend, rank=self.rank, world_size=self.world_size, timeout=timeout)
        if torch.cuda.is_available():
            visible_devices = [item for item in os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",") if item]
            device_index = 0 if len(visible_devices) == 1 else self.local_rank
            self.device = torch.device(f"cuda:{device_index}")
            torch.cuda.set_device(self.device)

    def _load_model(self) -> None:
        assert self.system_cfg is not None
        inference_dtype = self._actor_mixed_precision_dtype()
        train_dtype = self._actor_model_load_dtype()
        # 单卡时 FSDP 会退化成 NO_SHARD，这里直接跳过 wrapper，保留单卡 NCCL 进程组即可。
        self.use_fsdp = bool(self.system_cfg.actor.use_fsdp and self.world_size > 1)
        if self.rank == 0 and self.system_cfg.actor.use_fsdp and self.world_size == 1:
            print("[actor] world_size=1, skip FSDP wrapping and keep NCCL process group", flush=True)

        model = self._load_hf_model(train_dtype)
        if self.system_cfg.model.enable_gradient_checkpointing:
            try:
                model.gradient_checkpointing_enable(
                    gradient_checkpointing_kwargs={"use_reentrant": False}
                )
            except TypeError:
                model.gradient_checkpointing_enable()
            if hasattr(model, "config"):
                model.config.use_cache = False
        if hasattr(model, "tie_weights"):
            model.tie_weights()
        fsdp_initial_state = model.state_dict() if self.use_fsdp and self.rank == 0 else {}
        if self.use_fsdp:
            self.model = self._apply_fsdp2(model)
            self._load_fsdp2_initial_state(fsdp_initial_state)
            if hasattr(self.model, "tie_weights"):
                # FSDP2 state loading can recreate untied output embeddings on nonzero ranks.
                self.model.tie_weights()
        else:
            self.model = model.to(self.device)
        self.model.train()
        if self.backend_cfg.get("beta", 0.0) > 0:
            self.reference_model = self._load_hf_model(inference_dtype).to(self.device)
            self.reference_model.eval()
            for param in self.reference_model.parameters():
                param.requires_grad_(False)
        self.optimizer = AdamW(
            self.model.parameters(),
            lr=self.system_cfg.actor.lr,
            weight_decay=self.system_cfg.actor.weight_decay,
        )
        if self.system_cfg.actor.fsdp_config.optimizer_offload:
            self.release_optimizer_state_for_rollout()

    def get_master_addr_port(self) -> tuple[str, int]:
        with socket.socket() as sock:
            sock.bind(("", 0))
            return ray.util.get_node_ip_address(), int(sock.getsockname()[1])

    def get_ray_runtime_info(self) -> dict[str, Any]:
        visible_devices = os.environ.get("CUDA_VISIBLE_DEVICES", "")
        if not visible_devices:
            visible_devices = ",".join(str(gpu_id) for gpu_id in ray.get_gpu_ids())
        return {
            "node_id": ray.get_runtime_context().get_node_id(),
            "visible_devices": visible_devices,
        }

    def init_model(
        self,
        config: SystemConfig | dict,
        rank: int | None = None,
        world_size: int | None = None,
        master_addr: str | None = None,
        master_port: int | None = None,
        local_rank: int | None = None,
        local_world_size: int | None = None,
    ) -> None:
        env_updates = {
            "RANK": rank,
            "WORLD_SIZE": world_size,
            "MASTER_ADDR": master_addr,
            "MASTER_PORT": master_port,
            "LOCAL_RANK": local_rank,
            "LOCAL_WORLD_SIZE": local_world_size,
        }
        for name, value in env_updates.items():
            if value is not None:
                os.environ[name] = str(value)
        self.system_cfg = config if isinstance(config, SystemConfig) else SystemConfig(**config)
        self._init_distributed()
        self._load_model()

    def _broadcast_state(self, state: dict | None) -> dict | None:
        if self.world_size <= 1:
            return state
        obj = [state]
        dist.broadcast_object_list(obj, src=0)
        return obj[0]

    def _move_optimizer_state(self, device: torch.device) -> None:
        if self.optimizer is None:
            return
        for state in self.optimizer.state.values():
            for key, value in state.items():
                if isinstance(value, torch.Tensor):
                    state[key] = value.to(device)

    def _optimizer_checkpoint_device(self) -> torch.device:
        assert self.system_cfg is not None
        if self.system_cfg.actor.fsdp_config.optimizer_offload:
            return torch.device("cpu")
        return self.device

    def release_optimizer_state_for_rollout(self) -> None:
        if self.optimizer is None:
            return
        self._move_optimizer_state(torch.device("cpu"))
        self._clear_cuda_cache()

    def _ensure_optimizer_state_for_training(self) -> None:
        assert self.system_cfg is not None
        if self.optimizer is None:
            return
        if self.system_cfg.actor.fsdp_config.optimizer_offload:
            self._move_optimizer_state(self.device)

    def _clear_cuda_cache(self) -> None:
        trim_cpu_memory()
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            torch.cuda.ipc_collect()
            torch.cuda.empty_cache()

    def _should_clear_cache_after_update(self) -> bool:
        env_value = os.environ.get("NANOVERL_CLEAR_CACHE_AFTER_UPDATE")
        if env_value is not None:
            return env_value == "1"
        if self.system_cfg is None:
            return False
        return bool(getattr(self.system_cfg.actor, "clear_cache_after_update", False))

    def _restore_optimizer_state(self, checkpoint: dict[str, Any]) -> None:
        if self.optimizer is None or not checkpoint.get("optimizer"):
            return
        assert self.system_cfg is not None
        self.optimizer.load_state_dict(checkpoint["optimizer"])
        self._move_optimizer_state(self._optimizer_checkpoint_device())

    def _data_on_device(self, data: DataProto) -> DataProto:
        if all(tensor.device == self.device for tensor in data.batch.values()):
            return data
        return data.to(self.device)

    def _load_hf_checkpoint_dir(
        self,
        checkpoint_path: Path,
    ) -> tuple[dict[str, torch.Tensor] | None, dict[str, Any] | None]:
        assert self.system_cfg is not None
        if self.rank != 0:
            return None, None

        restore_model = AutoModelForCausalLM.from_pretrained(
            str(checkpoint_path),
            dtype=self._actor_model_load_dtype(),
            trust_remote_code=self.system_cfg.model.trust_remote_code,
        )
        state = {name: tensor.cpu() for name, tensor in restore_model.state_dict().items()}
        del restore_model

        trainer_state_path = checkpoint_path / "trainer_state.pt"
        checkpoint = (
            torch.load(trainer_state_path, map_location="cpu")
            if trainer_state_path.exists()
            else {"step": 0, "optimizer": {}}
        )
        return state, checkpoint

    def _apply_checkpoint(self, state: dict | None, checkpoint: dict[str, Any] | None) -> int:
        assert state is not None
        assert checkpoint is not None
        if self.use_fsdp:
            set_model_state_dict(
                self.model,
                state,
                options=StateDictOptions(full_state_dict=True, strict=True),
            )
        else:
            self.model.load_state_dict(state, strict=True)
        self._restore_optimizer_state(checkpoint)
        return int(checkpoint.get("step", 0))

    def _response_stats(
        self,
        model: Any,
        data: DataProto,
        *,
        calculate_entropy: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        device_batch = self._data_on_device(data)
        autocast_dtype = self._forward_autocast_dtype()
        with torch.autocast(
            device_type=self.device.type,
            dtype=autocast_dtype,
            enabled=autocast_dtype is not None,
        ):
            return compute_response_stats(
                model,
                device_batch.batch["prompt_ids"],
                device_batch.batch["prompt_attention_mask"],
                device_batch.batch["responses"],
                device_batch.batch["response_mask"],
                self.pad_token_id,
                self.device,
                calculate_entropy=calculate_entropy,
                use_remove_padding=(
                    bool(getattr(self.system_cfg.model, "use_remove_padding", False))
                    if self.system_cfg is not None
                    else False
                ),
            )

    @staticmethod
    def _iter_data_micro_batches(data: DataProto, micro_batch_size: int | None) -> list[DataProto]:
        if micro_batch_size is None or micro_batch_size <= 0 or len(data) <= micro_batch_size:
            return [data]
        return [data[start : start + micro_batch_size] for start in range(0, len(data), micro_batch_size)]

    def _dynamic_micro_batch_token_budget(self, configured_limit: int | None) -> int | None:
        if configured_limit is not None:
            return int(configured_limit)
        if self.system_cfg is None:
            return None
        model_max_length = int(getattr(self.system_cfg.model, "max_length", 0) or 0)
        if model_max_length <= 0:
            return None
        return min(model_max_length, _DEFAULT_DYNAMIC_MICRO_BATCH_MAX_TOKENS)

    @staticmethod
    def _sequence_lengths(data: DataProto) -> list[int] | None:
        prompt_mask = data.batch.get("prompt_attention_mask")
        response_mask = data.batch.get("response_mask")
        if prompt_mask is None or response_mask is None:
            return None
        lengths = prompt_mask.long().sum(dim=1) + response_mask.long().sum(dim=1)
        return [max(int(length.item()), 1) for length in lengths]

    @staticmethod
    def _attach_micro_batch_indices(data: DataProto, indices: list[int]) -> DataProto:
        micro_batch = data[indices]
        micro_batch.meta_info[_MICRO_BATCH_INDICES_KEY] = list(indices)
        return micro_batch

    def _iter_token_limited_micro_batches(self, data: DataProto, max_token_len: int) -> list[DataProto]:
        lengths = self._sequence_lengths(data)
        if lengths is None or len(data) <= 1:
            return self._iter_data_micro_batches(data, None)
        ordered_indices = sorted(range(len(lengths)), key=lambda idx: (lengths[idx], idx), reverse=True)
        micro_batches: list[DataProto] = []
        current: list[int] = []
        current_max_len = 0

        def flush_current() -> None:
            nonlocal current, current_max_len
            if current:
                micro_batches.append(self._attach_micro_batch_indices(data, current))
                current = []
                current_max_len = 0

        for idx in ordered_indices:
            seq_len = lengths[idx]
            next_max_len = max(current_max_len, seq_len)
            next_padded_tokens = next_max_len * (len(current) + 1)
            if current and next_padded_tokens > max_token_len:
                flush_current()
                next_max_len = seq_len
            current.append(idx)
            current_max_len = next_max_len
            if current_max_len * len(current) >= max_token_len:
                flush_current()
        flush_current()
        return micro_batches

    def _iter_policy_micro_batches(
        self,
        data: DataProto,
        micro_batch_size: int | None,
        *,
        use_dynamic_bsz: bool | None = None,
        max_token_len: int | None = None,
    ) -> list[DataProto]:
        if use_dynamic_bsz is None:
            use_dynamic_bsz = bool(
                self.system_cfg is not None and getattr(self.system_cfg.model, "use_remove_padding", False)
            )
        if use_dynamic_bsz and max_token_len is not None and max_token_len > 0:
            return self._iter_token_limited_micro_batches(data, int(max_token_len))
        return self._iter_data_micro_batches(data, micro_batch_size)

    @staticmethod
    def _collect_ordered_stats(
        data: DataProto,
        chunks: list[tuple[DataProto, torch.Tensor, torch.Tensor]],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if not chunks:
            return torch.empty(0), torch.empty(0)
        first_log_probs = chunks[0][1].detach().cpu()
        first_entropy = chunks[0][2].detach().cpu()
        has_index_metadata = all(_MICRO_BATCH_INDICES_KEY in micro_batch.meta_info for micro_batch, _, _ in chunks)
        if not has_index_metadata:
            return (
                torch.cat([log_probs.detach().cpu() for _, log_probs, _ in chunks], dim=0),
                torch.cat([entropy.detach().cpu() for _, _, entropy in chunks], dim=0),
            )
        log_probs_out = torch.empty((len(data), *first_log_probs.shape[1:]), dtype=first_log_probs.dtype)
        entropy_out = torch.empty((len(data), *first_entropy.shape[1:]), dtype=first_entropy.dtype)
        for micro_batch, log_probs, entropy in chunks:
            indices = micro_batch.meta_info[_MICRO_BATCH_INDICES_KEY]
            log_probs_out[indices] = log_probs.detach().cpu()
            entropy_out[indices] = entropy.detach().cpu()
        return log_probs_out, entropy_out

    def _iter_sync_weights(self, offload_to_cpu: bool = False):
        if self.use_fsdp:
            state = get_model_state_dict(
                self.model,
                options=StateDictOptions(full_state_dict=False, cpu_offload=False),
            )
            try:
                for name, tensor in state.items():
                    sync_tensor = _sync_weight_tensor(
                        tensor,
                        device=self.device,
                        offload_to_cpu=offload_to_cpu,
                    )
                    if self.rank == 0:
                        yield name, sync_tensor
            finally:
                state.clear()
            return
        assert self.model is not None
        state = self.model.state_dict()
        for name, tensor in state.items():
            yield name, _sync_weight_tensor(tensor, device=self.device, offload_to_cpu=offload_to_cpu)

    def _collect_checkpoint_state_dict(self, offload_to_cpu: bool = True) -> dict[str, torch.Tensor]:
        if self.use_fsdp:
            state = get_model_state_dict(
                self.model,
                options=StateDictOptions(full_state_dict=True, cpu_offload=True),
            )
            if self.rank != 0:
                return {}
            return {
                name: _detach_state_tensor(tensor, device=self.device, offload_to_cpu=offload_to_cpu)
                for name, tensor in state.items()
            }
        assert self.model is not None
        state = self.model.state_dict()
        return {
            name: tensor.detach().cpu() if offload_to_cpu else tensor.detach().to(self.device).contiguous()
            for name, tensor in state.items()
        }

    def compute_log_prob(self, data: DataProto, micro_batch_size: int | None = None) -> DataProto:
        self.model.eval()
        chunks: list[tuple[DataProto, torch.Tensor, torch.Tensor]] = []
        use_dynamic_bsz = bool(
            self.system_cfg is not None
            and self.system_cfg.model.use_remove_padding
            and self.system_cfg.rollout.log_prob_use_dynamic_bsz
        )
        max_token_len = (
            self._dynamic_micro_batch_token_budget(self.system_cfg.rollout.log_prob_max_token_len_per_gpu)
            if self.system_cfg is not None
            else None
        )
        with torch.no_grad():
            for micro_batch in self._iter_policy_micro_batches(
                data,
                micro_batch_size,
                use_dynamic_bsz=use_dynamic_bsz,
                max_token_len=max_token_len,
            ):
                log_probs, entropy = self._response_stats(self.model, micro_batch, calculate_entropy=False)
                chunks.append((micro_batch, log_probs, entropy))
        log_probs_out, entropy_out = self._collect_ordered_stats(data, chunks)
        return DataProto.from_dict(
            {"old_log_probs": log_probs_out, "entropy": entropy_out},
        )

    def compute_ref_log_prob(self, data: DataProto, micro_batch_size: int | None = None) -> DataProto:
        if self.reference_model is None:
            return DataProto()
        chunks: list[tuple[DataProto, torch.Tensor, torch.Tensor]] = []
        use_dynamic_bsz = bool(
            self.system_cfg is not None
            and self.system_cfg.model.use_remove_padding
            and self.system_cfg.rollout.log_prob_use_dynamic_bsz
        )
        max_token_len = (
            self._dynamic_micro_batch_token_budget(self.system_cfg.rollout.log_prob_max_token_len_per_gpu)
            if self.system_cfg is not None
            else None
        )
        with torch.no_grad():
            for micro_batch in self._iter_policy_micro_batches(
                data,
                micro_batch_size,
                use_dynamic_bsz=use_dynamic_bsz,
                max_token_len=max_token_len,
            ):
                log_probs, _ = self._response_stats(self.reference_model, micro_batch, calculate_entropy=False)
                chunks.append((micro_batch, log_probs, torch.zeros_like(log_probs)))
        log_probs_out, _ = self._collect_ordered_stats(data, chunks)
        return DataProto.from_dict({"ref_log_probs": log_probs_out})

    def _policy_micro_step(
        self,
        micro_batch: DataProto,
        total_tokens: float,
        debug_train: bool,
    ) -> tuple[dict[str, float], float]:
        old_log_probs = micro_batch.batch["old_log_probs"]
        advantages = micro_batch.batch["advantages"]
        response_mask = micro_batch.batch["response_mask"]
        entropy_coef = float(self.backend_cfg.get("entropy_coef", 0.0))
        log_probs, entropy = self._response_stats(
            self.model,
            micro_batch,
            calculate_entropy=True,
        )
        chunk_loss, aux = compute_policy_loss(
            old_log_prob=old_log_probs,
            log_prob=log_probs,
            advantages=advantages,
            response_mask=response_mask,
            clip_low=float(self.backend_cfg.get("clip_low", 0.2)),
            clip_high=float(self.backend_cfg.get("clip_high", 0.2)),
            clip_ratio_c=float(self.backend_cfg.get("clip_ratio_c", 3.0)),
        )
        mask = response_mask.float()
        token_count = mask.sum()
        denom = token_count.clamp_min(1.0)
        token_weight = float(token_count.item())
        entropy_mean = (entropy * mask).sum() / denom
        if entropy_coef != 0.0:
            chunk_loss = chunk_loss - entropy_coef * entropy_mean
        if debug_train:
            print("[actor] update_policy backward", flush=True)
        (chunk_loss * (token_weight / total_tokens)).backward()
        metrics = {
            "loss": float(chunk_loss.item()),
            "entropy": float(entropy_mean.item()),
            "approx_kl": float(aux["approx_kl"]),
            "clipfrac": float(aux["clipfrac"]),
        }
        return metrics, token_weight

    def update_policy(self, data: DataProto, micro_batch_size: int | None = None) -> dict[str, float]:
        assert self.optimizer is not None
        assert self.system_cfg is not None
        self._ensure_optimizer_state_for_training()
        self.model.train()
        debug_train = os.environ.get("NANOVERL_DEBUG_TRAIN") == "1" and self.rank == 0
        if debug_train:
            print("[actor] update_policy forward", flush=True)
        local_tokens = float(data.batch["response_mask"].sum().item())
        local_tokens = max(local_tokens, 1.0)
        micro_batch_size = micro_batch_size or len(data)
        self.optimizer.zero_grad(set_to_none=True)
        weighted_metrics = {
            "loss": 0.0,
            "entropy": 0.0,
            "approx_kl": 0.0,
            "clipfrac": 0.0,
        }
        backward_tokens = float(data.meta_info.get("global_response_tokens", local_tokens))
        backward_tokens = max(backward_tokens / max(self.world_size, 1), 1.0)
        use_dynamic_bsz = bool(self.system_cfg.model.use_remove_padding and self.system_cfg.actor.use_dynamic_bsz)
        max_token_len = self._dynamic_micro_batch_token_budget(self.system_cfg.actor.ppo_max_token_len_per_gpu)
        for micro_batch in self._iter_policy_micro_batches(
            data,
            micro_batch_size,
            use_dynamic_bsz=use_dynamic_bsz,
            max_token_len=max_token_len,
        ):
            micro_batch = micro_batch.to(self.device)
            micro_metrics, token_weight = self._policy_micro_step(micro_batch, backward_tokens, debug_train)
            for key, value in micro_metrics.items():
                weighted_metrics[key] += value * token_weight
        if debug_train:
            print("[actor] update_policy clip_grad", flush=True)
        grad_norm = clip_grad_norm_(self.model.parameters(), self.system_cfg.actor.max_grad_norm)
        if debug_train:
            print("[actor] update_policy optimizer_step", flush=True)
        self.optimizer.step()
        if self.system_cfg.actor.fsdp_config.optimizer_offload:
            self._move_optimizer_state(torch.device("cpu"))
        if debug_train:
            print("[actor] update_policy done", flush=True)
        metrics = {
            "loss": weighted_metrics["loss"] / local_tokens,
            "entropy": weighted_metrics["entropy"] / local_tokens,
            "grad_norm": float(grad_norm.item() if isinstance(grad_norm, torch.Tensor) else grad_norm),
        }
        metrics["approx_kl"] = weighted_metrics["approx_kl"] / local_tokens
        metrics["clipfrac"] = weighted_metrics["clipfrac"] / local_tokens
        self.optimizer.zero_grad(set_to_none=True)
        if self._should_clear_cache_after_update():
            self._clear_cuda_cache()
        return metrics

    async def update_rollout_weights(self, version: int, rollout_handles: list[Any]) -> int:
        try:
            if self.system_cfg is not None and self.system_cfg.actor.fsdp_config.optimizer_offload:
                self.release_optimizer_state_for_rollout()
            if self.rank != 0:
                if self.use_fsdp:
                    for _ in rollout_handles:
                        for _ in self._iter_sync_weights(offload_to_cpu=False):
                            pass
                return 0

            count = 0
            bucket_size_mb = int(self.system_cfg.rollout.weight_sync_bucket_mb) if self.system_cfg is not None else 16
            transport = self.system_cfg.rollout.weight_sync_transport if self.system_cfg is not None else "auto"
            for rollout_handle in rollout_handles:
                sent_count = 0

                def counted_weights():
                    nonlocal sent_count
                    for item in self._iter_sync_weights(offload_to_cpu=False):
                        sent_count += 1
                        yield item

                sync_engine = NaiveSyncEngine(
                    bucket_size_mb=bucket_size_mb,
                    transport=transport,
                )
                await sync_engine.send_weights(counted_weights(), rollout_handle=rollout_handle, version=version)
                if count == 0:
                    count = sent_count
            return count
        finally:
            self._clear_cuda_cache()

    def _ensure_sync_engine(self) -> NCCLSyncEngine:
        assert self.system_cfg is not None
        backend = str(self.system_cfg.sync.backend).lower().strip()
        if backend != "nccl":
            raise RuntimeError(
                f"FSDPActorWorker sync engine requires sync.backend='nccl', got {self.system_cfg.sync.backend}"
            )
        if self.sync_engine is None:
            self.sync_engine = NCCLSyncEngine(
                bucket_size_mb=int(self.system_cfg.sync.bucket_size_mb),
                group_name=self.system_cfg.sync.group_name,
                rebuild_group=bool(self.system_cfg.sync.rebuild_group),
            )
        return self.sync_engine

    def prepare(self) -> Any:
        return self._ensure_sync_engine().prepare(create_master_metadata=True)

    def init_process_group(self, rank: int, world_size: int, master_metadata: Any = None) -> None:
        self.sync_rank = int(rank)
        if self.sync_rank < 0:
            return None
        self._ensure_sync_engine().init_process_group(
            rank=self.sync_rank,
            world_size=int(world_size),
            master_metadata=master_metadata,
        )
        return None

    async def update_weights(self, version: int) -> int:
        _ = version
        try:
            if self.system_cfg is not None and self.system_cfg.actor.fsdp_config.optimizer_offload:
                self.release_optimizer_state_for_rollout()
            if self.sync_rank is not None and self.sync_rank < 0:
                if self.use_fsdp:
                    for _ in self._iter_sync_weights(offload_to_cpu=False):
                        pass
                return 0
            if self.sync_engine is None:
                if self.rank == 0:
                    raise RuntimeError("prepare/init_process_group must be called before update_weights")
                return 0
            count = 0

            def counted_weights():
                nonlocal count
                for item in self._iter_sync_weights(offload_to_cpu=False):
                    count += 1
                    yield item

            await self.sync_engine.send_weights(counted_weights())
            return count
        finally:
            self._clear_cuda_cache()

    def finalize(self) -> None:
        if self.sync_engine is not None:
            self.sync_engine.finalize()
            self.sync_engine = None
        self.sync_rank = None

    def save_checkpoint(self, path: str, step: int) -> None:
        state_dict: dict[str, torch.Tensor] | None = None
        try:
            state_dict = self._collect_checkpoint_state_dict()
            if self.rank != 0:
                return
            checkpoint_dir = Path(path)
            checkpoint_dir.mkdir(parents=True, exist_ok=True)
            # 这里直接导出成 HuggingFace checkpoint，便于后续推理和迁移使用。
            self.model.save_pretrained(
                str(checkpoint_dir),
                state_dict=state_dict,
                safe_serialization=True,
                max_shard_size="5GB",
            )
            torch.save(
                {
                    "step": step,
                    "optimizer": self.optimizer.state_dict() if self.optimizer is not None else {},
                },
                checkpoint_dir / "trainer_state.pt",
            )
        finally:
            if state_dict is not None:
                state_dict.clear()

    def load_checkpoint(self, path: str) -> int:
        checkpoint_path = Path(path)
        if checkpoint_path.is_file():
            checkpoint = torch.load(checkpoint_path, map_location="cpu")
            state = checkpoint["model"]
            if self.use_fsdp:
                state = self._broadcast_state(state)
            return self._apply_checkpoint(state, checkpoint)

        state, checkpoint = self._load_hf_checkpoint_dir(checkpoint_path)
        if self.use_fsdp:
            state = self._broadcast_state(state)
            checkpoint = self._broadcast_state(checkpoint)
        return self._apply_checkpoint(state, checkpoint)

    def shutdown(self) -> None:
        try:
            if dist.is_available() and dist.is_initialized():
                dist.destroy_process_group()
        except Exception:
            pass
        self.model = None
        self.reference_model = None
        self.optimizer = None
        self.finalize()
        self._clear_cuda_cache()
