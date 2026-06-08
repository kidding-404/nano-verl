from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import torch

TensorBatch = dict[str, torch.Tensor]
NonTensorBatch = dict[str, list[Any]]
MetaInfo = dict[str, Any]
Index = int | slice | list[int]


def _slice_non_tensor(value: Any, item: slice | list[int]) -> Any:
    if isinstance(item, slice):
        return value[item]
    return [value[i] for i in item]


@dataclass
class DataProto:
    batch: TensorBatch = field(default_factory=dict)
    non_tensor_batch: NonTensorBatch = field(default_factory=dict)
    meta_info: MetaInfo = field(default_factory=dict)

    @classmethod
    def from_mixed_dict(
        cls,
        data: dict[str, Any],
        meta_info: MetaInfo | None = None,
    ) -> "DataProto":
        tensors: TensorBatch = {}
        non_tensors: NonTensorBatch = {}
        for key, value in data.items():
            if isinstance(value, torch.Tensor):
                tensors[key] = value
            else:
                non_tensors[key] = list(value) if isinstance(value, (list, tuple)) else [value]
        return cls(batch=tensors, non_tensor_batch=non_tensors, meta_info=meta_info or {})

    @classmethod
    def from_dict(
        cls,
        tensors: TensorBatch | None,
        non_tensors: NonTensorBatch | None = None,
        meta_info: MetaInfo | None = None,
    ) -> "DataProto":
        return cls(batch=tensors or {}, non_tensor_batch=non_tensors or {}, meta_info=meta_info or {})

    def __len__(self) -> int:
        if self.batch:
            return next(iter(self.batch.values())).shape[0]
        if self.non_tensor_batch:
            return len(next(iter(self.non_tensor_batch.values())))
        return 0

    def __getitem__(self, item: Index) -> "DataProto":
        if isinstance(item, int):
            if item < 0:
                item += len(self)
            item = slice(item, item + 1)
        tensors = {key: value[item] for key, value in self.batch.items()}
        non_tensors = {key: _slice_non_tensor(value, item) for key, value in self.non_tensor_batch.items()}
        return DataProto.from_dict(tensors, non_tensors, dict(self.meta_info))

    def select(
        self,
        batch_keys: list[str],
        non_tensor_keys: list[str] | None = None,
    ) -> "DataProto":
        tensors = {key: self.batch[key] for key in batch_keys if key in self.batch}
        non_tensors = {
            key: self.non_tensor_batch[key]
            for key in (non_tensor_keys or [])
            if key in self.non_tensor_batch
        }
        return DataProto.from_dict(tensors, non_tensors, dict(self.meta_info))

    def union(self, other: "DataProto") -> "DataProto":
        return DataProto.from_dict(
            tensors={**self.batch, **other.batch},
            non_tensors={**self.non_tensor_batch, **other.non_tensor_batch},
            meta_info={**self.meta_info, **other.meta_info},
        )

    def repeat(self, times: int, interleave: bool = True) -> "DataProto":
        if times <= 1 or len(self) == 0:
            return self
        tensors: TensorBatch = {}
        non_tensors: NonTensorBatch = {}
        for key, value in self.batch.items():
            if interleave:
                tensors[key] = value.repeat_interleave(times, dim=0)
            else:
                tensors[key] = torch.cat([value] * times, dim=0)
        for key, value in self.non_tensor_batch.items():
            if interleave:
                non_tensors[key] = [item for item in value for _ in range(times)]
            else:
                non_tensors[key] = value * times
        return DataProto.from_dict(tensors, non_tensors, dict(self.meta_info))

    @staticmethod
    def concat(protos: list["DataProto"]) -> "DataProto":
        if not protos:
            return DataProto()
        tensor_keys = set().union(*(proto.batch.keys() for proto in protos))
        non_tensor_keys = set().union(*(proto.non_tensor_batch.keys() for proto in protos))
        tensors = {
            key: torch.cat([proto.batch[key] for proto in protos if key in proto.batch], dim=0)
            for key in tensor_keys
        }
        non_tensors = {
            key: [item for proto in protos for item in proto.non_tensor_batch.get(key, [])]
            for key in non_tensor_keys
        }
        meta_info = dict(protos[0].meta_info)
        return DataProto.from_dict(tensors, non_tensors, meta_info)

    def to(self, device: Any) -> "DataProto":
        tensors = {key: value.to(device) for key, value in self.batch.items()}
        return DataProto.from_dict(tensors, dict(self.non_tensor_batch), dict(self.meta_info))


@dataclass
class TokenOutput:
    token_ids: list[int]
    log_probs: list[float] | None = None
    stop_reason: str | None = None
    model_version: int | None = None
