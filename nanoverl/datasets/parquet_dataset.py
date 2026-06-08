from __future__ import annotations

import random
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader as TorchDataLoader
from torch.utils.data import Dataset


class DataLoader(Dataset):
    def __init__(
        self,
        dataset_source: str,
        tokenizer: Any,
        prompt_template: str,
        max_prompt_length: int,
        batch_size: int,
        prompt_key: str = "question",
        apply_chat_template_kwargs: dict[str, Any] | None = None,
        limit: int | None = None,
        shuffle: bool = False,
        num_workers: int = 0,
        random_sample: bool = False,
        seed: int = 42,
    ) -> None:
        self.dataset_source = Path(dataset_source)
        self.tokenizer = tokenizer
        self.prompt_template = prompt_template
        self.max_prompt_length = max_prompt_length
        self.batch_size = batch_size
        self.prompt_key = prompt_key
        self.apply_chat_template_kwargs = dict(apply_chat_template_kwargs or {})
        self.limit = limit
        self.shuffle = shuffle
        self.num_workers = num_workers
        self.random_sample = random_sample
        self.seed = int(seed)
        self.records = self._load_records()

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict[str, Any]:
        record = self.records[index]
        question = record[self.prompt_key]
        prompt_text = self._build_prompt(question)
        tokens = self.tokenizer(
            prompt_text,
            truncation=False,
            add_special_tokens=True,
            return_tensors="pt",
        )
        return {
            "prompt_ids": tokens["input_ids"].squeeze(0),
            "prompt_attention_mask": tokens["attention_mask"].squeeze(0),
            "question": question,
            "ground_truth": record["answer"],
            "data_source": record.get("data_source", "gsm8k"),
            "prompt_text": prompt_text,
            "sample_id": record.get("sample_id", str(index)),
        }

    def build(self) -> TorchDataLoader:
        generator = None
        if self.shuffle:
            generator = torch.Generator()
            generator.manual_seed(self.seed)
        return TorchDataLoader(
            self,
            batch_size=self.batch_size,
            shuffle=self.shuffle,
            num_workers=self.num_workers,
            collate_fn=self._collate,
            drop_last=False,
            generator=generator,
        )

    def _load_records(self) -> list[dict[str, Any]]:
        records = pq.read_table(str(self.dataset_source)).to_pylist()
        if self.random_sample:
            count = len(records) if self.limit is None else min(self.limit, len(records))
            records = random.Random(self.seed).sample(records, count)
            return self._filter_overlong_prompts(records)
        if self.limit is not None:
            records = records[: self.limit]
        return self._filter_overlong_prompts(records)

    @staticmethod
    def _input_ids_length(input_ids: Any) -> int:
        if isinstance(input_ids, torch.Tensor):
            return int(input_ids.shape[-1])
        if input_ids and isinstance(input_ids[0], list):
            return len(input_ids[0])
        return len(input_ids)

    def _prompt_length(self, prompt_text: str) -> int:
        tokens = self.tokenizer(
            prompt_text,
            truncation=False,
            add_special_tokens=True,
        )
        return self._input_ids_length(tokens["input_ids"])

    def _filter_overlong_prompts(self, records: list[dict[str, Any]]) -> list[dict[str, Any]]:
        return [
            record
            for record in records
            if self._prompt_length(self._build_prompt(record[self.prompt_key])) <= self.max_prompt_length
        ]

    def _build_prompt(self, prompt_value: Any) -> str:
        if isinstance(prompt_value, list):
            return str(
                self.tokenizer.apply_chat_template(
                    prompt_value,
                    tokenize=False,
                    add_generation_prompt=True,
                    **self.apply_chat_template_kwargs,
                )
            )
        prompt_text = str(prompt_value)
        return self.prompt_template.format(**{self.prompt_key: prompt_text, "question": prompt_text})

    @staticmethod
    def _collate(data_list: list[dict[str, Any]]) -> dict[str, Any]:
        prompt_ids = [item["prompt_ids"] for item in data_list]
        prompt_mask = [item["prompt_attention_mask"] for item in data_list]
        return {
            "prompt_ids": pad_sequence(prompt_ids, batch_first=True, padding_value=0),
            "prompt_attention_mask": pad_sequence(prompt_mask, batch_first=True, padding_value=0),
            "question": [item["question"] for item in data_list],
            "ground_truth": [item["ground_truth"] for item in data_list],
            "data_source": [item["data_source"] for item in data_list],
            "prompt_text": [item["prompt_text"] for item in data_list],
            "sample_id": [item["sample_id"] for item in data_list],
        }
