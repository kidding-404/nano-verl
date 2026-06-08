from __future__ import annotations

import json
import math
import os
from importlib import import_module
from pathlib import Path
from typing import Any

_SUPPORTED_BACKENDS = {"console", "wandb", "swanlab"}
_API_KEY_ENV = {
    "wandb": "WANDB_API_KEY",
    "swanlab": "SWANLAB_API_KEY",
}
_SAMPLE_FIELDS = ("input", "output", "score")


def _truncate(text: str, limit: int) -> str:
    text = text.replace("\n", " ").strip()
    if len(text) <= limit:
        return text
    return text[: limit - 3] + "..."


def _write_jsonl(path: Path, rows: list[dict[str, Any]], *, default: Any | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as file:
        for row in rows:
            file.write(json.dumps(row, ensure_ascii=False, default=default) + "\n")


class ExperimentLogger:
    def __init__(
        self,
        backend: str = "console",
        output_dir: str = "outputs",
        project_name: str = "nanoverl",
        experiment_name: str = "run",
        run_config: dict[str, Any] | None = None,
        api_key: str | None = None,
    ) -> None:
        self.project_name = str(project_name)
        self.experiment_name = str(experiment_name)
        self.backend = self._normalize_backend(backend)
        self.run_config = dict(run_config or {})
        self.api_key = api_key
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.sample_dir = self.output_dir / "samples"
        self.trajectory_dir = self.output_dir / "trajectories"
        self._module: Any | None = None
        self._run: Any | None = None
        self._init_backend()

    @staticmethod
    def _normalize_backend(backend: str) -> str:
        normalized = backend.lower().strip()
        if normalized not in _SUPPORTED_BACKENDS:
            raise ValueError(f"Unsupported logger backend: {backend}")
        return normalized

    def _resolve_api_key(self) -> str | None:
        if self.api_key:
            return self.api_key
        env_name = _API_KEY_ENV.get(self.backend)
        return os.getenv(env_name) if env_name else None

    def _init_backend(self) -> None:
        if self.backend == "console":
            return
        try:
            self._module = import_module(self.backend)
        except ImportError:
            print(f"[logger] backend={self.backend} unavailable, falling back to console", flush=True)
            self.backend = "console"
            return
        api_key = self._resolve_api_key()
        if api_key and hasattr(self._module, "login"):
            if self.backend == "swanlab":
                self._module.login(api_key=api_key)
            else:
                self._module.login(key=api_key)
        if self.backend == "wandb":
            self._run = self._module.init(
                project=self.project_name,
                name=self.experiment_name,
                config=self.run_config,
                dir=str(self.output_dir),
            )
            return
        self._run = self._module.init(
            project=self.project_name,
            experiment_name=self.experiment_name,
            config=self.run_config,
            logdir=str(self.output_dir / "swanlab"),
        )

    def _require_run(self) -> Any:
        if self._run is None:
            raise RuntimeError(f"Logger backend {self.backend} is not initialized")
        return self._run

    @staticmethod
    def _format_metric(value: float) -> str:
        if not math.isfinite(value):
            return str(value)
        abs_value = abs(value)
        if abs_value == 0.0:
            return "0"
        if abs_value >= 1e-3:
            return f"{value:.4f}"
        return f"{value:.3e}"

    def _normalize_samples(self, samples: list[Any]) -> list[dict[str, Any]]:
        normalized: list[dict[str, Any]] = []
        for sample in samples:
            if not isinstance(sample, (list, tuple)) or len(sample) != 3:
                raise ValueError("Each sample must be [input, output, score]")
            prompt, output, score = sample
            normalized.append(
                {
                    "input": str(prompt),
                    "output": str(output),
                    "score": float(score),
                }
            )
        return normalized

    def _console_log_samples(self, normalized: list[dict[str, Any]], step: int, sample_path: Path) -> None:
        print(f"[samples] step={step:04d} saved={sample_path}", flush=True)
        for idx, row in enumerate(normalized):
            print(
                f"[samples] idx={idx} score={self._format_metric(row['score'])} "
                f"input={_truncate(row['input'], 160)}",
                flush=True,
            )
            print(f"[samples] output={_truncate(row['output'], 240)}", flush=True)

    def _build_remote_sample_payload(self, normalized: list[dict[str, Any]]) -> dict[str, Any]:
        if self.backend == "wandb" and self._module is not None and hasattr(self._module, "Table"):
            table = self._module.Table(columns=list(_SAMPLE_FIELDS))
            for row in normalized:
                table.add_data(row["input"], row["output"], row["score"])
            return {"reward_samples": table}
        if self.backend == "swanlab":
            text = "\n\n".join(
                f"[{idx}] score={self._format_metric(row['score'])}\n"
                f"input: {row['input']}\noutput: {row['output']}"
                for idx, row in enumerate(normalized)
            )
            text_type = getattr(self._module, "Text", None) if self._module is not None else None
            return {"reward_samples_text": text_type(text) if text_type is not None else text}
        return {"reward_samples": normalized}

    def log_metrics(self, step: int, metrics: dict[str, Any]) -> None:
        ordered = {key: float(value) for key, value in metrics.items()}
        if self.backend == "console":
            text = " ".join(f"{key}={self._format_metric(value)}" for key, value in ordered.items())
            print(f"[metrics] step={step:04d} {text}", flush=True)
            return
        self._require_run().log(ordered, step=step)

    def log_samples(self, step: int, samples: list[Any]) -> None:
        normalized = self._normalize_samples(samples)
        if not normalized:
            return
        sample_path = self.sample_dir / f"step_{step:06d}.jsonl"
        _write_jsonl(sample_path, normalized)
        if self.backend == "console":
            self._console_log_samples(normalized, step, sample_path)
            return
        self._require_run().log(self._build_remote_sample_payload(normalized), step=step)

    def log_trajectories(self, step: int, rows: list[dict[str, Any]]) -> None:
        if not rows:
            return
        dump_path = self.trajectory_dir / f"step_{step:06d}.jsonl"
        _write_jsonl(dump_path, rows, default=str)
        preview_rows = sorted(rows, key=lambda item: float(item.get("reward", 0.0)), reverse=True)[:2]
        preview_samples = [
            [row.get("prompt", ""), row.get("response", ""), float(row.get("reward", 0.0))]
            for row in preview_rows
        ]
        self.log_samples(step, preview_samples)

    def close(self) -> None:
        run = self._run
        self._run = None
        if run is not None and hasattr(run, "finish"):
            run.finish()
