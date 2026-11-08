"""Experiment configuration and metrics persistence (JSON, JSONL, and optional W&B)."""

from datetime import datetime, timezone
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml


class ExperimentTracker:
    """
    Saves resolved experiment configurations and training/evaluation metrics to disk,
    and optionally logs to Weights & Biases on rank 0.
    """

    def __init__(
        self,
        output_dir: str | Path,
        config_dict: Dict[str, Any],
        global_rank: int = 0,
        wandb_run: Optional[Any] = None,
    ):
        self.output_dir = Path(os.path.expanduser(str(output_dir)))
        self.config_dict = config_dict
        self.global_rank = global_rank
        self.wandb_run = wandb_run
        self.history: List[Dict[str, Any]] = []

        self.config_json_path = self.output_dir / "experiment_config.json"
        self.config_yaml_path = self.output_dir / "experiment_config.yaml"
        self.metrics_jsonl_path = self.output_dir / "metrics.jsonl"
        self.summary_json_path = self.output_dir / "metrics_summary.json"

        if self.global_rank == 0:
            self.output_dir.mkdir(parents=True, exist_ok=True)
            self.save_config()

    def save_config(self) -> None:
        """Persist the full resolved experiment configuration as JSON and YAML."""
        if self.global_rank != 0:
            return
        serializable = _make_json_serializable(self.config_dict)
        with open(self.config_json_path, "w", encoding="utf-8") as f:
            json.dump(serializable, f, indent=2)
        with open(self.config_yaml_path, "w", encoding="utf-8") as f:
            yaml.safe_dump(serializable, f, sort_keys=False)

    def log_metrics(self, step: int, metrics: Dict[str, Any]) -> None:
        """Record metrics for a training window/step to disk and optional W&B."""
        if self.global_rank != 0:
            return

        record = {
            "step": int(step),
            "timestamp": datetime.now(timezone.utc).isoformat(),
            **_make_json_serializable(metrics),
        }
        self.history.append(record)

        with open(self.metrics_jsonl_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")

        if self.wandb_run is not None:
            self.wandb_run.log(metrics, step=step)

    def save_summary(self, extra_summary: Optional[Dict[str, Any]] = None) -> None:
        """Write a summary JSON file containing config and final/best metrics."""
        if self.global_rank != 0:
            return

        val_entries = [
            (r["step"], r["eval/valid_loss"])
            for r in self.history
            if "eval/valid_loss" in r and r["eval/valid_loss"] is not None
        ]
        best_valid_round = None
        best_valid_loss = None
        if val_entries:
            best_valid_round, best_valid_loss = min(val_entries, key=lambda x: x[1])

        summary = {
            "updated_at": datetime.now(timezone.utc).isoformat(),
            "num_logged_steps": len(self.history),
            "best_valid_loss": best_valid_loss,
            "best_valid_round": best_valid_round,
            "final_metrics": self.history[-1] if self.history else None,
            "config": _make_json_serializable(self.config_dict),
        }
        if extra_summary:
            summary.update(_make_json_serializable(extra_summary))

        with open(self.summary_json_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)


def _make_json_serializable(obj: Any) -> Any:
    """Recursively convert objects into JSON/YAML-serializable primitives."""
    if isinstance(obj, dict):
        return {
            str(k): _make_json_serializable(v)
            for k, v in obj.items()
            if not str(k).startswith("_")
        }
    if isinstance(obj, (list, tuple)):
        return [_make_json_serializable(v) for v in obj]
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, (int, float, str, bool)) or obj is None:
        return obj
    return str(obj)
