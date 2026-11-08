#!/usr/bin/env python3
"""
Sequential benchmark runner comparing DiLoCo, IsoLoCo, and Decoupled Spectral Allocation
on mC4 IID (15M TinyGPT, 4 workers, H=30, 100 outer rounds, seed=42, outer_lr=1.5).

Generates:
  - outputs/comparison_iid_lr1p5/convergence_rounds.png
  - outputs/comparison_iid_lr1p5/runtime_comparison.png
  - outputs/comparison_iid_lr1p5/convergence_time.png
  - outputs/comparison_iid_lr1p5/comparison.csv
"""

import argparse
import csv
from dataclasses import dataclass, field
from datetime import datetime
import json
import math
from pathlib import Path
import subprocess
import sys
import time
from typing import Any, Dict, List, Optional, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parent.parent
TRAIN_SCRIPT = REPO_ROOT / "train.py"
DEFAULT_CONFIG = REPO_ROOT / "configs" / "experiment.yaml"
DEFAULT_SUMMARY_DIR = REPO_ROOT / "outputs" / "comparison_iid_lr1p5"

METHODS_ORDER: List[str] = [
    "diloco",
    "isoloco",
    "spectral_alloc_decoupled",
]

METHOD_DISPLAY_NAMES: Dict[str, str] = {
    "diloco": "DiLoCo",
    "isoloco": "IsoLoCo",
    "spectral_alloc_decoupled": "Spectral Alloc Decoupled",
}

METHOD_COLORS: Dict[str, str] = {
    "diloco": "#1f77b4",
    "isoloco": "#ff7f0e",
    "spectral_alloc_decoupled": "#2ca02c",
}


@dataclass
class ValidationPoint:
    """Single validation evaluation point extracted from metrics.jsonl."""

    outer_round: int
    valid_loss: float
    elapsed_sec: float
    elapsed_min: float


@dataclass
class MethodBenchmarkResult:
    """Summary and validation trajectory for one benchmarked method."""

    method: str
    outer_learning_rate: float
    calibration_batch_size: Optional[int]
    final_valid_loss: float
    best_valid_loss: float
    total_runtime_sec: float
    total_runtime_min: float
    run_dir: Path
    exit_status: int
    validation_points: List[ValidationPoint] = field(default_factory=list)


def resolve_unique_run_dir(
    outputs_root: Path,
    dataset: str,
    model_size: str,
    method: str,
    partition: str,
    base_run_id: str,
) -> Tuple[str, str, Path]:
    """
    Construct a unique (run_id, run_name, run_dir) tuple under outputs_root so
    existing completed runs are never overwritten and stale metrics are never reused.
    """
    prefix = f"{dataset.lower()}_{model_size.lower()}_{method.lower()}_{partition.lower()}"
    run_id = base_run_id
    run_name = f"{prefix}_{run_id}"
    run_dir = outputs_root / run_name

    if run_dir.exists() and any(run_dir.iterdir()):
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_id = f"{base_run_id}_{ts}"
        run_name = f"{prefix}_{run_id}"
        run_dir = outputs_root / run_name
        suffix = 1
        while run_dir.exists() and any(run_dir.iterdir()):
            run_id = f"{base_run_id}_{ts}_{suffix}"
            run_name = f"{prefix}_{run_id}"
            run_dir = outputs_root / run_name
            suffix += 1

    return run_id, run_name, run_dir


def load_validation_trajectory(metrics_jsonl_path: Path) -> List[ValidationPoint]:
    """
    Parse actual validation loss measurements and their recorded monotonic elapsed
    timestamps from metrics.jsonl without interpolation, smoothing, or estimation.
    """
    if not metrics_jsonl_path.is_file():
        raise FileNotFoundError(f"Metrics file not found: {metrics_jsonl_path}")

    points: List[ValidationPoint] = []
    with open(metrics_jsonl_path, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            record: Dict[str, Any] = json.loads(line)
            if "eval/valid_loss" not in record or record["eval/valid_loss"] is None:
                continue

            val_loss = float(record["eval/valid_loss"])
            if not math.isfinite(val_loss):
                continue

            # step is 0-indexed window index; outer synchronization round is step + 1
            step_idx = int(record.get("step", record.get("train/window", 0)))
            outer_round = step_idx + 1

            elapsed_sec_raw = record.get(
                "eval/elapsed_sec", record.get("eval/elapsed_time_sec", None)
            )
            elapsed_min_raw = record.get(
                "eval/elapsed_min", record.get("eval/elapsed_time_min", None)
            )
            if elapsed_sec_raw is None and elapsed_min_raw is None:
                raise ValueError(
                    f"Missing recorded evaluation timestamp ('eval/elapsed_sec' / 'eval/elapsed_min') "
                    f"at line {line_num} (step {step_idx}) in {metrics_jsonl_path}."
                )

            if elapsed_sec_raw is not None:
                elapsed_sec = float(elapsed_sec_raw)
                elapsed_min = (
                    float(elapsed_min_raw)
                    if elapsed_min_raw is not None
                    else elapsed_sec / 60.0
                )
            else:
                elapsed_min = float(elapsed_min_raw)
                elapsed_sec = elapsed_min * 60.0

            points.append(
                ValidationPoint(
                    outer_round=outer_round,
                    valid_loss=val_loss,
                    elapsed_sec=elapsed_sec,
                    elapsed_min=elapsed_min,
                )
            )

    if not points:
        raise ValueError(
            f"No finite validation measurements ('eval/valid_loss') found in {metrics_jsonl_path}."
        )
    return points


def run_single_method(
    method: str,
    config_path: Path,
    outputs_root: Path,
    dataset: str = "mc4",
    partition: str = "iid",
    model_size: str = "15M",
    num_workers: int = 4,
    inner_steps: int = 30,
    max_steps: int = 100,
    seed: int = 42,
    outer_lr: float = 1.5,
    eval_frequency: int = 10,
    calibration_batch_size: int = 32,
) -> MethodBenchmarkResult:
    """Execute one training method sequentially in a subprocess and collect its results."""
    lr_tag = str(outer_lr).replace(".", "p")
    if method == "spectral_alloc_decoupled":
        base_run_id = (
            f"bench_lr{lr_tag}_cb{calibration_batch_size}_w{num_workers}"
            f"_h{inner_steps}_r{max_steps}_s{seed}"
        )
        calib_bs: Optional[int] = int(calibration_batch_size)
    else:
        base_run_id = (
            f"bench_lr{lr_tag}_w{num_workers}_h{inner_steps}_r{max_steps}_s{seed}"
        )
        calib_bs = None

    run_id, run_name, run_dir = resolve_unique_run_dir(
        outputs_root=outputs_root,
        dataset=dataset,
        model_size=model_size,
        method=method,
        partition=partition,
        base_run_id=base_run_id,
    )

    cmd: List[str] = [
        sys.executable,
        str(TRAIN_SCRIPT),
        "--config",
        str(config_path),
        "--method",
        method,
        "--dataset",
        dataset,
        "--partition",
        partition,
        "--model_size",
        model_size,
        "--num_workers",
        str(num_workers),
        "--virtual_workers",
        str(num_workers),
        "--inner_steps",
        str(inner_steps),
        "--max_steps",
        str(max_steps),
        "--seed",
        str(seed),
        "--outer_learning_rate",
        str(outer_lr),
        "--eval_frequency",
        str(eval_frequency),
        "--run_id",
        run_id,
        "--run_name",
        run_name,
        "--save_path",
        str(run_dir),
    ]
    if calib_bs is not None:
        cmd.extend(["--calibration_batch_size", str(calib_bs)])

    print("=" * 80)
    print(f"[Benchmark] Launching method '{method}' sequentially")
    print(f"[Benchmark] Command: {' '.join(cmd)}")
    print(f"[Benchmark] Run directory: {run_dir}")
    print("=" * 80, flush=True)

    t0 = time.perf_counter()
    completed = subprocess.run(cmd, cwd=str(REPO_ROOT), check=False)
    total_runtime_sec = time.perf_counter() - t0
    total_runtime_min = total_runtime_sec / 60.0
    exit_status = int(completed.returncode)

    if exit_status != 0:
        raise RuntimeError(
            f"Method '{method}' failed with exit status {exit_status} "
            f"(run_dir={run_dir}). Aborting benchmark comparison."
        )

    metrics_jsonl = run_dir / "metrics.jsonl"
    val_points = load_validation_trajectory(metrics_jsonl)
    final_val_loss = val_points[-1].valid_loss
    best_val_loss = min(pt.valid_loss for pt in val_points)

    print(
        f"[Benchmark] Completed '{method}' in {total_runtime_sec:.2f}s "
        f"({total_runtime_min:.2f} min) | final_val_loss={final_val_loss:.4f} | "
        f"best_val_loss={best_val_loss:.4f}",
        flush=True,
    )

    return MethodBenchmarkResult(
        method=method,
        outer_learning_rate=float(outer_lr),
        calibration_batch_size=calib_bs,
        final_valid_loss=final_val_loss,
        best_valid_loss=best_val_loss,
        total_runtime_sec=total_runtime_sec,
        total_runtime_min=total_runtime_min,
        run_dir=run_dir,
        exit_status=exit_status,
        validation_points=val_points,
    )


def plot_convergence_rounds(
    results: List[MethodBenchmarkResult], output_path: Path
) -> None:
    """Generate Figure 1: Validation loss vs. outer synchronization round."""
    fig, ax = plt.subplots(figsize=(8, 5), dpi=150)
    for res in results:
        rounds = [pt.outer_round for pt in res.validation_points]
        losses = [pt.valid_loss for pt in res.validation_points]
        label = METHOD_DISPLAY_NAMES.get(res.method, res.method)
        color = METHOD_COLORS.get(res.method, None)
        ax.plot(
            rounds,
            losses,
            marker="o",
            linewidth=2.0,
            markersize=5.0,
            label=label,
            color=color,
        )

    ax.set_xlabel("Outer Synchronization Round", fontsize=11)
    ax.set_ylabel("Validation Loss", fontsize=11)
    ax.set_title("Validation Loss vs. Outer Synchronization Round (mC4 15M IID)", fontsize=12)
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend(frameon=True, fontsize=10)
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def plot_runtime_comparison(
    results: List[MethodBenchmarkResult], output_path: Path
) -> None:
    """Generate Figure 2: Total wall-clock runtime bar chart in minutes."""
    fig, ax = plt.subplots(figsize=(7.5, 5), dpi=150)
    labels = [METHOD_DISPLAY_NAMES.get(r.method, r.method) for r in results]
    runtimes_min = [r.total_runtime_min for r in results]
    colors = [METHOD_COLORS.get(r.method, "#4c72b0") for r in results]

    bars = ax.bar(
        labels, runtimes_min, color=colors, width=0.55, edgecolor="black", alpha=0.9
    )
    max_val = max(runtimes_min) if runtimes_min else 1.0
    ax.set_ylim(0.0, max_val * 1.18)

    for bar, r in zip(bars, results):
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            height + max_val * 0.02,
            f"{r.total_runtime_min:.2f} min\n({r.total_runtime_sec:.1f}s)",
            ha="center",
            va="bottom",
            fontsize=9.5,
            fontweight="bold",
        )

    ax.set_xlabel("Method", fontsize=11)
    ax.set_ylabel("Total Wall-Clock Runtime (minutes)", fontsize=11)
    ax.set_title("Total Wall-Clock Runtime Comparison (mC4 15M IID)", fontsize=12)
    ax.grid(True, axis="y", linestyle="--", alpha=0.5)
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def plot_convergence_time(
    results: List[MethodBenchmarkResult], output_path: Path
) -> None:
    """
    Generate Figure 3: Validation loss vs. actual recorded elapsed wall-clock
    training time in minutes (no estimation, interpolation, or smoothing).
    """
    fig, ax = plt.subplots(figsize=(8, 5), dpi=150)
    for res in results:
        times_min = [pt.elapsed_min for pt in res.validation_points]
        losses = [pt.valid_loss for pt in res.validation_points]
        label = METHOD_DISPLAY_NAMES.get(res.method, res.method)
        color = METHOD_COLORS.get(res.method, None)
        ax.plot(
            times_min,
            losses,
            marker="o",
            linewidth=2.0,
            markersize=5.0,
            label=label,
            color=color,
        )

    ax.set_xlabel("Elapsed Wall-Clock Training Time (minutes)", fontsize=11)
    ax.set_ylabel("Validation Loss", fontsize=11)
    ax.set_title("Validation Loss vs. Elapsed Wall-Clock Time (mC4 15M IID)", fontsize=12)
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend(frameon=True, fontsize=10)
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def write_comparison_csv(
    results: List[MethodBenchmarkResult], csv_path: Path
) -> None:
    """Write comparison.csv with all required benchmark summary columns."""
    fieldnames = [
        "method",
        "outer_learning_rate",
        "calibration_batch_size",
        "final_validation_loss",
        "best_validation_loss",
        "total_runtime_seconds",
        "total_runtime_minutes",
        "run_directory",
        "exit_status",
    ]
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in results:
            writer.writerow(
                {
                    "method": r.method,
                    "outer_learning_rate": f"{r.outer_learning_rate:.4f}",
                    "calibration_batch_size": (
                        str(r.calibration_batch_size)
                        if r.calibration_batch_size is not None
                        else ""
                    ),
                    "final_validation_loss": f"{r.final_valid_loss:.6f}",
                    "best_validation_loss": f"{r.best_valid_loss:.6f}",
                    "total_runtime_seconds": f"{r.total_runtime_sec:.3f}",
                    "total_runtime_minutes": f"{r.total_runtime_min:.4f}",
                    "run_directory": str(r.run_dir),
                    "exit_status": r.exit_status,
                }
            )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Sequential benchmark runner for DiLoCo, IsoLoCo, and Spectral Alloc Decoupled."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG,
        help="Path to experiment YAML configuration (default: configs/experiment.yaml).",
    )
    parser.add_argument(
        "--summary_dir",
        type=Path,
        default=DEFAULT_SUMMARY_DIR,
        help="Directory to save comparison.csv and figures (default: outputs/comparison_iid_lr1p5).",
    )
    parser.add_argument("--dataset", type=str, default="mc4")
    parser.add_argument("--partition", type=str, default="iid")
    parser.add_argument("--model_size", type=str, default="15M")
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--inner_steps", type=int, default=30)
    parser.add_argument("--max_steps", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--outer_learning_rate", type=float, default=1.5)
    parser.add_argument("--eval_frequency", type=int, default=10)
    parser.add_argument(
        "--calibration_batch_size",
        type=int,
        default=32,
        help="Calibration batch size for spectral_alloc_decoupled only (default: 32).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config_path = (
        args.config.resolve()
        if args.config.is_absolute()
        else (REPO_ROOT / args.config).resolve()
    )
    summary_dir = (
        args.summary_dir.resolve()
        if args.summary_dir.is_absolute()
        else (REPO_ROOT / args.summary_dir).resolve()
    )
    outputs_root = REPO_ROOT / "outputs"
    outputs_root.mkdir(parents=True, exist_ok=True)

    results: List[MethodBenchmarkResult] = []
    for method in METHODS_ORDER:
        res = run_single_method(
            method=method,
            config_path=config_path,
            outputs_root=outputs_root,
            dataset=args.dataset,
            partition=args.partition,
            model_size=args.model_size,
            num_workers=args.num_workers,
            inner_steps=args.inner_steps,
            max_steps=args.max_steps,
            seed=args.seed,
            outer_lr=args.outer_learning_rate,
            eval_frequency=args.eval_frequency,
            calibration_batch_size=args.calibration_batch_size,
        )
        results.append(res)

    summary_dir.mkdir(parents=True, exist_ok=True)
    fig1_path = summary_dir / "convergence_rounds.png"
    fig2_path = summary_dir / "runtime_comparison.png"
    fig3_path = summary_dir / "convergence_time.png"
    csv_path = summary_dir / "comparison.csv"

    plot_convergence_rounds(results, fig1_path)
    plot_runtime_comparison(results, fig2_path)
    plot_convergence_time(results, fig3_path)
    write_comparison_csv(results, csv_path)

    print("=" * 80)
    print("[Benchmark] All 3 methods completed successfully.")
    print(f"[Benchmark] Summary CSV:      {csv_path}")
    print(f"[Benchmark] Figure 1 (Rounds):  {fig1_path}")
    print(f"[Benchmark] Figure 2 (Runtime): {fig2_path}")
    print(f"[Benchmark] Figure 3 (Time):    {fig3_path}")
    print("=" * 80, flush=True)


if __name__ == "__main__":
    main()



