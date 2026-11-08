"""YAML and CLI configuration loading and normalization for experiments."""

import argparse
import json
import os
from pathlib import Path
import sys
from types import SimpleNamespace
from typing import Any, Dict, List, Optional

import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG_PATH = REPO_ROOT / "configs" / "default.yaml"
EXPERIMENT_CONFIG_PATH = REPO_ROOT / "configs" / "experiment.yaml"


def dict_parser_type(value: str) -> Dict[str, Any]:
    """Helper function to parse a JSON string into a dict for argparse."""
    try:
        value = value.replace("'", '"')
        loaded_dict = json.loads(value)
        return loaded_dict
    except json.JSONDecodeError:
        raise argparse.ArgumentTypeError(f"Invalid JSON format for dictionary: {value}")


def str_to_bool(v: Any) -> bool:
    if isinstance(v, bool):
        return v
    if str(v).lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif str(v).lower() in ("no", "false", "f", "n", "0"):
        return False
    raise argparse.ArgumentTypeError("Boolean value expected.")


def _expand_path(val: Optional[str]) -> Optional[str]:
    if not val or not isinstance(val, str):
        return val
    return os.path.expanduser(os.path.expandvars(val))


def _deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    result = dict(base)
    for k, v in override.items():
        if isinstance(v, dict) and isinstance(result.get(k), dict):
            result[k] = _deep_merge(result[k], v)
        else:
            result[k] = v
    return result


def flatten_yaml_config(
    raw: Dict[str, Any], include_none: bool = False
) -> Dict[str, Any]:
    """
    Convert structured YAML configuration sections into the flat namespace
    expected by DistributedModelTrainer while also accepting top-level flat keys.
    """
    flat: Dict[str, Any] = {}

    section_mappings = {
        "experiment": {
            "run_name": "run_name",
            "run_id": "run_id",
            "seed": "seed",
            "project": "project",
            "log_wandb": "log_wandb",
            "debug": "debug",
            "trace": "trace",
        },
        "dataset": {
            "key": "dataset_key",
            "name": "dataset_name",
            "revision": "dataset_revision",
            "shards_path": "shards_path",
            "shard_token_size": "shard_token_size",
            "token_budget": "token_budget",
            "sequence_length": "sequence_length",
            "languages": "languages",
            "pin_data_to_gpu": "pin_data_to_gpu",
        },
        "partitioning": {
            "strategy": "partition_strategy",
            "partition_file": "partition_file",
            "save_partitions": "save_partitions",
            "eval_partition_file": "eval_partition_file",
        },
        "model": {
            "architecture": "model_architecture",
            "size": "model_size",
            "hparams_file": "hparams_file",
            "tokenizer_name": "tokenizer_name",
            "tokenizer_revision": "tokenizer_revision",
            "hparams": "model_hparams",
            "use_compile": "use_compile",
            "torchtitan_init": "torchtitan_init",
        },
        "workers": {
            "num_workers": "num_workers",
            "virtual_workers": "virtual_workers",
        },
        "training": {
            "inner_outer_strategy": "inner_outer_strategy",
            "inner_steps": "inner_steps",
            "batch_size": "batch_size",
            "micro_batch_size": "micro_batch_size",
            "max_steps": "max_steps",
        },
        "inner_optimizer": {
            "type": "inner_optimizer",
            "learning_rate": "inner_learning_rate",
            "beta2": "inner_beta2",
            "weight_decay": "weight_decay",
            "warmup_steps": "warmup_steps",
        },
        "method": {
            "name": "method",
            "strategy": "strategy",
            "outer_optimizer": "outer_optimizer",
            "learning_rate": "outer_learning_rate",
            "momentum": "outer_momentum",
            "nesterov": "outer_nesterov",
            "weight_decay": "outer_weight_decay",
            "use_outer_scheduler": "use_outer_scheduler",
            "outer_warmup_steps": "outer_warmup_steps",
            "isoc_orthogonalize_object": "isoc_orthogonalize_object",
            "isoc_exclude_keys": "isoc_exclude_keys",
            "isocts_common_space_fraction": "isocts_common_space_fraction",
            "isocts_exclude_keys": "isocts_exclude_keys",
            "tsv_singular_ratio": "tsv_singular_ratio",
            "tsv_decorrelate": "tsv_decorrelate",
            "sparsity": "sparsity",
            "prune_method": "prune_method",
            "prune_dimension": "prune_dimension",
            "rescale_pruned": "rescale_pruned",
            "ties_disjoint": "ties_disjoint",
            "calibration_batch_size": "calibration_batch_size",
            "calibration_micro_batch_size": "calibration_micro_batch_size",
            "calibration_token_chunk_size": "calibration_token_chunk_size",
            "eval_actual_calib_loss_change": "eval_actual_calib_loss_change",
            "kkt_tol": "kkt_tol",
            "feas_tol": "feas_tol",
        },
        "outer_optimizer": {
            "type": "outer_optimizer",
            "learning_rate": "outer_learning_rate",
            "momentum": "outer_momentum",
            "nesterov": "outer_nesterov",
            "weight_decay": "outer_weight_decay",
            "use_outer_scheduler": "use_outer_scheduler",
            "outer_warmup_steps": "outer_warmup_steps",
        },
        "evaluation": {
            "eval_frequency": "eval_frequency",
            "val_token_budget": "val_token_budget",
            "per_worker_eval": "per_worker_eval",
        },
        "checkpointing": {
            "save_path": "save_path",
            "checkpoint_frequency": "checkpoint_frequency",
            "load_checkpoint": "load_checkpoint",
            "resume_from_checkpoint": "resume_from_checkpoint",
            "save_final_diagnostics": "save_final_diagnostics",
        },
    }

    for section, mapping in section_mappings.items():
        sec_dict = raw.get(section)
        if isinstance(sec_dict, dict):
            for yaml_key, attr_name in mapping.items():
                if yaml_key in sec_dict and (
                    include_none or sec_dict[yaml_key] is not None
                ):
                    flat[attr_name] = sec_dict[yaml_key]

    for k, v in raw.items():
        if k not in section_mappings and not isinstance(v, dict):
            if include_none or v is not None:
                flat[k] = v

    return flat


def build_arg_parser() -> argparse.ArgumentParser:
    """Build CLI argument parser compatible with both YAML configs and official IsoLoCo flags."""
    parser = argparse.ArgumentParser(
        description="Distributed training for DiLoCo, IsoLoCo, Iso-CTS, and merging methods."
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to YAML experiment configuration file (defaults to configs/experiment.yaml if present).",
    )
    parser.add_argument("--run_name", type=str, default="auto")
    parser.add_argument("--run_id", type=str, default=None)
    parser.add_argument("--project", type=str, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--method",
        type=str,
        default=None,
        help="Aggregation method selector (diloco, isoloco, signal_iso, soft_signal_iso, spectral_alloc, spectral_alloc_decoupled, isocloco, isoctsloco, tsvloco, mergeloco, adam_baseline)",
    )
    parser.add_argument(
        "--strategy",
        type=str,
        default=None,
        help="Preset strategy (diloco_baseline, isoloco, signal_iso, soft_signal_iso, spectral_alloc, spectral_alloc_decoupled, isocloco, isoctsloco, tsvloco, mergeloco, diloco_sgd_noMom, adam_baseline)",
    )
    parser.add_argument(
        "--inner_outer_strategy",
        type=str,
        default="diloco",
        choices=["simple_accum", "diloco"],
    )
    parser.add_argument(
        "--inner_optimizer",
        type=str,
        default=None,
        choices=["adamw", "sgd", None],
    )
    parser.add_argument(
        "--outer_optimizer",
        type=str,
        default="adamw",
        choices=[
            "adamw",
            "nesterov",
            "sgd",
            "mergeloco",
            "tsvloco",
            "isocloco",
            "isoctsloco",
            "signal_iso",
            "soft_signal_iso",
            "spectral_alloc",
            "spectral_alloc_decoupled",
        ],
    )
    parser.add_argument("--num_workers", type=int, default=None)
    parser.add_argument("--virtual_workers", type=int, default=1)
    parser.add_argument("--inner_learning_rate", type=float, default=6e-4)
    parser.add_argument("--inner_steps", "--local_steps", dest="inner_steps", type=int, default=15)
    parser.add_argument("--inner_beta2", type=float, default=0.95)
    parser.add_argument("--outer_learning_rate", type=float, default=0.7)
    parser.add_argument("--outer_momentum", type=float, default=0.0)
    parser.add_argument(
        "--outer_nesterov", type=str_to_bool, nargs="?", const=True, default=False
    )
    parser.add_argument("--outer_weight_decay", type=float, default=0.0)
    parser.add_argument("--use_outer_scheduler", action="store_true", default=False)
    parser.add_argument("--outer_warmup_steps", type=float, default=None)
    parser.add_argument("--weight_decay", type=float, default=0.1)
    parser.add_argument("--warmup_steps", type=float, default=250)
    parser.add_argument("--max_steps", "--outer_rounds", dest="max_steps", type=int, default=-1)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--micro_batch_size", type=int, default=16)
    parser.add_argument("--sequence_length", type=int, default=2048)

    parser.add_argument(
        "--dataset",
        dest="dataset_key",
        type=str,
        default=None,
        help="Dataset preset selector: 'dclm' or 'mc4'",
    )
    parser.add_argument(
        "--dataset_name", type=str, default="mlfoundations/dclm-baseline-1.0-parquet"
    )
    parser.add_argument("--dataset_revision", type=str, default="main")
    parser.add_argument("--shards_path", type=str, default="/lustre/hdd/LAS/jannesar-lab/aaasif/datasets/dclm_100m")
    parser.add_argument("--token_budget", type=int, default=100_000_000)
    parser.add_argument("--shard_token_size", type=int, default=100_000_000)
    parser.add_argument("--pin_data_to_gpu", action="store_true", default=False)

    parser.add_argument(
        "--partition",
        "--partition_strategy",
        dest="partition_strategy",
        type=str,
        default="iid",
        help="Partition strategy: 'iid' or 'noniid_language'",
    )
    parser.add_argument("--partition_file", type=str, default=None)
    parser.add_argument(
        "--save_partitions", type=str_to_bool, nargs="?", const=True, default=True
    )
    parser.add_argument("--eval_partition_file", type=str, default=None)

    parser.add_argument("--eval_frequency", type=int, default=2)
    parser.add_argument("--val_token_budget", type=int, default=2_000_000)
    parser.add_argument(
        "--per_worker_eval", type=str_to_bool, nargs="?", const=True, default=False
    )

    parser.add_argument("--save_path", type=str, default="auto")
    parser.add_argument("--checkpoint_frequency", type=int, default=6)
    parser.add_argument("--load_checkpoint", type=str, default=None)
    parser.add_argument("--resume_from_checkpoint", action="store_true", default=False)
    parser.add_argument("--save_final_diagnostics", action="store_true", default=False)

    parser.add_argument("--model_architecture", type=str, default="llama")
    parser.add_argument("--model_size", "--model", dest="model_size", type=str, default="178M")
    parser.add_argument(
        "--hparams_file", type=str, default="models/configs/178M_model_hparams.json"
    )
    parser.add_argument("--tokenizer_name", type=str, default=None)
    parser.add_argument("--tokenizer_revision", type=str, default="main")
    parser.add_argument(
        "--model_hparams",
        type=dict_parser_type,
        default={},
    )
    parser.add_argument("--torchtitan_init", action="store_true", default=False)
    parser.add_argument(
        "--use_compile", type=str_to_bool, nargs="?", const=True, default=False
    )

    parser.add_argument("--debug", action="store_true", default=False)
    parser.add_argument("--trace", action="store_true", default=False)
    parser.add_argument(
        "--log_wandb", type=str_to_bool, nargs="?", const=True, default=True
    )

    parser.add_argument("--sparsity", type=float, default=0.0)
    parser.add_argument(
        "--prune_method",
        type=str,
        default="magnitude",
        choices=["magnitude", "random"],
    )
    parser.add_argument(
        "--prune_dimension",
        type=str,
        default="all",
        choices=["all", "row", "column"],
    )
    parser.add_argument(
        "--rescale_pruned", type=str_to_bool, nargs="?", const=True, default=False
    )
    parser.add_argument(
        "--ties_disjoint", type=str_to_bool, nargs="?", const=True, default=False
    )

    parser.add_argument("--tsv_singular_ratio", type=float, default=None)
    parser.add_argument(
        "--tsv_decorrelate", type=str_to_bool, nargs="?", const=True, default=True
    )

    parser.add_argument(
        "--isoc_orthogonalize_object",
        type=str,
        default="gradient",
        choices=["gradient", "update"],
    )
    parser.add_argument(
        "--isocts_common_space_fraction",
        type=float,
        default=0.8,
    )

    parser.add_argument("--calibration_batch_size", type=int, default=8)
    parser.add_argument("--calibration_micro_batch_size", type=int, default=2)
    parser.add_argument("--calibration_token_chunk_size", type=int, default=128)
    parser.add_argument(
        "--eval_actual_calib_loss_change",
        type=str_to_bool,
        nargs="?",
        const=True,
        default=True,
    )
    parser.add_argument("--kkt_tol", type=float, default=1e-5)
    parser.add_argument("--feas_tol", type=float, default=1e-7)
    return parser


CLI_FLAG_ALIASES: Dict[str, str] = {
    "model": "model_size",
    "dataset": "dataset_key",
    "partition": "partition_strategy",
    "local_steps": "inner_steps",
    "outer_rounds": "max_steps",
    "rounds": "max_steps",
}


def load_experiment_config(argv: Optional[List[str]] = None) -> SimpleNamespace:
    """
    Load configuration from configs/default.yaml, resolve dataset/model/method presets,
    merge configs/experiment.yaml (or --config <file>), and apply explicit CLI overrides.
    """
    raw_argv = list(sys.argv[1:] if argv is None else argv)
    parser = build_arg_parser()
    parsed_cli = parser.parse_args(raw_argv)

    explicit_cli_keys = set()
    for token in raw_argv:
        if token.startswith("--"):
            raw_key = token[2:].split("=")[0].replace("-", "_")
            explicit_cli_keys.add(CLI_FLAG_ALIASES.get(raw_key, raw_key))

    default_data: Dict[str, Any] = {}
    if DEFAULT_CONFIG_PATH.is_file():
        with open(DEFAULT_CONFIG_PATH, "r", encoding="utf-8") as f:
            default_data = yaml.safe_load(f) or {}

    user_data: Dict[str, Any] = {}
    cfg_to_load: Optional[Path] = None
    if parsed_cli.config:
        cfg_to_load = Path(_expand_path(parsed_cli.config))
        if not cfg_to_load.is_file():
            cfg_to_load = REPO_ROOT / parsed_cli.config
        if not cfg_to_load.is_file():
            raise FileNotFoundError(
                f"Configuration YAML file not found: {parsed_cli.config}"
            )
    elif EXPERIMENT_CONFIG_PATH.is_file():
        cfg_to_load = EXPERIMENT_CONFIG_PATH

    if cfg_to_load is not None:
        with open(cfg_to_load, "r", encoding="utf-8") as f:
            user_data = yaml.safe_load(f) or {}

    flat_default = flatten_yaml_config(default_data, include_none=False)
    flat_user = flatten_yaml_config(user_data, include_none=True)

    presets = default_data.get("presets", {}) or {}
    ds_presets = presets.get("datasets", {}) or {}
    model_presets = presets.get("models", {}) or {}

    # 1. Determine active dataset_key
    ds_key = (
        parsed_cli.dataset_key
        if "dataset_key" in explicit_cli_keys
        else (flat_user.get("dataset_key") or flat_default.get("dataset_key"))
    )
    if not ds_key:
        ds_name_probe = str(
            flat_user.get("dataset_name") or flat_default.get("dataset_name") or ""
        ).lower()
        ds_key = "mc4" if "c4" in ds_name_probe else "dclm"
    ds_key = str(ds_key).lower()
    ds_preset = dict(ds_presets.get(ds_key, {}))

    # 2. Determine active model_size
    raw_model_size = (
        parsed_cli.model_size
        if "model_size" in explicit_cli_keys
        else (flat_user.get("model_size") or flat_default.get("model_size") or "178M")
    )
    model_size_key = str(raw_model_size).upper()
    model_preset_raw = dict(
        model_presets.get(model_size_key)
        or model_presets.get(str(raw_model_size))
        or {}
    )
    model_methods_map = dict(model_preset_raw.pop("methods", {}) or {})

    # 3. Determine active method / strategy
    if "method" in explicit_cli_keys and parsed_cli.method:
        raw_method = parsed_cli.method
    elif "strategy" in explicit_cli_keys and parsed_cli.strategy:
        raw_method = parsed_cli.strategy
    else:
        raw_method = (
            flat_user.get("method")
            or flat_user.get("strategy")
            or flat_default.get("method")
            or flat_default.get("strategy")
            or "diloco"
        )
    method_aliases = {
        "diloco": "diloco",
        "diloco_baseline": "diloco",
        "isoloco": "isoloco",
        "signal_iso": "signal_iso",
        "signal_isoloco": "signal_iso",
        "signalisoloco": "signal_iso",
        "soft_signal_iso": "soft_signal_iso",
        "soft_signal_isoloco": "soft_signal_iso",
        "softsignalisoloco": "soft_signal_iso",
        "spectral_alloc": "spectral_alloc",
        "spectral_alloc_loco": "spectral_alloc",
        "spectralalloc": "spectral_alloc",
        "spectral_alloc_decoupled": "spectral_alloc_decoupled",
        "spectral_alloc_decoupled_loco": "spectral_alloc_decoupled",
        "spectralallocdecoupled": "spectral_alloc_decoupled",
        "isoc": "isocloco",
        "isocloco": "isocloco",
        "isocts": "isoctsloco",
        "iso_cts": "isoctsloco",
        "isoctsloco": "isoctsloco",
        "tsv": "tsvloco",
        "tsvloco": "tsvloco",
        "mergeloco": "mergeloco",
        "ties": "ties",
        "dare": "dare",
        "diloco_sgd": "diloco_sgd",
        "diloco_sgd_nomom": "diloco_sgd",
        "diloco_sgd_noMom": "diloco_sgd",
        "adam": "adam_baseline",
        "adamw": "adam_baseline",
        "adam_baseline": "adam_baseline",
    }
    method_lookup = method_aliases.get(
        str(raw_method).lower(), str(raw_method).lower()
    )
    method_preset = dict(model_methods_map.get(method_lookup, {}))
    if not method_preset and raw_method:
        method_preset = {"method": method_lookup, "strategy": str(raw_method)}

    # Filter out user_data fields that belong to a preset selector overridden on the CLI
    filtered_user = dict(flat_user)
    if "dataset_key" in explicit_cli_keys:
        for k in ds_preset:
            filtered_user.pop(k, None)
    if "model_size" in explicit_cli_keys:
        for k in list(model_preset_raw.keys()) + list(method_preset.keys()):
            filtered_user.pop(k, None)
    if "method" in explicit_cli_keys or "strategy" in explicit_cli_keys:
        filtered_user.pop("method", None)
        filtered_user.pop("strategy", None)
        for k in method_preset:
            filtered_user.pop(k, None)

    flat_yaml = {
        **flat_default,
        **ds_preset,
        **model_preset_raw,
        **method_preset,
        **filtered_user,
    }
    if "method" in explicit_cli_keys or "strategy" in explicit_cli_keys:
        flat_yaml.update(method_preset)
    if "dataset_key" in explicit_cli_keys:
        flat_yaml.update(ds_preset)

    final_dict = vars(parsed_cli).copy()
    for k, v in flat_yaml.items():
        if k not in explicit_cli_keys:
            if v is not None or k in filtered_user:
                final_dict[k] = v

    if "method" in explicit_cli_keys or "strategy" in explicit_cli_keys:
        final_dict["method"] = method_preset.get("method", method_lookup)
        if "strategy" not in explicit_cli_keys:
            final_dict["strategy"] = method_preset.get("strategy", method_lookup)

    # If --virtual_workers was explicitly passed on CLI without --num_workers, let virtual_workers govern
    if (
        "virtual_workers" in explicit_cli_keys
        and "num_workers" not in explicit_cli_keys
    ):
        final_dict["num_workers"] = None

    # If --outer_optimizer was explicitly passed on CLI without --strategy/--method, do not let YAML strategy overwrite it
    if (
        "outer_optimizer" in explicit_cli_keys
        and "strategy" not in explicit_cli_keys
        and "method" not in explicit_cli_keys
    ):
        final_dict["strategy"] = None

    # Precedence for model_size vs hparams_file
    user_yaml_has_hparams_file = (
        isinstance(user_data.get("model"), dict)
        and user_data["model"].get("hparams_file") is not None
    ) or ("hparams_file" in user_data and user_data["hparams_file"] is not None)
    if "hparams_file" not in explicit_cli_keys:
        if "model_size" in explicit_cli_keys or not user_yaml_has_hparams_file:
            if final_dict.get("model_size") and not model_preset_raw.get("hparams_file"):
                final_dict["hparams_file"] = None

    # Keep token_budget synchronized when fixed-round parameters are overridden on the CLI,
    # without shrinking an existing dataset token_budget during short --max_steps diagnostic
    # runs so frozen training partition manifests (e.g. 12.288M-token mC4 partitions) are preserved.
    round_keys_overridden = bool(
        explicit_cli_keys
        & {
            "max_steps",
            "inner_steps",
            "batch_size",
            "sequence_length",
            "virtual_workers",
            "num_workers",
        }
    )
    if (
        round_keys_overridden
        and "token_budget" not in explicit_cli_keys
        and int(final_dict.get("max_steps", -1)) > 0
    ):
        eff_w = int(
            final_dict.get("num_workers") or final_dict.get("virtual_workers") or 1
        )
        required_tokens = (
            int(final_dict["max_steps"])
            * eff_w
            * int(final_dict["inner_steps"])
            * int(final_dict["batch_size"])
            * int(final_dict["sequence_length"])
        )
        current_budget = int(final_dict.get("token_budget") or 0)
        if "max_steps" in explicit_cli_keys and current_budget >= required_tokens:
            pass
        else:
            final_dict["token_budget"] = required_tokens

    yaml_has_momentum = (
        "outer_momentum" in flat_yaml and flat_yaml["outer_momentum"] is not None
    )
    final_dict["_outer_momentum_explicit"] = (
        "outer_momentum" in explicit_cli_keys or yaml_has_momentum
    )

    # Generate structured run_name and output directory: outputs/<dataset>_<model>_<method>_<partition>_<run_id>/
    dataset_tag = str(
        final_dict.get("dataset_key")
        or ("mc4" if "c4" in str(final_dict.get("dataset_name", "")).lower() else "dclm")
    ).lower()
    model_tag = str(final_dict.get("model_size") or "15m").lower()
    raw_strat = str(
        final_dict.get("method")
        or final_dict.get("strategy")
        or final_dict.get("outer_optimizer")
        or "diloco"
    ).lower()
    method_tag = "diloco" if raw_strat in ("diloco", "diloco_baseline") else raw_strat
    partition_tag = str(final_dict.get("partition_strategy") or "iid").lower()
    eff_workers = int(
        final_dict.get("num_workers") or final_dict.get("virtual_workers") or 4
    )
    tokens_per_round = max(
        1,
        eff_workers
        * int(final_dict["inner_steps"])
        * int(final_dict["batch_size"])
        * int(final_dict["sequence_length"]),
    )
    eff_rounds = (
        int(final_dict["max_steps"])
        if int(final_dict.get("max_steps", -1)) > 0
        else max(1, int(final_dict["token_budget"]) // tokens_per_round)
    )
    run_id = final_dict.get("run_id") or (
        f"w{eff_workers}_h{final_dict['inner_steps']}_r{eff_rounds}_s{final_dict['seed']}"
    )
    structured_name = f"{dataset_tag}_{model_tag}_{method_tag}_{partition_tag}_{run_id}"

    if final_dict.get("run_name") in (None, "", "auto", "run"):
        final_dict["run_name"] = structured_name
    if final_dict.get("save_path") in (None, "", "auto", "./outputs"):
        final_dict["save_path"] = str(REPO_ROOT / "outputs" / structured_name)

    for path_key in (
        "shards_path",
        "save_path",
        "load_checkpoint",
        "hparams_file",
        "partition_file",
        "eval_partition_file",
    ):
        if final_dict.get(path_key):
            final_dict[path_key] = _expand_path(final_dict[path_key])

    # Validate core numerical settings early so invalid settings fail clearly
    if final_dict["batch_size"] <= 0 or final_dict["micro_batch_size"] <= 0:
        raise ValueError("batch_size and micro_batch_size must be positive integers.")
    if final_dict["batch_size"] < final_dict["micro_batch_size"]:
        raise ValueError(
            f"batch_size ({final_dict['batch_size']}) must be >= micro_batch_size ({final_dict['micro_batch_size']})."
        )
    if final_dict["inner_steps"] <= 0:
        raise ValueError(f"inner_steps must be > 0, got {final_dict['inner_steps']}.")
    if final_dict["token_budget"] <= 0 or final_dict["sequence_length"] <= 0:
        raise ValueError("token_budget and sequence_length must be positive integers.")

    return SimpleNamespace(**final_dict)

