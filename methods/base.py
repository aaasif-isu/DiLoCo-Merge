"""Registry and factory for outer aggregation methods and strategy presets."""

from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional

import torch
import torch.distributed as dist
import torch.nn as nn

from .diloco import build_adamw_outer, build_diloco_nesterov, build_diloco_sgd
from .isocloco import ISOCLoCo
from .isoctsloco import ISOCTSLoCo
from .mergeloco import MergeLoCo
from .signal_iso import SignalIsoLoCo
from .soft_signal_iso import SoftSignalIsoLoCo
from .spectral_alloc import SpectralAllocDecoupledLoCo, SpectralAllocLoCo
from .tsvloco import TSVLoCo


@dataclass
class MethodSpec:
    name: str
    builder: Callable[..., torch.optim.Optimizer]
    handles_own_communication: bool = False


METHOD_REGISTRY: Dict[str, MethodSpec] = {}


def register_method(name: str, handles_own_communication: bool = False):
    """Decorator to register an outer aggregation optimizer builder."""

    def decorator(fn: Callable[..., torch.optim.Optimizer]):
        METHOD_REGISTRY[name.lower()] = MethodSpec(
            name=name.lower(),
            builder=fn,
            handles_own_communication=handles_own_communication,
        )
        return fn

    return decorator


def method_handles_own_communication(outer_optimizer_name: str) -> bool:
    """Return True if the outer optimizer gathers worker pseudo-gradients internally."""
    key = outer_optimizer_name.lower()
    if key in METHOD_REGISTRY:
        return METHOD_REGISTRY[key].handles_own_communication
    return False


@register_method("adamw", handles_own_communication=False)
def _build_adamw(
    virtual_models: List[nn.Module],
    config: Any,
    virtual_workers: int = 1,
    process_group: Optional[dist.ProcessGroup] = None,
) -> torch.optim.Optimizer:
    return build_adamw_outer(
        virtual_models[0].parameters(),
        lr=config.outer_learning_rate,
        weight_decay=config.weight_decay,
        betas=(0.9, config.inner_beta2),
    )


@register_method("nesterov", handles_own_communication=False)
def _build_nesterov(
    virtual_models: List[nn.Module],
    config: Any,
    virtual_workers: int = 1,
    process_group: Optional[dist.ProcessGroup] = None,
) -> torch.optim.Optimizer:
    return build_diloco_nesterov(
        virtual_models[0].parameters(),
        lr=config.outer_learning_rate,
        momentum=config.outer_momentum,
    )


@register_method("sgd", handles_own_communication=False)
def _build_sgd(
    virtual_models: List[nn.Module],
    config: Any,
    virtual_workers: int = 1,
    process_group: Optional[dist.ProcessGroup] = None,
) -> torch.optim.Optimizer:
    return build_diloco_sgd(
        virtual_models[0].parameters(),
        lr=config.outer_learning_rate,
    )


@register_method("mergeloco", handles_own_communication=True)
def _build_mergeloco(
    virtual_models: List[nn.Module],
    config: Any,
    virtual_workers: int = 1,
    process_group: Optional[dist.ProcessGroup] = None,
) -> torch.optim.Optimizer:
    return MergeLoCo(
        [vm.parameters() for vm in virtual_models],
        lr=config.outer_learning_rate,
        momentum=config.outer_momentum,
        nesterov=config.outer_nesterov,
        weight_decay=config.outer_weight_decay,
        sparsity=config.sparsity,
        prune_method=config.prune_method,
        prune_dimension=config.prune_dimension,
        rescale_pruned=config.rescale_pruned,
        ties_disjoint=config.ties_disjoint,
        virtual_workers=virtual_workers,
        process_group=process_group,
    )


@register_method("tsvloco", handles_own_communication=True)
def _build_tsvloco(
    virtual_models: List[nn.Module],
    config: Any,
    virtual_workers: int = 1,
    process_group: Optional[dist.ProcessGroup] = None,
) -> torch.optim.Optimizer:
    return TSVLoCo(
        [vm.parameters() for vm in virtual_models],
        lr=config.outer_learning_rate,
        singular_ratio=config.tsv_singular_ratio,
        decorrelate=config.tsv_decorrelate,
        virtual_workers=virtual_workers,
        process_group=process_group,
    )


@register_method("isocloco", handles_own_communication=True)
def _build_isocloco(
    virtual_models: List[nn.Module],
    config: Any,
    virtual_workers: int = 1,
    process_group: Optional[dist.ProcessGroup] = None,
) -> torch.optim.Optimizer:
    exclude_keys = getattr(config, "isoc_exclude_keys", None) or [
        "embed_tokens",
        "wpe",
        "lm_head",
    ]
    return ISOCLoCo(
        [vm.parameters() for vm in virtual_models],
        lr=config.outer_learning_rate,
        momentum=config.outer_momentum,
        nesterov=config.outer_nesterov,
        orthogonalize_object=config.isoc_orthogonalize_object,
        exclude_keys=exclude_keys,
        virtual_workers=virtual_workers,
        process_group=process_group,
    )


@register_method("isoctsloco", handles_own_communication=True)
def _build_isoctsloco(
    virtual_models: List[nn.Module],
    config: Any,
    virtual_workers: int = 1,
    process_group: Optional[dist.ProcessGroup] = None,
) -> torch.optim.Optimizer:
    exclude_keys = getattr(config, "isocts_exclude_keys", None) or [
        "embed_tokens",
        "wpe",
        "lm_head",
    ]
    return ISOCTSLoCo(
        [vm.parameters() for vm in virtual_models],
        lr=config.outer_learning_rate,
        momentum=config.outer_momentum,
        nesterov=config.outer_nesterov,
        common_space_fraction=config.isocts_common_space_fraction,
        exclude_keys=exclude_keys,
        virtual_workers=virtual_workers,
        process_group=process_group,
    )


@register_method("signal_iso", handles_own_communication=True)
def _build_signal_iso(
    virtual_models: List[nn.Module],
    config: Any,
    virtual_workers: int = 1,
    process_group: Optional[dist.ProcessGroup] = None,
) -> torch.optim.Optimizer:
    exclude_keys = getattr(config, "isoc_exclude_keys", None) or [
        "embed_tokens",
        "wpe",
        "lm_head",
    ]
    return SignalIsoLoCo(
        [vm.parameters() for vm in virtual_models],
        lr=config.outer_learning_rate,
        momentum=config.outer_momentum,
        nesterov=config.outer_nesterov,
        orthogonalize_object=getattr(config, "isoc_orthogonalize_object", "gradient"),
        exclude_keys=exclude_keys,
        virtual_workers=virtual_workers,
        process_group=process_group,
    )


@register_method("soft_signal_iso", handles_own_communication=True)
def _build_soft_signal_iso(
    virtual_models: List[nn.Module],
    config: Any,
    virtual_workers: int = 1,
    process_group: Optional[dist.ProcessGroup] = None,
) -> torch.optim.Optimizer:
    exclude_keys = getattr(config, "isoc_exclude_keys", None) or [
        "embed_tokens",
        "wpe",
        "lm_head",
    ]
    return SoftSignalIsoLoCo(
        [vm.parameters() for vm in virtual_models],
        lr=config.outer_learning_rate,
        momentum=config.outer_momentum,
        nesterov=config.outer_nesterov,
        orthogonalize_object=getattr(config, "isoc_orthogonalize_object", "gradient"),
        exclude_keys=exclude_keys,
        virtual_workers=virtual_workers,
        process_group=process_group,
    )


@register_method("spectral_alloc", handles_own_communication=True)
def _build_spectral_alloc(
    virtual_models: List[nn.Module],
    config: Any,
    virtual_workers: int = 1,
    process_group: Optional[dist.ProcessGroup] = None,
) -> torch.optim.Optimizer:
    exclude_keys = getattr(config, "isoc_exclude_keys", None) or [
        "embed_tokens",
        "wpe",
        "lm_head",
    ]
    return SpectralAllocLoCo(
        [vm.parameters() for vm in virtual_models],
        lr=config.outer_learning_rate,
        momentum=config.outer_momentum,
        nesterov=config.outer_nesterov,
        orthogonalize_object=getattr(config, "isoc_orthogonalize_object", "gradient"),
        exclude_keys=exclude_keys,
        virtual_workers=virtual_workers,
        process_group=process_group,
        model=virtual_models[0] if virtual_models else None,
        shards_path=getattr(config, "shards_path", None),
        sequence_length=int(getattr(config, "sequence_length", 128)),
        calibration_batch_size=int(getattr(config, "calibration_batch_size", 8)),
        calibration_micro_batch_size=int(
            getattr(config, "calibration_micro_batch_size", 2)
        ),
        calibration_token_chunk_size=int(
            getattr(config, "calibration_token_chunk_size", 128)
        ),
        pad_token_id=getattr(config, "pad_token_id", None),
        seed=int(getattr(config, "seed", 42)),
        languages=getattr(config, "languages", None),
        eval_actual_calib_loss_change=bool(
            getattr(config, "eval_actual_calib_loss_change", True)
        ),
        kkt_tol=float(getattr(config, "kkt_tol", 1e-5)),
        feas_tol=float(getattr(config, "feas_tol", 1e-7)),
    )


@register_method("spectral_alloc_decoupled", handles_own_communication=True)
def _build_spectral_alloc_decoupled(
    virtual_models: List[nn.Module],
    config: Any,
    virtual_workers: int = 1,
    process_group: Optional[dist.ProcessGroup] = None,
) -> torch.optim.Optimizer:
    exclude_keys = getattr(config, "isoc_exclude_keys", None) or [
        "embed_tokens",
        "wpe",
        "lm_head",
    ]
    return SpectralAllocDecoupledLoCo(
        [vm.parameters() for vm in virtual_models],
        lr=config.outer_learning_rate,
        momentum=config.outer_momentum,
        nesterov=config.outer_nesterov,
        orthogonalize_object=getattr(config, "isoc_orthogonalize_object", "gradient"),
        exclude_keys=exclude_keys,
        virtual_workers=virtual_workers,
        process_group=process_group,
        model=virtual_models[0] if virtual_models else None,
        shards_path=getattr(config, "shards_path", None),
        sequence_length=int(getattr(config, "sequence_length", 128)),
        calibration_batch_size=int(getattr(config, "calibration_batch_size", 8)),
        calibration_micro_batch_size=int(
            getattr(config, "calibration_micro_batch_size", 2)
        ),
        calibration_token_chunk_size=int(
            getattr(config, "calibration_token_chunk_size", 128)
        ),
        pad_token_id=getattr(config, "pad_token_id", None),
        seed=int(getattr(config, "seed", 42)),
        languages=getattr(config, "languages", None),
        eval_actual_calib_loss_change=bool(
            getattr(config, "eval_actual_calib_loss_change", True)
        ),
        kkt_tol=float(getattr(config, "kkt_tol", 1e-5)),
        feas_tol=float(getattr(config, "feas_tol", 1e-7)),
    )


STRATEGY_ALIASES: Dict[str, str] = {
    "diloco": "diloco_baseline",
    "diloco_baseline": "diloco_baseline",
    "diloco_sgd": "diloco_sgd_noMom",
    "diloco_sgd_nomom": "diloco_sgd_noMom",
    "diloco_sgd_noMom": "diloco_sgd_noMom",
    "adam": "adam_baseline",
    "adamw": "adam_baseline",
    "adam_baseline": "adam_baseline",
    "isoc": "isocloco",
    "isocloco": "isocloco",
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
    "isocts": "isoctsloco",
    "iso_cts": "isoctsloco",
    "isoctsloco": "isoctsloco",
    "tsv": "tsvloco",
    "tsvloco": "tsvloco",
    "mergeloco": "mergeloco",
    "ties": "mergeloco",
    "dare": "mergeloco",
}


def apply_strategy_preset(config: Any) -> None:
    """
    Apply strategy preset rules to config.
    Preserves the exact strategy preset behavior from the official IsoLoCo train.py.
    """
    if getattr(config, "strategy", None) is None:
        return

    canonical = STRATEGY_ALIASES.get(config.strategy, config.strategy)

    if canonical == "adam_baseline":
        config.inner_outer_strategy = "simple_accum"
        config.inner_optimizer = None
        config.outer_optimizer = "adamw"
        config.use_outer_scheduler = True
        config.virtual_workers = 1
    elif canonical == "diloco_baseline":
        config.inner_outer_strategy = "diloco"
        config.inner_optimizer = "adamw"
        config.outer_optimizer = "nesterov"
        config.use_outer_scheduler = False
    elif canonical == "diloco_sgd_noMom":
        config.inner_outer_strategy = "diloco"
        config.inner_optimizer = "adamw"
        config.outer_optimizer = "sgd"
        config.use_outer_scheduler = False
    elif canonical == "mergeloco":
        config.inner_outer_strategy = "diloco"
        config.inner_optimizer = "adamw"
        config.outer_optimizer = "mergeloco"
        config.use_outer_scheduler = False
    elif canonical == "tsvloco":
        config.inner_outer_strategy = "diloco"
        config.inner_optimizer = "adamw"
        config.outer_optimizer = "tsvloco"
        config.use_outer_scheduler = False
    elif canonical == "isocloco":
        config.inner_outer_strategy = "diloco"
        config.inner_optimizer = "adamw"
        config.outer_optimizer = "isocloco"
        config.use_outer_scheduler = False
    elif canonical == "isoloco":
        config.inner_outer_strategy = "diloco"
        config.inner_optimizer = "adamw"
        config.outer_optimizer = "isocloco"
        config.use_outer_scheduler = False
        if (
            not getattr(config, "_outer_momentum_explicit", False)
            and config.outer_momentum == 0.0
        ):
            config.outer_momentum = 0.8
        config.outer_nesterov = True
    elif canonical == "signal_iso":
        config.inner_outer_strategy = "diloco"
        config.inner_optimizer = "adamw"
        config.outer_optimizer = "signal_iso"
        config.use_outer_scheduler = False
        if (
            not getattr(config, "_outer_momentum_explicit", False)
            and config.outer_momentum == 0.0
        ):
            config.outer_momentum = 0.8
        if config.outer_momentum > 0:
            config.outer_nesterov = True
    elif canonical == "soft_signal_iso":
        config.inner_outer_strategy = "diloco"
        config.inner_optimizer = "adamw"
        config.outer_optimizer = "soft_signal_iso"
        config.use_outer_scheduler = False
        if (
            not getattr(config, "_outer_momentum_explicit", False)
            and config.outer_momentum == 0.0
        ):
            config.outer_momentum = 0.8
        if config.outer_momentum > 0:
            config.outer_nesterov = True
    elif canonical == "spectral_alloc":
        config.inner_outer_strategy = "diloco"
        config.inner_optimizer = "adamw"
        config.outer_optimizer = "spectral_alloc"
        config.use_outer_scheduler = False
        if (
            not getattr(config, "_outer_momentum_explicit", False)
            and config.outer_momentum == 0.0
        ):
            config.outer_momentum = 0.8
        if config.outer_momentum > 0:
            config.outer_nesterov = True
    elif canonical == "spectral_alloc_decoupled":
        config.inner_outer_strategy = "diloco"
        config.inner_optimizer = "adamw"
        config.outer_optimizer = "spectral_alloc_decoupled"
        config.use_outer_scheduler = False
        if (
            not getattr(config, "_outer_momentum_explicit", False)
            and config.outer_momentum == 0.0
        ):
            config.outer_momentum = 0.8
        if config.outer_momentum > 0:
            config.outer_nesterov = True
    elif canonical == "isoctsloco":
        config.inner_outer_strategy = "diloco"
        config.inner_optimizer = "adamw"
        config.outer_optimizer = "isoctsloco"
        config.use_outer_scheduler = False
    elif canonical.lower() in METHOD_REGISTRY:
        config.inner_outer_strategy = "diloco"
        config.inner_optimizer = "adamw"
        config.outer_optimizer = canonical.lower()
        config.use_outer_scheduler = False
    else:
        raise ValueError(
            f"Unknown strategy '{config.strategy}'. Supported strategies: "
            f"{sorted(set(STRATEGY_ALIASES.values()) | set(METHOD_REGISTRY.keys()))}"
        )


def build_outer_optimizer(
    virtual_models: List[nn.Module],
    config: Any,
    virtual_workers: int = 1,
    process_group: Optional[dist.ProcessGroup] = None,
) -> torch.optim.Optimizer:
    """Instantiate the configured outer optimizer from METHOD_REGISTRY."""
    key = config.outer_optimizer.lower()
    if key not in METHOD_REGISTRY:
        raise ValueError(
            f"Unknown outer_optimizer '{config.outer_optimizer}'. Available: {sorted(METHOD_REGISTRY.keys())}"
        )
    return METHOD_REGISTRY[key].builder(
        virtual_models=virtual_models,
        config=config,
        virtual_workers=virtual_workers,
        process_group=process_group,
    )

