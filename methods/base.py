"""Registry and factory for outer aggregation methods and strategy presets."""

from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional

import torch
import torch.distributed as dist
import torch.nn as nn

from .diloco import build_adamw_outer, build_diloco_nesterov, build_diloco_sgd
from .erank_iso import ErankIsoLoCo
from .isocloco import ISOCLoCo
from .rankiso_v2 import RankIsoV2LoCo


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
        momentum=config.outer_momentum,
        nesterov=config.outer_nesterov,
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
        "lm_head",
    ]
    return ISOCLoCo(
        [vm.parameters() for vm in virtual_models],
        lr=config.outer_learning_rate,
        momentum=config.outer_momentum,
        nesterov=config.outer_nesterov,
        orthogonalize_object=getattr(config, "isoc_orthogonalize_object", "gradient"),
        exclude_keys=exclude_keys,
        virtual_workers=virtual_workers,
        process_group=process_group,
    )


@register_method("erank_iso", handles_own_communication=True)
@register_method("erank_isoloco", handles_own_communication=True)
def _build_erank_iso(
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
    return ErankIsoLoCo(
        [vm.parameters() for vm in virtual_models],
        lr=config.outer_learning_rate,
        momentum=config.outer_momentum,
        nesterov=config.outer_nesterov,
        orthogonalize_object=getattr(config, "isoc_orthogonalize_object", "gradient"),
        exclude_keys=exclude_keys,
        virtual_workers=virtual_workers,
        process_group=process_group,
    )


@register_method("rankiso_v2", handles_own_communication=True)
@register_method("rankisov2", handles_own_communication=True)
def _build_rankiso_v2(
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
    return RankIsoV2LoCo(
        [vm.parameters() for vm in virtual_models],
        lr=config.outer_learning_rate,
        momentum=config.outer_momentum,
        nesterov=config.outer_nesterov,
        orthogonalize_object=getattr(config, "isoc_orthogonalize_object", "gradient"),
        exclude_keys=exclude_keys,
        virtual_workers=virtual_workers,
        process_group=process_group,
    )


STRATEGY_ALIASES: Dict[str, str] = {
    "adam_baseline": "adam_baseline",
    "adamw": "adam_baseline",
    "adam": "adam_baseline",
    "diloco_baseline": "diloco_baseline",
    "diloco": "diloco_baseline",
    "diloco_sgd_nomom": "diloco_sgd_noMom",
    "diloco_sgd_noMom": "diloco_sgd_noMom",
    "diloco_sgd": "diloco_sgd_noMom",
    "isocloco": "isocloco",
    "isoc": "isocloco",
    "isoloco": "isoloco",
    "erank_iso": "erank_iso",
    "erank_isoloco": "erank_iso",
    "erankisoloco": "erank_iso",
    "rankiso_v2": "rankiso_v2",
    "rankisov2": "rankiso_v2",
}


def apply_strategy_preset(config: Any) -> None:
    """Apply strategy preset overrides to config in-place."""
    if getattr(config, "strategy", None) is None:
        return

    raw_strategy = str(config.strategy)
    canonical = STRATEGY_ALIASES.get(
        raw_strategy, STRATEGY_ALIASES.get(raw_strategy.lower(), raw_strategy)
    )

    if canonical == "adam_baseline":
        config.inner_outer_strategy = "simple_accum"
        config.inner_optimizer = None
        config.outer_optimizer = "adamw"
        config.use_outer_scheduler = True
    elif canonical == "diloco_baseline":
        config.inner_outer_strategy = "diloco"
        config.inner_optimizer = "adamw"
        config.outer_optimizer = "nesterov"
        config.use_outer_scheduler = False
        if (
            not getattr(config, "_outer_momentum_explicit", False)
            and config.outer_momentum == 0.0
        ):
            config.outer_momentum = 0.9
        if config.outer_momentum > 0:
            config.outer_nesterov = True
    elif canonical == "diloco_sgd_noMom":
        config.inner_outer_strategy = "diloco"
        config.inner_optimizer = "adamw"
        config.outer_optimizer = "sgd"
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
        if config.outer_momentum > 0:
            config.outer_nesterov = True
    elif canonical == "erank_iso":
        config.inner_outer_strategy = "diloco"
        config.inner_optimizer = "adamw"
        config.outer_optimizer = "erank_iso"
        config.use_outer_scheduler = False
        if (
            not getattr(config, "_outer_momentum_explicit", False)
            and config.outer_momentum == 0.0
        ):
            config.outer_momentum = 0.8
        if config.outer_momentum > 0:
            config.outer_nesterov = True
    elif canonical == "rankiso_v2":
        config.inner_outer_strategy = "diloco"
        config.inner_optimizer = "adamw"
        config.outer_optimizer = "rankiso_v2"
        config.use_outer_scheduler = False
        if (
            not getattr(config, "_outer_momentum_explicit", False)
            and config.outer_momentum == 0.0
        ):
            config.outer_momentum = 0.8
        if config.outer_momentum > 0:
            config.outer_nesterov = True
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
