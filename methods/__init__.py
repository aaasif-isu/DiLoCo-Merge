"""Aggregation algorithms and outer optimizers for distributed language model training."""

from .base import (
    METHOD_REGISTRY,
    STRATEGY_ALIASES,
    MethodSpec,
    apply_strategy_preset,
    build_outer_optimizer,
    method_handles_own_communication,
    register_method,
)
from .diloco import build_adamw_outer, build_diloco_nesterov, build_diloco_sgd
from .erank_iso import (
    ERANKIsoLoCo,
    ErankIsoLoCo,
    apply_erank_iso,
    compute_shannon_effective_rank,
    erank_iso_transform,
)
from .isocloco import ISOCLoCo
from .rankiso_v2 import (
    RANKIsoV2,
    RANKIsoV2LoCo,
    RankIsoV2,
    RankIsoV2LoCo,
    apply_rankiso_v2,
    rankiso_v2_transform,
)

__all__ = [
    "ERANKIsoLoCo",
    "ErankIsoLoCo",
    "ISOCLoCo",
    "METHOD_REGISTRY",
    "MethodSpec",
    "RANKIsoV2",
    "RANKIsoV2LoCo",
    "RankIsoV2",
    "RankIsoV2LoCo",
    "STRATEGY_ALIASES",
    "apply_erank_iso",
    "apply_rankiso_v2",
    "apply_strategy_preset",
    "build_adamw_outer",
    "build_diloco_nesterov",
    "build_diloco_sgd",
    "build_outer_optimizer",
    "compute_shannon_effective_rank",
    "erank_iso_transform",
    "method_handles_own_communication",
    "rankiso_v2_transform",
    "register_method",
]

