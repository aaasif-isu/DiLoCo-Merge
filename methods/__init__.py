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
from .isocloco import ISOCLoCo
from .isoctsloco import ISOCTSLoCo
from .mergeloco import MergeLoCo
from .signal_iso import SignalIsoLoCo, apply_signal_iso, signal_selective_iso_transform
from .soft_signal_iso import (
    SoftSignalIsoLoCo,
    apply_soft_signal_iso,
    soft_signal_iso_transform,
)
from .spectral_alloc import (
    MatrixSpectralDecomposition,
    SpectralAllocDecoupledLoCo,
    SpectralAllocLoCo,
    assign_rank_to_dyadic_bin,
    build_matrix_bin_direction,
    compute_dyadic_bin_slices,
    compute_e2_persistence_diagnostics,
    compute_num_dyadic_bins,
    decompose_matrix_dyadic,
    extract_compact_e2_state,
    reconstruct_matrix_from_allocation,
)
from .spectral_alloc_curvature import (
    CalibrationBatch,
    CalibrationStream,
    CurvatureCalibrationStats,
    compute_calibration_gradient_and_ggn,
    compute_decoupled_nesterov_scaling_and_lookahead,
    compute_nesterov_scaling_and_lookahead,
    evaluate_calibration_loss_at_params,
)
from .spectral_alloc_qp import (
    SpectralAllocationQPResult,
    compute_simplex_kkt_residual,
    solve_spectral_allocation_qp,
)
from .tsvloco import TSVLoCo

__all__ = [
    "CalibrationBatch",
    "CalibrationStream",
    "CurvatureCalibrationStats",
    "ISOCLoCo",
    "ISOCTSLoCo",
    "MatrixSpectralDecomposition",
    "MergeLoCo",
    "SignalIsoLoCo",
    "SoftSignalIsoLoCo",
    "SpectralAllocDecoupledLoCo",
    "SpectralAllocLoCo",
    "SpectralAllocationQPResult",
    "TSVLoCo",
    "METHOD_REGISTRY",
    "STRATEGY_ALIASES",
    "MethodSpec",
    "apply_signal_iso",
    "apply_soft_signal_iso",
    "apply_strategy_preset",
    "assign_rank_to_dyadic_bin",
    "build_adamw_outer",
    "build_diloco_nesterov",
    "build_diloco_sgd",
    "build_matrix_bin_direction",
    "build_outer_optimizer",
    "compute_calibration_gradient_and_ggn",
    "compute_decoupled_nesterov_scaling_and_lookahead",
    "compute_dyadic_bin_slices",
    "compute_e2_persistence_diagnostics",
    "compute_nesterov_scaling_and_lookahead",
    "compute_num_dyadic_bins",
    "compute_simplex_kkt_residual",
    "decompose_matrix_dyadic",
    "evaluate_calibration_loss_at_params",
    "extract_compact_e2_state",
    "method_handles_own_communication",
    "reconstruct_matrix_from_allocation",
    "register_method",
    "signal_selective_iso_transform",
    "soft_signal_iso_transform",
    "solve_spectral_allocation_qp",
]
