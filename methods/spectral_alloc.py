"""Joint Spectral Allocation (spectral_alloc): Dyadic rank-bin decomposition, GGN calibration, and Nesterov outer optimizer."""

from dataclasses import dataclass
import copy
import logging
import math
import time
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple, TypeAlias, Union

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn

from models import unwrap_model
from .spectral_alloc_curvature import (
    CalibrationBatch,
    CalibrationStream,
    compute_calibration_gradient_and_ggn,
    compute_decoupled_nesterov_scaling_and_lookahead,
    compute_nesterov_scaling_and_lookahead,
    evaluate_calibration_loss_at_params,
)
from .spectral_alloc_qp import solve_spectral_allocation_qp

ParamsT: TypeAlias = Union[Iterable[torch.Tensor], Iterable[dict[str, Any]]]

logger = logging.getLogger("loco")


def compute_num_dyadic_bins(r: int) -> int:
    """
    Compute the number of dyadic rank bins K_r for rank r:
      - For r <= 0: 0
      - For r == 1: 1
      - For r >= 2: ceil(log2(r))
    """
    if r <= 0:
        return 0
    if r == 1:
        return 1
    return int((r - 1).bit_length())


def assign_rank_to_dyadic_bin(one_indexed_rank: int, k_r: int) -> int:
    """
    Map a 1-indexed singular rank i >= 1 into its dyadic bin index:
        b(i) = min(floor(log2(i)), K_r - 1)   for K_r >= 2,
        b(1) = 0                              for K_r == 1.
    """
    if one_indexed_rank < 1:
        raise ValueError(f"one_indexed_rank must be >= 1, got {one_indexed_rank}")
    if k_r <= 1:
        return 0
    floor_log2_i = int(one_indexed_rank.bit_length() - 1)
    return min(floor_log2_i, k_r - 1)


def compute_dyadic_bin_slices(r: int) -> List[Tuple[int, int]]:
    """
    Return 0-indexed contiguous (start, end) slices for each dyadic bin b in 0..K_r-1.

    For r == 256, this produces 8 bins corresponding to 1-indexed ranks:
        [1], [2-3], [4-7], [8-15], [16-31], [32-63], [64-127], [128-256],
    i.e. 0-indexed slices:
        [(0, 1), (1, 3), (3, 7), (7, 15), (15, 31), (31, 63), (63, 127), (127, 256)].
    """
    if r <= 0:
        return []
    if r == 1:
        return [(0, 1)]
    k_r = compute_num_dyadic_bins(r)
    slices: List[Tuple[int, int]] = []
    for b in range(k_r):
        start_1idx = 1 << b
        if b < k_r - 1:
            end_1idx = (1 << (b + 1)) - 1
        else:
            end_1idx = r
        slices.append((start_1idx - 1, end_1idx))
    return slices


@dataclass
class MatrixSpectralDecomposition:
    """Per-matrix SVD and dyadic bin metadata for an eligible 2D parameter tensor."""

    param_idx: int
    param_name: str
    orig_dtype: torch.dtype
    device: torch.device
    shape: Tuple[int, int]
    U: Optional[torch.Tensor]
    S: Optional[torch.Tensor]
    Vh: Optional[torch.Tensor]
    sigma_bar: float
    bin_slices: List[Tuple[int, int]]
    bin_counts: List[int]
    bin_masses: List[float]
    is_zero: bool
    is_nonfinite: bool
    orig_grad: torch.Tensor


def decompose_matrix_dyadic(
    grad: torch.Tensor,
    param_idx: int = 0,
    param_name: str = "param_0",
) -> MatrixSpectralDecomposition:
    """
    Compute reduced float32 SVD and dyadic rank-bin metadata for a 2D pseudo-gradient matrix.
    """
    orig_dtype = grad.dtype
    device = grad.device
    m, n = int(grad.shape[0]), int(grad.shape[1])
    r = min(m, n)
    bin_slices = compute_dyadic_bin_slices(r)
    bin_counts = [end - start for start, end in bin_slices]

    grad_fp32 = grad.detach().to(dtype=torch.float32)

    if not torch.isfinite(grad_fp32).all():
        logger.warning(
            f"[SpectralAlloc] Non-finite values in 2D pseudo-gradient '{param_name}'; "
            "preserving original tensor."
        )
        return MatrixSpectralDecomposition(
            param_idx=param_idx,
            param_name=param_name,
            orig_dtype=orig_dtype,
            device=device,
            shape=(m, n),
            U=None,
            S=None,
            Vh=None,
            sigma_bar=float("nan"),
            bin_slices=bin_slices,
            bin_counts=bin_counts,
            bin_masses=[0.0 for _ in bin_counts],
            is_zero=False,
            is_nonfinite=True,
            orig_grad=grad.clone(),
        )

    if r == 0 or not torch.any(grad_fp32 != 0):
        return MatrixSpectralDecomposition(
            param_idx=param_idx,
            param_name=param_name,
            orig_dtype=orig_dtype,
            device=device,
            shape=(m, n),
            U=None,
            S=None,
            Vh=None,
            sigma_bar=0.0,
            bin_slices=bin_slices,
            bin_counts=bin_counts,
            bin_masses=[0.0 for _ in bin_counts],
            is_zero=True,
            is_nonfinite=False,
            orig_grad=torch.zeros_like(grad),
        )

    U, S, Vh = torch.linalg.svd(grad_fp32, full_matrices=False)
    sigma_bar = float(S.to(dtype=torch.float64).mean().item())
    bin_masses = [sigma_bar * float(cnt) for cnt in bin_counts]

    return MatrixSpectralDecomposition(
        param_idx=param_idx,
        param_name=param_name,
        orig_dtype=orig_dtype,
        device=device,
        shape=(m, n),
        U=U,
        S=S,
        Vh=Vh,
        sigma_bar=sigma_bar,
        bin_slices=bin_slices,
        bin_counts=bin_counts,
        bin_masses=bin_masses,
        is_zero=False,
        is_nonfinite=False,
        orig_grad=grad,
    )


def build_matrix_bin_direction(
    decomp: MatrixSpectralDecomposition,
    bin_idx: int,
) -> Optional[torch.Tensor]:
    """
    Construct D_{ell, b} = bar_sigma_ell * U_{ell, b} @ V_{ell, b}^T in float32
    for a single matrix decomposition and bin index b.
    """
    if decomp.is_zero or decomp.is_nonfinite or decomp.U is None or decomp.Vh is None:
        return None
    if bin_idx < 0 or bin_idx >= len(decomp.bin_slices):
        return None
    start, end = decomp.bin_slices[bin_idx]
    if start >= end:
        return None
    u_slice = decomp.U[:, start:end]
    vh_slice = decomp.Vh[start:end, :]
    return float(decomp.sigma_bar) * (u_slice @ vh_slice)


def reconstruct_matrix_from_allocation(
    decomp: MatrixSpectralDecomposition,
    s: Union[Sequence[float], np.ndarray],
) -> Tuple[torch.Tensor, float, float]:
    """
    Reconstruct the corrected 2D update:
        widetilde_G_ell(s) = sum_b s_b D_{ell, b} = U_ell diag(widetilde_sigma_ell(s)) V_ell^T
    where widetilde_sigma_{ell, j}(s) = s_{b(j+1)} * bar_sigma_ell.

    Returns:
        Tuple of (reconstructed_tensor_in_orig_dtype, fro_sq_norm, spectral_norm).
    """
    if decomp.is_nonfinite:
        return decomp.orig_grad.clone(), float("nan"), float("nan")
    if decomp.is_zero or decomp.U is None or decomp.Vh is None:
        return torch.zeros_like(decomp.orig_grad), 0.0, 0.0

    r = decomp.U.shape[1]
    s_tilde = torch.zeros(r, dtype=torch.float32, device=decomp.device)
    max_spec = 0.0
    for b_idx, (start, end) in enumerate(decomp.bin_slices):
        if start >= end:
            continue
        s_b = float(s[b_idx]) if b_idx < len(s) else 0.0
        val = s_b * float(decomp.sigma_bar)
        s_tilde[start:end] = val
        if abs(val) > max_spec:
            max_spec = abs(val)

    g_new_fp32 = (decomp.U * s_tilde.unsqueeze(0)) @ decomp.Vh
    fro_sq = float(torch.sum(g_new_fp32.to(dtype=torch.float64).square()).item())
    return g_new_fp32.to(dtype=decomp.orig_dtype), fro_sq, max_spec


def extract_compact_e2_state(
    decompositions: Dict[int, MatrixSpectralDecomposition],
    s_vec: Union[Sequence[float], np.ndarray],
    a_scale: float,
    max_bins: int,
) -> Dict[str, Any]:
    """
    Retain compact CPU SVD factors and bin metadata for round t's spectral-bin
    directions D_b^(t) = direct_sum_ell bar_sigma_ell^(t) U_{ell,b}^(t) V_{ell,b}^(t)^T
    along with s_b^(t) and a^(t) for consecutive-round E2 persistence evaluation.
    """
    matrices_state: Dict[int, Dict[str, Any]] = {}
    for p_idx, decomp in decompositions.items():
        if (
            decomp.is_zero
            or decomp.is_nonfinite
            or decomp.U is None
            or decomp.Vh is None
        ):
            continue
        matrices_state[int(p_idx)] = {
            "U": decomp.U.detach().to(device="cpu", dtype=torch.float32).clone(),
            "Vh": decomp.Vh.detach().to(device="cpu", dtype=torch.float32).clone(),
            "sigma_bar": float(decomp.sigma_bar),
            "bin_slices": list(decomp.bin_slices),
        }
    return {
        "max_bins": int(max_bins),
        "s": [float(s_vec[b]) for b in range(len(s_vec))],
        "a": float(a_scale),
        "matrices": matrices_state,
    }


def compute_e2_persistence_diagnostics(
    eligible_grads: Dict[Any, torch.Tensor],
    prev_state: Optional[Dict[str, Any]],
    default_bins: int = 8,
) -> Dict[str, Any]:
    """
    Compute the E2 spectral-direction persistence diagnostic at the start of round t+1:
        r_b = <G_avg^(t+1), D_b^(t)> / ||D_b^(t)||_F^2
    and cosine similarity cos(G_avg^(t+1), D_b^(t)) using only eligible 2D tensors.

    Zero-norm directions (||D_b^(t)||_F == 0) are handled safely and logged as
    undefined (NaN / status="undefined") rather than treated as zero.
    """
    if prev_state is None:
        num_bins = max(1, int(default_bins))
        metrics: Dict[str, Any] = {
            "spectral_alloc_e2_available": 0.0,
            "spectral_alloc_e2_defined_bins": 0,
            "spectral_alloc_e2_negative_proj_fraction": float("nan"),
            "spectral_alloc_e2_head_mean_r": float("nan"),
            "spectral_alloc_e2_mid_mean_r": float("nan"),
            "spectral_alloc_e2_tail_mean_r": float("nan"),
            "spectral_alloc_e2_mid_tail_mean_r": float("nan"),
            "spectral_alloc_e2_head_minus_mid_tail_r": float("nan"),
            "spectral_alloc_e2_head_minus_tail_r": float("nan"),
            "spectral_alloc_e2_head_mean_cos": float("nan"),
            "spectral_alloc_e2_mid_mean_cos": float("nan"),
            "spectral_alloc_e2_tail_mean_cos": float("nan"),
            "spectral_alloc_e2_mid_tail_mean_cos": float("nan"),
        }
        for b in range(num_bins):
            metrics[f"spectral_alloc_e2_r_{b}"] = float("nan")
            metrics[f"spectral_alloc_e2_cos_{b}"] = float("nan")
            metrics[f"spectral_alloc_e2_prev_s_{b}"] = float("nan")
            metrics[f"spectral_alloc_e2_prev_eff_step_{b}"] = float("nan")
            metrics[f"spectral_alloc_e2_defined_{b}"] = 0.0
            metrics[f"spectral_alloc_e2_status_{b}"] = "undefined"
        return metrics

    prev_s = list(prev_state.get("s", []))
    prev_a = float(prev_state.get("a", float("nan")))
    num_bins = int(prev_state.get("max_bins", max(len(prev_s), default_bins, 1)))

    g_norm_sq = 0.0
    g_nonfinite = False
    for g_mat in eligible_grads.values():
        g_fp64 = g_mat.detach().to(device="cpu", dtype=torch.float64)
        if not torch.isfinite(g_fp64).all():
            g_nonfinite = True
            break
        g_norm_sq += float(torch.sum(g_fp64.square()).item())
    g_norm = math.sqrt(g_norm_sq) if (not g_nonfinite and g_norm_sq > 0.0) else 0.0

    dots = np.zeros(num_bins, dtype=np.float64)
    d_norm_sqs = np.zeros(num_bins, dtype=np.float64)

    if "bin_directions" in prev_state and prev_state["bin_directions"] is not None:
        bin_dirs: List[Dict[Any, torch.Tensor]] = prev_state["bin_directions"]
        for b in range(min(num_bins, len(bin_dirs))):
            dir_b = bin_dirs[b]
            for key, d_mat in dir_b.items():
                d_fp64 = d_mat.detach().to(device="cpu", dtype=torch.float64)
                d_norm_sqs[b] += float(torch.sum(d_fp64.square()).item())
                if key in eligible_grads and not g_nonfinite:
                    g_fp64 = eligible_grads[key].detach().to(
                        device="cpu", dtype=torch.float64
                    )
                    dots[b] += float(torch.sum(g_fp64 * d_fp64).item())
    else:
        matrices_state: Dict[Any, Dict[str, Any]] = prev_state.get("matrices", {})
        for key, m_info in matrices_state.items():
            U_cpu = m_info["U"].to(device="cpu", dtype=torch.float64)
            Vh_cpu = m_info["Vh"].to(device="cpu", dtype=torch.float64)
            sigma_bar = float(m_info["sigma_bar"])
            bin_slices = m_info["bin_slices"]
            g_fp64 = (
                eligible_grads[key].detach().to(device="cpu", dtype=torch.float64)
                if (key in eligible_grads and not g_nonfinite)
                else None
            )
            for b, (start, end) in enumerate(bin_slices):
                if b >= num_bins or start >= end or sigma_bar == 0.0:
                    continue
                d_ell_b = sigma_bar * (U_cpu[:, start:end] @ Vh_cpu[start:end, :])
                d_norm_sqs[b] += float(torch.sum(d_ell_b.square()).item())
                if g_fp64 is not None:
                    dots[b] += float(torch.sum(g_fp64 * d_ell_b).item())

    r_vals: List[float] = []
    cos_vals: List[float] = []
    defined_flags: List[bool] = []

    for b in range(num_bins):
        d_sq = float(d_norm_sqs[b])
        dot_b = float(dots[b])
        if (
            not g_nonfinite
            and math.isfinite(d_sq)
            and d_sq > 0.0
            and math.isfinite(dot_b)
        ):
            r_b = dot_b / d_sq
            d_norm = math.sqrt(d_sq)
            cos_b = dot_b / (g_norm * d_norm) if g_norm > 0.0 else float("nan")
            r_vals.append(r_b)
            cos_vals.append(cos_b)
            defined_flags.append(True)
        else:
            r_vals.append(float("nan"))
            cos_vals.append(float("nan"))
            defined_flags.append(False)

    defined_indices = [b for b, ok in enumerate(defined_flags) if ok]
    num_defined = len(defined_indices)

    if num_defined > 0:
        neg_count = sum(1 for b in defined_indices if r_vals[b] < 0.0)
        neg_frac = float(neg_count) / float(num_defined)
    else:
        neg_frac = float("nan")

    if num_bins >= 4:
        head_bins = [0, 1]
        mid_bins = list(range(2, num_bins - 2))
        tail_bins = [num_bins - 2, num_bins - 1]
    elif num_bins == 3:
        head_bins = [0]
        mid_bins = [1]
        tail_bins = [2]
    elif num_bins == 2:
        head_bins = [0]
        mid_bins = []
        tail_bins = [1]
    else:
        head_bins = [0]
        mid_bins = []
        tail_bins = []
    mid_tail_bins = mid_bins + tail_bins

    def _mean_over_defined(indices: List[int], arr: List[float]) -> float:
        vals = [arr[b] for b in indices if b < len(arr) and math.isfinite(arr[b])]
        return float(sum(vals) / len(vals)) if vals else float("nan")

    head_mean_r = _mean_over_defined(head_bins, r_vals)
    mid_mean_r = _mean_over_defined(mid_bins, r_vals)
    tail_mean_r = _mean_over_defined(tail_bins, r_vals)
    mid_tail_mean_r = _mean_over_defined(mid_tail_bins, r_vals)

    head_minus_mid_tail_r = (
        head_mean_r - mid_tail_mean_r
        if (math.isfinite(head_mean_r) and math.isfinite(mid_tail_mean_r))
        else float("nan")
    )
    head_minus_tail_r = (
        head_mean_r - tail_mean_r
        if (math.isfinite(head_mean_r) and math.isfinite(tail_mean_r))
        else float("nan")
    )

    head_mean_cos = _mean_over_defined(head_bins, cos_vals)
    mid_mean_cos = _mean_over_defined(mid_bins, cos_vals)
    tail_mean_cos = _mean_over_defined(tail_bins, cos_vals)
    mid_tail_mean_cos = _mean_over_defined(mid_tail_bins, cos_vals)

    metrics = {
        "spectral_alloc_e2_available": 1.0,
        "spectral_alloc_e2_defined_bins": int(num_defined),
        "spectral_alloc_e2_negative_proj_fraction": float(neg_frac),
        "spectral_alloc_e2_head_mean_r": float(head_mean_r),
        "spectral_alloc_e2_mid_mean_r": float(mid_mean_r),
        "spectral_alloc_e2_tail_mean_r": float(tail_mean_r),
        "spectral_alloc_e2_mid_tail_mean_r": float(mid_tail_mean_r),
        "spectral_alloc_e2_head_minus_mid_tail_r": float(head_minus_mid_tail_r),
        "spectral_alloc_e2_head_minus_tail_r": float(head_minus_tail_r),
        "spectral_alloc_e2_head_mean_cos": float(head_mean_cos),
        "spectral_alloc_e2_mid_mean_cos": float(mid_mean_cos),
        "spectral_alloc_e2_tail_mean_cos": float(tail_mean_cos),
        "spectral_alloc_e2_mid_tail_mean_cos": float(mid_tail_mean_cos),
    }

    for b in range(num_bins):
        s_prev_b = float(prev_s[b]) if b < len(prev_s) else float("nan")
        eff_step_b = (
            float(prev_a * s_prev_b)
            if (math.isfinite(prev_a) and math.isfinite(s_prev_b))
            else float("nan")
        )
        metrics[f"spectral_alloc_e2_r_{b}"] = float(r_vals[b])
        metrics[f"spectral_alloc_e2_cos_{b}"] = float(cos_vals[b])
        metrics[f"spectral_alloc_e2_prev_s_{b}"] = s_prev_b
        metrics[f"spectral_alloc_e2_prev_eff_step_{b}"] = eff_step_b
        metrics[f"spectral_alloc_e2_defined_{b}"] = 1.0 if defined_flags[b] else 0.0
        metrics[f"spectral_alloc_e2_status_{b}"] = (
            "defined" if defined_flags[b] else "undefined"
        )

    return metrics





class SpectralAllocLoCo(torch.optim.Optimizer):
    """
    Joint Spectral Allocation outer optimizer (spectral_alloc).

    Operates on the globally averaged worker pseudo-gradient, decomposes eligible 2D matrices
    into dyadic rank bins, evaluates calibration gradient and GGN curvature at the Nesterov
    look-ahead point on a training calibration batch, solves the nuclear-norm-preserving QP
    for bin multipliers s_b, and feeds the reconstructed pseudo-gradient into Nesterov momentum.
    """

    def __init__(
        self,
        models_params: ParamsT,
        lr: float = 1.5,
        momentum: float = 0.8,
        nesterov: bool = True,
        orthogonalize_object: str = "gradient",
        exclude_keys: Optional[List[str]] = None,
        virtual_workers: int = 1,
        process_group: Optional[dist.ProcessGroup] = None,
        model: Optional[nn.Module] = None,
        calibration_stream: Optional[Any] = None,
        eval_batch_b_stream: Optional[Any] = None,
        shards_path: Optional[str] = None,
        sequence_length: int = 128,
        calibration_batch_size: int = 8,
        calibration_micro_batch_size: int = 2,
        calibration_token_chunk_size: int = 128,
        pad_token_id: Optional[int] = None,
        seed: int = 42,
        languages: Optional[Sequence[str]] = None,
        eval_actual_calib_loss_change: bool = True,
        kkt_tol: float = 1e-5,
        feas_tol: float = 1e-7,
        decoupled_momentum: bool = False,
    ):
        models_params_list = list(models_params)
        if models_params_list and isinstance(models_params_list[0], torch.Tensor):
            self.models_params = [models_params_list]
        else:
            self.models_params = [list(model_param) for model_param in models_params_list]

        all_params = [p for model_params in self.models_params for p in model_params]
        defaults = dict(lr=lr)
        super().__init__(all_params, defaults)

        self.lr = lr
        self.momentum = momentum
        self.nesterov = nesterov
        self.orthogonalize_object = orthogonalize_object
        self.virtual_workers = virtual_workers
        self.process_group = process_group
        self.is_distributed = dist.is_initialized() and process_group is not None
        self.exclude_keys = exclude_keys or []

        self.model = model
        self.calibration_stream = calibration_stream
        self.eval_batch_b_stream = eval_batch_b_stream
        self.shards_path = shards_path
        self.sequence_length = int(sequence_length)
        self.calibration_batch_size = int(calibration_batch_size)
        self.calibration_micro_batch_size = int(calibration_micro_batch_size)
        self.calibration_token_chunk_size = int(calibration_token_chunk_size)
        self.pad_token_id = pad_token_id
        self.seed = int(seed)
        self.languages = list(languages) if languages else None
        self.eval_actual_calib_loss_change = bool(eval_actual_calib_loss_change)
        self.kkt_tol = float(kkt_tol)
        self.feas_tol = float(feas_tol)
        self.decoupled_momentum = bool(decoupled_momentum)

        if self.orthogonalize_object not in ["gradient", "update"]:
            raise ValueError(
                f"orthogonalize_object must be 'gradient' or 'update', got '{self.orthogonalize_object}'"
            )
        if self.nesterov and self.momentum <= 0:
            raise ValueError("Nesterov momentum requires momentum > 0")

        if self.is_distributed:
            self.total_workers = virtual_workers * dist.get_world_size(self.process_group)
        else:
            self.total_workers = virtual_workers

        self.momentum_buffers: List[Optional[torch.Tensor]] = []
        if self.momentum > 0:
            for p in self.models_params[0]:
                if p.requires_grad:
                    buf = torch.zeros_like(p)
                    self.state[p]["momentum_buffer"] = buf
                    self.momentum_buffers.append(buf)
                else:
                    self.momentum_buffers.append(None)

        self.cumulative_calibration_tokens: int = 0
        self.cumulative_calibration_batches: int = 0
        self.cumulative_batch_b_tokens: int = 0
        self.cumulative_batch_b_batches: int = 0
        self._pending_calibration_stream_state: Optional[Dict[str, Any]] = None
        self._prev_e2_state: Optional[Dict[str, Any]] = None
        self.last_metrics: Dict[str, Any] = {}

    def state_dict(self) -> Dict[str, Any]:
        """Return optimizer state_dict including momentum buffers, calibration stream state, and E2 state."""
        if self.momentum > 0:
            for p_idx, p in enumerate(self.models_params[0]):
                if p_idx < len(self.momentum_buffers):
                    buf = self.momentum_buffers[p_idx]
                    if buf is not None:
                        self.state[p]["momentum_buffer"] = buf
        sd = super().state_dict()
        sd["momentum_buffers"] = [
            buf.detach().clone() if isinstance(buf, torch.Tensor) else None
            for buf in self.momentum_buffers
        ]
        sd["cumulative_calibration_tokens"] = int(self.cumulative_calibration_tokens)
        sd["cumulative_calibration_batches"] = int(self.cumulative_calibration_batches)
        sd["cumulative_batch_b_tokens"] = int(self.cumulative_batch_b_tokens)
        sd["cumulative_batch_b_batches"] = int(self.cumulative_batch_b_batches)
        if self.calibration_stream is not None and hasattr(
            self.calibration_stream, "state_dict"
        ):
            sd["calibration_stream_state"] = self.calibration_stream.state_dict()
        elif self._pending_calibration_stream_state is not None:
            sd["calibration_stream_state"] = copy.deepcopy(
                self._pending_calibration_stream_state
            )
        else:
            sd["calibration_stream_state"] = None
        sd["prev_e2_state"] = (
            copy.deepcopy(self._prev_e2_state)
            if self._prev_e2_state is not None
            else None
        )
        return sd

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        """Restore optimizer state_dict, momentum_buffers, calibration stream state, and E2 state."""
        saved_buffers = state_dict.get("momentum_buffers", None)
        saved_calib_toks = state_dict.get("cumulative_calibration_tokens", None)
        saved_calib_batches = state_dict.get("cumulative_calibration_batches", None)
        saved_b_toks = state_dict.get("cumulative_batch_b_tokens", None)
        saved_b_batches = state_dict.get("cumulative_batch_b_batches", None)
        saved_stream_state = state_dict.get("calibration_stream_state", None)
        saved_e2_state = state_dict.get("prev_e2_state", None)
        state_copy = {
            k: v
            for k, v in state_dict.items()
            if k
            not in (
                "momentum_buffers",
                "cumulative_calibration_tokens",
                "cumulative_calibration_batches",
                "cumulative_batch_b_tokens",
                "cumulative_batch_b_batches",
                "calibration_stream_state",
                "prev_e2_state",
            )
        }
        super().load_state_dict(state_copy)

        if saved_e2_state is not None:
            self._prev_e2_state = copy.deepcopy(saved_e2_state)

        if saved_calib_toks is not None:
            self.cumulative_calibration_tokens = int(saved_calib_toks)
        if saved_calib_batches is not None:
            self.cumulative_calibration_batches = int(saved_calib_batches)
        if saved_b_toks is not None:
            self.cumulative_batch_b_tokens = int(saved_b_toks)
        if saved_b_batches is not None:
            self.cumulative_batch_b_batches = int(saved_b_batches)
        if isinstance(saved_stream_state, dict):
            self._pending_calibration_stream_state = copy.deepcopy(saved_stream_state)
            if (
                saved_calib_toks is None
                and "cumulative_tokens" in saved_stream_state
            ):
                self.cumulative_calibration_tokens = int(
                    saved_stream_state["cumulative_tokens"]
                )
            if (
                saved_calib_batches is None
                and "cumulative_batches" in saved_stream_state
            ):
                self.cumulative_calibration_batches = int(
                    saved_stream_state["cumulative_batches"]
                )
            if (
                saved_b_toks is None
                and "cumulative_eval_tokens" in saved_stream_state
            ):
                self.cumulative_batch_b_tokens = int(
                    saved_stream_state["cumulative_eval_tokens"]
                )
            if (
                saved_b_batches is None
                and "cumulative_eval_batches" in saved_stream_state
            ):
                self.cumulative_batch_b_batches = int(
                    saved_stream_state["cumulative_eval_batches"]
                )
            if self.calibration_stream is not None and hasattr(
                self.calibration_stream, "load_state_dict"
            ):
                self.calibration_stream.load_state_dict(
                    self._pending_calibration_stream_state
                )
                self._pending_calibration_stream_state = None

        if self.momentum > 0:
            new_buffers: List[Optional[torch.Tensor]] = []
            for p_idx, p in enumerate(self.models_params[0]):
                if not p.requires_grad:
                    new_buffers.append(None)
                    continue
                if p in self.state and isinstance(
                    self.state[p].get("momentum_buffer"), torch.Tensor
                ):
                    buf = self.state[p]["momentum_buffer"].to(
                        device=p.device, dtype=p.dtype
                    )
                    self.state[p]["momentum_buffer"] = buf
                    new_buffers.append(buf)
                elif (
                    saved_buffers is not None
                    and p_idx < len(saved_buffers)
                    and isinstance(saved_buffers[p_idx], torch.Tensor)
                ):
                    buf = saved_buffers[p_idx].to(device=p.device, dtype=p.dtype)
                    self.state[p]["momentum_buffer"] = buf
                    new_buffers.append(buf)
                else:
                    buf = torch.zeros_like(p)
                    self.state[p]["momentum_buffer"] = buf
                    new_buffers.append(buf)
            self.momentum_buffers = new_buffers

    def _all_gather_tensor(self, tensor: torch.Tensor) -> List[torch.Tensor]:
        """Gather tensor from all workers."""
        if not self.is_distributed:
            return [tensor]

        world_size = dist.get_world_size(self.process_group)
        gathered = [torch.zeros_like(tensor) for _ in range(world_size)]
        dist.all_gather(gathered, tensor, group=self.process_group)
        return gathered

    def _should_apply_spectral_alloc(self, param_name: str, grad: torch.Tensor) -> bool:
        """Determine whether to apply spectral allocation to this parameter."""
        for exclude_key in self.exclude_keys:
            if exclude_key in param_name:
                return False
        return grad.dim() == 2

    def _ensure_calibration_stream(self) -> Optional[Any]:
        """Lazily initialize the training-split CalibrationStream and apply any pending checkpoint state."""
        if self.calibration_stream is None and self.shards_path:
            self.calibration_stream = CalibrationStream(
                shards_path=self.shards_path,
                sequence_length=self.sequence_length,
                batch_size=self.calibration_batch_size,
                seed=self.seed,
                languages=self.languages,
            )
        if (
            self.calibration_stream is not None
            and self._pending_calibration_stream_state is not None
            and hasattr(self.calibration_stream, "load_state_dict")
        ):
            self.calibration_stream.load_state_dict(
                self._pending_calibration_stream_state
            )
            self._pending_calibration_stream_state = None
        return self.calibration_stream

    def _get_next_calibration_batch(
        self, device: torch.device
    ) -> Optional[CalibrationBatch]:
        """Obtain the next fresh calibration batch from the calibration stream."""
        stream = self._ensure_calibration_stream()
        if stream is None:
            return None
        if isinstance(stream, CalibrationStream):
            return stream.next_batch(device=device)
        if callable(stream):
            raw = stream(device)
        elif hasattr(stream, "__next__"):
            raw = next(stream)
        elif isinstance(stream, Iterable):
            if not hasattr(self, "_stream_iter"):
                self._stream_iter = iter(stream)
            try:
                raw = next(self._stream_iter)
            except StopIteration:
                self._stream_iter = iter(stream)
                raw = next(self._stream_iter)
        else:
            return None

        if isinstance(raw, CalibrationBatch):
            return CalibrationBatch(
                input_ids=raw.input_ids.to(device=device, dtype=torch.long),
                language_slices=dict(raw.language_slices),
            )
        if isinstance(raw, torch.Tensor):
            return CalibrationBatch(
                input_ids=raw.to(device=device, dtype=torch.long),
                language_slices={},
            )
        if isinstance(raw, dict) and "input_ids" in raw:
            return CalibrationBatch(
                input_ids=raw["input_ids"].to(device=device, dtype=torch.long),
                language_slices=dict(raw.get("language_slices", {})),
            )
        return None

    def _get_next_eval_batch_b(
        self, device: torch.device
    ) -> Optional[CalibrationBatch]:
        """
        Obtain independent evaluation batch B from the training-derived calibration
        distribution without advancing or perturbing batch A's RNG state.
        """
        if self.eval_batch_b_stream is not None:
            stream_b = self.eval_batch_b_stream
            if isinstance(stream_b, CalibrationStream):
                return stream_b.next_eval_batch(device=device)
            if callable(stream_b):
                raw = stream_b(device)
            elif hasattr(stream_b, "__next__"):
                raw = next(stream_b)
            elif isinstance(stream_b, Iterable):
                if not hasattr(self, "_stream_b_iter"):
                    self._stream_b_iter = iter(stream_b)
                try:
                    raw = next(self._stream_b_iter)
                except StopIteration:
                    self._stream_b_iter = iter(stream_b)
                    raw = next(self._stream_b_iter)
            else:
                return None

            if isinstance(raw, CalibrationBatch):
                return CalibrationBatch(
                    input_ids=raw.input_ids.to(device=device, dtype=torch.long),
                    language_slices=dict(raw.language_slices),
                )
            if isinstance(raw, torch.Tensor):
                return CalibrationBatch(
                    input_ids=raw.to(device=device, dtype=torch.long),
                    language_slices={},
                )
            if isinstance(raw, dict) and "input_ids" in raw:
                return CalibrationBatch(
                    input_ids=raw["input_ids"].to(device=device, dtype=torch.long),
                    language_slices=dict(raw.get("language_slices", {})),
                )
            return None

        stream = self._ensure_calibration_stream()
        if isinstance(stream, CalibrationStream):
            return stream.next_eval_batch(device=device)
        return None


    def get_diagnostics(self) -> Dict[str, Any]:
        """Return per-round diagnostics for logging."""
        return dict(self.last_metrics)

    @torch.no_grad()
    def step(self, closure=None):
        """Perform a single outer optimization step with Joint Spectral Allocation."""
        if closure is not None:
            closure()

        step_t0 = time.perf_counter()
        lr = float(self.lr)
        master_params = self.models_params[0]
        if not master_params:
            return

        device = master_params[0].device

        # 1. Globally average worker pseudo-gradients and decompose eligible 2D matrices
        svd_t0 = time.perf_counter()
        grad_means: Dict[int, torch.Tensor] = {}
        eligible_grads: Dict[int, torch.Tensor] = {}
        grad_rest: Dict[int, torch.Tensor] = {}
        decompositions: Dict[int, MatrixSpectralDecomposition] = {}
        param_names: List[str] = []
        any_nonfinite = False

        for p_idx in range(len(master_params)):
            grads_local = []
            param_name = f"param_{p_idx}"

            for vmodel_params in self.models_params:
                p = vmodel_params[p_idx]
                if not p.requires_grad or p.grad is None:
                    continue
                grads_local.append(p.grad)
                if hasattr(p, "param_name"):
                    param_name = p.param_name

            param_names.append(param_name)
            if not grads_local:
                continue

            local_stacked = torch.stack(grads_local, dim=0)
            gathered = self._all_gather_tensor(local_stacked)

            all_grads = []
            for worker_grads in gathered:
                for v in range(worker_grads.shape[0]):
                    all_grads.append(worker_grads[v])

            grad_mean = torch.stack(all_grads, dim=0).mean(dim=0)
            grad_means[p_idx] = grad_mean

            if self._should_apply_spectral_alloc(param_name, grad_mean):
                eligible_grads[p_idx] = grad_mean
                decomp = decompose_matrix_dyadic(
                    grad_mean, param_idx=p_idx, param_name=param_name
                )
                decompositions[p_idx] = decomp
                if decomp.is_nonfinite:
                    any_nonfinite = True
            else:
                grad_rest[p_idx] = grad_mean

        svd_time_sec = time.perf_counter() - svd_t0

        # 2. Aggregate global dyadic bin masses m_b = sum_ell bar_sigma_ell * n_{ell, b}
        max_bins = max(
            (len(d.bin_slices) for d in decompositions.values()),
            default=8,
        )
        max_bins = max(1, max_bins)

        # Evaluate E2 spectral-direction persistence against previous round's D_b^(t-1)
        e2_metrics = compute_e2_persistence_diagnostics(
            eligible_grads=eligible_grads,
            prev_state=self._prev_e2_state,
            default_bins=max_bins,
        )
        m_vec = np.zeros(max_bins, dtype=np.float64)
        for decomp in decompositions.values():
            if not decomp.is_zero and not decomp.is_nonfinite:
                for b_idx, mass_b in enumerate(decomp.bin_masses):
                    m_vec[b_idx] += mass_b

        active_mask = m_vec > 0.0
        active_bins = int(np.sum(active_mask))

        # Construct tilde_G(1) across all parameters when decoupled_momentum is enabled
        g_tilde_ones: Dict[int, torch.Tensor] = {}
        if self.decoupled_momentum:
            s_ones = np.ones(max_bins, dtype=np.float64)
            for p_idx in range(len(master_params)):
                if p_idx not in grad_means:
                    continue
                if p_idx in decompositions:
                    g_iso, _, _ = reconstruct_matrix_from_allocation(
                        decomp=decompositions[p_idx],
                        s=s_ones,
                    )
                    g_tilde_ones[p_idx] = g_iso
                else:
                    g_tilde_ones[p_idx] = grad_means[p_idx]

        # 3. Calibration & QP solve
        calib_t0 = time.perf_counter()
        curvature_time_sec = 0.0
        qp_time_sec = 0.0
        batch_b_eval_time_sec = 0.0
        calib_tokens_round = 0
        batch_b_tokens_round = 0
        loss_lookahead = float("nan")
        actual_loss_change = float("nan")
        actual_loss_change_iso = float("nan")
        batch_b_loss_lookahead = float("nan")
        batch_b_loss_selected = float("nan")
        batch_b_loss_isoloco = float("nan")
        batch_b_loss_diff = float("nan")
        batch_b_loss_change = float("nan")
        batch_b_loss_change_iso = float("nan")
        per_lang_calib_loss: Dict[str, float] = {}

        if any_nonfinite:
            s_vec = np.where(active_mask, 1.0, 0.0)
            qp_status = "nonfinite_input_fallback"
            qp_fallback = True
            kkt_residual = float("nan")
            eig_min = float("nan")
            eig_max = float("nan")
            pred_quad_change = float("nan")
            pred_quad_change_iso = float("nan")
        elif active_bins == 0:
            s_vec = np.zeros(max_bins, dtype=np.float64)
            qp_status = "zero_mass"
            qp_fallback = False
            kkt_residual = 0.0
            eig_min = 0.0
            eig_max = 0.0
            pred_quad_change = 0.0
            pred_quad_change_iso = 0.0
        elif active_bins == 1:
            s_vec = np.where(active_mask, 1.0, 0.0)
            qp_status = "single_active_bin"
            qp_fallback = False
            kkt_residual = 0.0
            eig_min = 0.0
            eig_max = 0.0
            pred_quad_change = 0.0
            pred_quad_change_iso = 0.0
        else:
            calib_batch = (
                self._get_next_calibration_batch(device=device)
                if self.model is not None
                else None
            )
            if calib_batch is None or self.model is None:
                s_vec = np.where(active_mask, 1.0, 0.0)
                qp_status = "no_calibration_stream_fallback"
                qp_fallback = True
                kkt_residual = 0.0
                eig_min = 0.0
                eig_max = 0.0
                pred_quad_change = 0.0
                pred_quad_change_iso = 0.0
            else:
                calib_tokens_round = int(calib_batch.input_ids.numel())
                self.cumulative_calibration_tokens += calib_tokens_round
                self.cumulative_calibration_batches += 1

                base_model = unwrap_model(self.model)
                named_params = list(base_model.named_parameters())

                if self.decoupled_momentum:
                    a_scale, theta_hat_fp32 = (
                        compute_decoupled_nesterov_scaling_and_lookahead(
                            named_params=named_params,
                            momentum_buffers=self.momentum_buffers,
                            g_tilde_ones=g_tilde_ones,
                            lr=self.lr,
                            momentum=self.momentum,
                            nesterov=self.nesterov,
                        )
                    )
                else:
                    a_scale, theta_hat_fp32 = compute_nesterov_scaling_and_lookahead(
                        named_params=named_params,
                        momentum_buffers=self.momentum_buffers,
                        grad_rest=grad_rest,
                        lr=self.lr,
                        momentum=self.momentum,
                        nesterov=self.nesterov,
                    )

                bin_directions: List[Dict[str, torch.Tensor]] = [
                    {} for _ in range(max_bins)
                ]
                for p_idx, decomp in decompositions.items():
                    if p_idx < len(named_params):
                        p_name = named_params[p_idx][0]
                    else:
                        p_name = decomp.param_name
                    for b_idx in range(len(decomp.bin_slices)):
                        if not active_mask[b_idx]:
                            continue
                        d_mat = build_matrix_bin_direction(decomp, b_idx)
                        if d_mat is not None:
                            bin_directions[b_idx][p_name] = d_mat

                curv_stats = compute_calibration_gradient_and_ggn(
                    model=base_model,
                    theta_hat_fp32=theta_hat_fp32,
                    bin_directions=bin_directions,
                    active_bins_mask=active_mask,
                    calib_batch=calib_batch,
                    pad_token_id=self.pad_token_id,
                    micro_batch_size=self.calibration_micro_batch_size,
                    token_chunk_size=self.calibration_token_chunk_size,
                )
                curvature_time_sec = curv_stats.curvature_time_sec
                loss_lookahead = curv_stats.loss_lookahead
                per_lang_calib_loss = dict(curv_stats.per_language_loss)

                qp_res = solve_spectral_allocation_qp(
                    m=m_vec,
                    g=curv_stats.g,
                    K=curv_stats.K,
                    a=a_scale,
                    kkt_tol=self.kkt_tol,
                    feas_tol=self.feas_tol,
                )
                qp_time_sec = qp_res.solve_time_sec
                s_vec = qp_res.s
                qp_status = qp_res.status
                qp_fallback = qp_res.fallback
                kkt_residual = qp_res.kkt_residual
                eig_min = qp_res.eig_min
                eig_max = qp_res.eig_max
                pred_quad_change = qp_res.pred_quad_change
                pred_quad_change_iso = qp_res.pred_quad_change_isoloco

                if self.eval_actual_calib_loss_change:
                    theta_sel_fp32 = {k: v.clone() for k, v in theta_hat_fp32.items()}
                    theta_iso_fp32 = {k: v.clone() for k, v in theta_hat_fp32.items()}
                    for b_idx in range(max_bins):
                        if not active_mask[b_idx]:
                            continue
                        s_b = float(s_vec[b_idx])
                        for p_name, d_mat in bin_directions[b_idx].items():
                            theta_sel_fp32[p_name].add_(d_mat, alpha=-a_scale * s_b)
                            theta_iso_fp32[p_name].add_(d_mat, alpha=-a_scale)

                    loss_sel, _ = evaluate_calibration_loss_at_params(
                        model=base_model,
                        params_fp32=theta_sel_fp32,
                        calib_batch=calib_batch,
                        pad_token_id=self.pad_token_id,
                        micro_batch_size=self.calibration_micro_batch_size,
                    )
                    loss_iso, _ = evaluate_calibration_loss_at_params(
                        model=base_model,
                        params_fp32=theta_iso_fp32,
                        calib_batch=calib_batch,
                        pad_token_id=self.pad_token_id,
                        micro_batch_size=self.calibration_micro_batch_size,
                    )
                    actual_loss_change = float(loss_sel - loss_lookahead)
                    actual_loss_change_iso = float(loss_iso - loss_lookahead)

                    # Independent-batch evaluation (E1) on batch B (does not affect s* or updates)
                    b_t0 = time.perf_counter()
                    calib_batch_b = self._get_next_eval_batch_b(device=device)
                    if calib_batch_b is not None:
                        batch_b_tokens_round = int(calib_batch_b.input_ids.numel())
                        self.cumulative_batch_b_tokens += batch_b_tokens_round
                        self.cumulative_batch_b_batches += 1

                        loss_b_pre, _ = evaluate_calibration_loss_at_params(
                            model=base_model,
                            params_fp32=theta_hat_fp32,
                            calib_batch=calib_batch_b,
                            pad_token_id=self.pad_token_id,
                            micro_batch_size=self.calibration_micro_batch_size,
                        )
                        loss_b_sel, _ = evaluate_calibration_loss_at_params(
                            model=base_model,
                            params_fp32=theta_sel_fp32,
                            calib_batch=calib_batch_b,
                            pad_token_id=self.pad_token_id,
                            micro_batch_size=self.calibration_micro_batch_size,
                        )
                        loss_b_iso, _ = evaluate_calibration_loss_at_params(
                            model=base_model,
                            params_fp32=theta_iso_fp32,
                            calib_batch=calib_batch_b,
                            pad_token_id=self.pad_token_id,
                            micro_batch_size=self.calibration_micro_batch_size,
                        )
                        batch_b_loss_lookahead = float(loss_b_pre)
                        batch_b_loss_selected = float(loss_b_sel)
                        batch_b_loss_isoloco = float(loss_b_iso)
                        batch_b_loss_diff = float(loss_b_sel - loss_b_iso)
                        batch_b_loss_change = float(loss_b_sel - loss_b_pre)
                        batch_b_loss_change_iso = float(loss_b_iso - loss_b_pre)
                    batch_b_eval_time_sec = time.perf_counter() - b_t0

        calibration_time_sec = time.perf_counter() - calib_t0

        # Retain round t's spectral-bin directions D_b^(t), s_b^(t), and a^(t) for round t+1 E2 diagnostic
        if self.decoupled_momentum:
            eff_a_round = float(self.lr)
        else:
            eff_a_round = (
                float(self.lr) * (1.0 + float(self.momentum))
                if (self.nesterov and self.momentum > 0.0)
                else float(self.lr)
            )
        self._prev_e2_state = extract_compact_e2_state(
            decompositions=decompositions,
            s_vec=s_vec,
            a_scale=eff_a_round,
            max_bins=max_bins,
        )

        # 4. Reconstruct corrected 2D matrices and apply outer Nesterov update
        total_update_fro_sq = 0.0
        max_spectral_norm = 0.0

        for p_idx in range(len(master_params)):
            if p_idx not in grad_means:
                continue

            if p_idx in decompositions:
                decomp = decompositions[p_idx]
                merged, fro_sq, spec_norm = reconstruct_matrix_from_allocation(
                    decomp=decomp,
                    s=s_vec,
                )
                total_update_fro_sq += fro_sq
                if math.isfinite(spec_norm) and spec_norm > max_spectral_norm:
                    max_spectral_norm = spec_norm
            else:
                merged = grad_means[p_idx]

            if self.momentum > 0:
                buf = self.momentum_buffers[p_idx]
                buf_input = (
                    g_tilde_ones[p_idx] if self.decoupled_momentum else merged
                )
                buf.mul_(self.momentum).add_(buf_input)
                if self.nesterov:
                    update = merged.add(buf, alpha=self.momentum)
                else:
                    update = buf.clone()
            else:
                update = merged

            for vmodel_params in self.models_params:
                p = vmodel_params[p_idx]
                if not p.requires_grad or p.grad is None:
                    continue
                p.grad = update.clone()
                p.data.add_(p.grad, alpha=-lr)

        update_fro_norm = (
            math.sqrt(total_update_fro_sq)
            if math.isfinite(total_update_fro_sq) and total_update_fro_sq >= 0.0
            else float("nan")
        )
        peak_gpu_memory_mb = (
            float(torch.cuda.max_memory_allocated(device) / (1024.0 * 1024.0))
            if device.type == "cuda" and torch.cuda.is_available()
            else 0.0
        )
        total_step_time_sec = time.perf_counter() - step_t0

        metrics: Dict[str, Any] = {
            "spectral_alloc_active_bins": int(active_bins),
            "spectral_alloc_qp_status": str(qp_status),
            "spectral_alloc_qp_fallback": 1.0 if qp_fallback else 0.0,
            "spectral_alloc_kkt_residual": float(kkt_residual),
            "spectral_alloc_curvature_eig_min": float(eig_min),
            "spectral_alloc_curvature_eig_max": float(eig_max),
            "spectral_alloc_pred_quad_change": float(pred_quad_change),
            "spectral_alloc_pred_quad_change_isoloco": float(pred_quad_change_iso),
            "spectral_alloc_calib_loss_lookahead": float(loss_lookahead),
            "spectral_alloc_actual_calib_loss_change": float(actual_loss_change),
            "spectral_alloc_actual_calib_loss_change_isoloco": float(
                actual_loss_change_iso
            ),
            "spectral_alloc_batch_b_loss_lookahead": float(batch_b_loss_lookahead),
            "spectral_alloc_batch_b_loss_selected": float(batch_b_loss_selected),
            "spectral_alloc_batch_b_loss_isoloco": float(batch_b_loss_isoloco),
            "spectral_alloc_batch_b_loss_diff": float(batch_b_loss_diff),
            "spectral_alloc_batch_b_loss_change": float(batch_b_loss_change),
            "spectral_alloc_batch_b_loss_change_isoloco": float(
                batch_b_loss_change_iso
            ),
            "spectral_alloc_batch_b_eval_time_sec": float(batch_b_eval_time_sec),
            "spectral_alloc_batch_b_tokens": int(batch_b_tokens_round),
            "spectral_alloc_cumulative_batch_b_tokens": int(
                self.cumulative_batch_b_tokens
            ),
            "spectral_alloc_cumulative_batch_b_batches": int(
                self.cumulative_batch_b_batches
            ),
            "spectral_alloc_update_fro_norm": float(update_fro_norm),
            "spectral_alloc_max_spectral_norm": float(max_spectral_norm),
            "spectral_alloc_calibration_time_sec": float(calibration_time_sec),
            "spectral_alloc_svd_time_sec": float(svd_time_sec),
            "spectral_alloc_curvature_time_sec": float(curvature_time_sec),
            "spectral_alloc_qp_time_sec": float(qp_time_sec),
            "spectral_alloc_step_time_sec": float(total_step_time_sec),
            "spectral_alloc_peak_gpu_memory_mb": float(peak_gpu_memory_mb),
            "spectral_alloc_calibration_tokens": int(calib_tokens_round),
            "spectral_alloc_cumulative_calibration_tokens": int(
                self.cumulative_calibration_tokens
            ),
            "spectral_alloc_cumulative_calibration_batches": int(
                self.cumulative_calibration_batches
            ),
        }

        for b_idx in range(len(s_vec)):
            metrics[f"spectral_alloc_s_{b_idx}"] = float(s_vec[b_idx])
            metrics[f"spectral_alloc_mass_{b_idx}"] = float(m_vec[b_idx])

        for lang, lang_loss in per_lang_calib_loss.items():
            metrics[f"spectral_alloc_calib_loss_{lang}"] = float(lang_loss)

        metrics.update(e2_metrics)

        self.last_metrics = metrics

        is_main_rank = (not self.is_distributed) or (
            dist.get_rank(self.process_group) == 0
        )
        if is_main_rank:
            s_formatted = ", ".join(
                f"s_{b}={s_vec[b]:.3f}" for b in range(len(s_vec))
            )
            logger.info(
                "[SpectralAlloc] status=%s, fallback=%s, active_bins=%d, "
                "kkt=%.2e, eig=[%.3e, %.3e], dQ_pred=%.4e (iso=%.4e), "
                "dL_act=%.4e (iso=%.4e), fro=%.4f, max_spec=%.4f, "
                "t_svd=%.3fs, t_curv=%.3fs, t_qp=%.3fs | %s",
                qp_status,
                qp_fallback,
                active_bins,
                kkt_residual,
                eig_min,
                eig_max,
                pred_quad_change,
                pred_quad_change_iso,
                actual_loss_change,
                actual_loss_change_iso,
                update_fro_norm,
                max_spectral_norm,
                svd_time_sec,
                curvature_time_sec,
                qp_time_sec,
                s_formatted,
            )


class SpectralAllocDecoupledLoCo(SpectralAllocLoCo):
    """
    Decoupled Joint Spectral Allocation outer optimizer (spectral_alloc_decoupled).

    Updates the outer Nesterov momentum buffer using the unallocated IsoLoCo update
    tilde_G(1):
        B_t = mu * B_{t-1} + tilde_G(1)
        P_t = tilde_G(s_t) + mu * B_t
        theta_{t+1} = theta_t - lr * P_t
    while calibrating s_t at hat_theta = theta_t - lr * mu^2 * B_{t-1} - lr * mu * tilde_G(1)
    with step scale a = lr.
    """

    def __init__(self, *args, **kwargs):
        kwargs["decoupled_momentum"] = True
        super().__init__(*args, **kwargs)








