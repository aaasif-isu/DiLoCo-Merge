"""Spectral-Ceiling L1 Effective-Rank Tikhonov IsoLoCo (rankiso_v2): Adaptive effective-rank Tikhonov spectral equalization with IsoLoCo spectral-ceiling L1 normalization."""

import logging
import math
import time
from typing import Any, Dict, Iterable, List, Optional, Tuple, TypeAlias, Union

import torch
import torch.distributed as dist

ParamsT: TypeAlias = Union[Iterable[torch.Tensor], Iterable[dict[str, Any]]]

logger = logging.getLogger("loco")


def _zero_erank_stats(
    eligible: bool,
    r: Optional[int] = None,
    correction_time_sec: float = 0.0,
    nan_vals: bool = False,
) -> Dict[str, Any]:
    if not eligible or r is None or r <= 0:
        s_before = None
        s_after = None
    else:
        s_before = torch.zeros(r, dtype=torch.float32, device="cpu")
        s_after = s_before.clone()
    val = float("nan") if nan_vals else 0.0
    return {
        "eligible": eligible,
        "s_before": s_before,
        "s_after": s_after,
        "r_eff": val,
        "ceil_idx": 0,
        "tau": val,
        "orig_sq_norm": val,
        "orig_frob_norm": val,
        "iso_target_sq_norm": val,
        "iso_target_frob_norm": val,
        "trans_sq_norm": val,
        "trans_frob_norm": val,
        "rel_frob_match_error": val,
        "diff_sq_norm": val,
        "sv_sum_before": val,
        "sv_sum_after": val,
        "correction_time_sec": float(correction_time_sec),
    }


def compute_shannon_effective_rank(s: torch.Tensor) -> Tuple[float, int, float]:
    """
    Compute Shannon effective rank r_eff, 1-based ceiling index ceil(r_eff), and
    transition scale tau = sigma_{ceil(r_eff)} for a descending singular-value vector s:
        p_r = sigma_r / sum_j sigma_j
        r_eff = exp(-sum_r p_r log p_r)
        tau = sigma_{ceil(r_eff)}   (1-based index into descending singular values)

    Returns:
        Tuple of (r_eff, ceil_idx_1based, tau).
    """
    r = int(s.numel())
    if r <= 0:
        return 0.0, 0, 0.0

    s_f64 = s.detach().to(dtype=torch.float64)
    s_sum = float(s_f64.sum().item())
    if not math.isfinite(s_sum) or s_sum <= 0.0:
        return 0.0, 0, 0.0

    pos_mask = s_f64 > 0.0
    k_pos = int(pos_mask.sum().item())
    if k_pos <= 0:
        return 0.0, 0, 0.0

    p = s_f64 / s_sum
    log_p = torch.where(pos_mask, torch.log(p.clamp_min(1e-300)), torch.zeros_like(p))
    entropy = -float(torch.sum(p * log_p).item())
    r_eff_raw = math.exp(entropy)
    r_eff = min(float(k_pos), max(1.0, r_eff_raw))

    nearest_int = round(r_eff)
    if abs(r_eff - float(nearest_int)) <= 1e-12:
        ceil_idx = int(nearest_int)
    else:
        ceil_idx = int(math.ceil(r_eff))
    ceil_idx = min(k_pos, max(1, ceil_idx))

    tau = float(s_f64[ceil_idx - 1].item())
    return r_eff, ceil_idx, tau



def rankiso_v2_transform(
    grad: torch.Tensor,
    return_stats: bool = False,
) -> Union[torch.Tensor, Tuple[torch.Tensor, Dict[str, Any]]]:
    """
    Apply Spectral-Ceiling L1 Effective-Rank Tikhonov IsoLoCo transformation (rankiso_v2)
    to a 2D averaged pseudo-gradient G:
      1. Compute reduced SVD: G = U diag(sigma) V^T.
      2. Compute L1 Shannon effective rank:
             p_r = sigma_r / sum_j sigma_j
             r_eff = exp(-sum_r p_r log p_r)
      3. Select transition scale:
             tau_1 = sigma_{ceil(r_eff)}   (1-based index)
      4. Apply Tikhonov spectral shaping:
             psi_i = sigma_i / sqrt(sigma_i^2 + tau_1^2)
      5. Normalize via Spectral-Ceiling L1:
             sigma_tilde_i = sigma_bar * (psi_i / psi_1)
      6. Reconstruct:
             G_tilde = U diag(sigma_tilde) V^T
    """
    t0 = time.perf_counter()

    if grad.dim() != 2 or grad.numel() == 0:
        elapsed = time.perf_counter() - t0
        if return_stats:
            return grad, _zero_erank_stats(eligible=False, correction_time_sec=elapsed)
        return grad

    r_dim = min(int(grad.shape[0]), int(grad.shape[1]))
    work_dtype = (
        grad.dtype
        if grad.dtype in (torch.float32, torch.float64)
        else torch.float32
    )
    work_grad = (
        grad.detach()
        if grad.dtype == work_dtype
        else grad.detach().to(dtype=work_dtype)
    )

    if not torch.isfinite(work_grad).all():
        logger.warning(
            "[RankIsoV2] Non-finite values detected in 2D pseudo-gradient; "
            "returning original tensor without spectral modification."
        )
        elapsed = time.perf_counter() - t0
        if return_stats:
            return grad.clone(), _zero_erank_stats(
                eligible=True, r=r_dim, correction_time_sec=elapsed, nan_vals=True
            )
        return grad.clone()

    if not torch.any(work_grad != 0):
        zero_out = torch.zeros_like(grad)
        elapsed = time.perf_counter() - t0
        if return_stats:
            return zero_out, _zero_erank_stats(
                eligible=True, r=r_dim, correction_time_sec=elapsed
            )
        return zero_out

    orig_dtype = grad.dtype
    U, S, Vh = torch.linalg.svd(work_grad, full_matrices=False)
    r = int(S.numel())
    if r == 0:
        zero_out = torch.zeros_like(grad)
        elapsed = time.perf_counter() - t0
        if return_stats:
            return zero_out, _zero_erank_stats(
                eligible=True, r=0, correction_time_sec=elapsed
            )
        return zero_out

    s_f64 = S.to(dtype=torch.float64)
    s_max = float(s_f64[0].item())
    if s_max <= 0.0:
        zero_out = torch.zeros_like(grad)
        elapsed = time.perf_counter() - t0
        if return_stats:
            return zero_out, _zero_erank_stats(
                eligible=True, r=r, correction_time_sec=elapsed
            )
        return zero_out

    r_eff, ceil_idx, tau = compute_shannon_effective_rank(s_f64)

    # Scale by s_max in float64 before squaring so sigma_r^2 + tau^2 never underflows/overflows at any scale
    s_scaled = s_f64 / s_max
    tau_scaled = tau / s_max
    denom_sq = s_scaled.square() + (tau_scaled * tau_scaled)
    psi_f64 = torch.where(
        denom_sq > 0.0,
        s_scaled / torch.sqrt(denom_sq.clamp_min(1e-300)),
        torch.zeros_like(s_scaled),
    )

    sigma_bar_f64 = s_f64.mean()
    f_iso_f64 = math.sqrt(float(r)) * sigma_bar_f64
    psi_1_f64 = psi_f64[0]

    if float(psi_1_f64.item()) <= 0.0 or float(sigma_bar_f64.item()) <= 0.0:
        zero_out = torch.zeros_like(grad)
        elapsed = time.perf_counter() - t0
        if return_stats:
            return zero_out, _zero_erank_stats(
                eligible=True, r=r, correction_time_sec=elapsed
            )
        return zero_out

    s_tilde_f64 = psi_f64 * (sigma_bar_f64 / psi_1_f64)
    S_tilde = s_tilde_f64.to(dtype=S.dtype)
    merged = torch.linalg.multi_dot((U, torch.diag(S_tilde), Vh))
    out = merged.to(dtype=orig_dtype)
    elapsed = time.perf_counter() - t0

    if not return_stats:
        return out

    orig_sq_norm = float(torch.sum(work_grad.to(dtype=torch.float64).square()).item())
    orig_frob_norm = math.sqrt(orig_sq_norm) if orig_sq_norm > 0.0 else 0.0
    iso_target_frob_norm = float(f_iso_f64.item())
    iso_target_sq_norm = iso_target_frob_norm * iso_target_frob_norm

    trans_sq_norm = float(torch.sum(merged.to(dtype=torch.float64).square()).item())
    trans_frob_norm = math.sqrt(trans_sq_norm) if trans_sq_norm > 0.0 else 0.0

    rel_frob_match_error = (
        abs(trans_frob_norm - iso_target_frob_norm) / iso_target_frob_norm
        if iso_target_frob_norm > 0.0
        else 0.0
    )
    diff_sq_norm = float(
        torch.sum((merged - work_grad).to(dtype=torch.float64).square()).item()
    )
    sv_sum_before = float(s_f64.sum().item())
    sv_sum_after = float(s_tilde_f64.sum().item())

    return out, {
        "eligible": True,
        "s_before": S.detach().to(device="cpu", dtype=torch.float32),
        "s_after": S_tilde.detach().to(device="cpu", dtype=torch.float32),
        "r_eff": float(r_eff),
        "ceil_idx": int(ceil_idx),
        "tau": float(tau),
        "orig_sq_norm": orig_sq_norm,
        "orig_frob_norm": orig_frob_norm,
        "iso_target_sq_norm": iso_target_sq_norm,
        "iso_target_frob_norm": iso_target_frob_norm,
        "trans_sq_norm": trans_sq_norm,
        "trans_frob_norm": trans_frob_norm,
        "rel_frob_match_error": rel_frob_match_error,
        "diff_sq_norm": diff_sq_norm,
        "sv_sum_before": sv_sum_before,
        "sv_sum_after": sv_sum_after,
        "correction_time_sec": float(elapsed),
    }


def apply_rankiso_v2(grad: torch.Tensor) -> torch.Tensor:
    """Apply Spectral-Ceiling L1 Effective-Rank Tikhonov IsoLoCo transformation and return only the 2D tensor."""
    out = rankiso_v2_transform(grad, return_stats=False)
    assert isinstance(out, torch.Tensor)
    return out


class RankIsoV2LoCo(torch.optim.Optimizer):
    """
    Spectral-Ceiling L1 Effective-Rank Tikhonov IsoLoCo outer optimizer (rankiso_v2).

    Averages worker pseudo-gradients, computes one reduced SVD per eligible 2D matrix,
    determines the transition scale tau = sigma_{ceil(r_eff)} from Shannon effective rank,
    applies Tikhonov spectral shaping psi_i = sigma_i / sqrt(sigma_i^2 + tau^2), and
    normalizes via Spectral-Ceiling L1 sigma_tilde_i = sigma_bar * (psi_i / psi_1)
    before the outer Nesterov update.
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
        self.exclude_keys = (
            list(exclude_keys)
            if exclude_keys is not None
            else ["embed_tokens", "wpe", "lm_head"]
        )

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
                    self.momentum_buffers.append(buf)
                    self.state[p]["momentum_buffer"] = buf
                else:
                    self.momentum_buffers.append(None)

        self.last_metrics: Dict[str, float] = {
            "erank_iso_r_eff_mean": 0.0,
            "erank_iso_r_eff_min": 0.0,
            "erank_iso_r_eff_max": 0.0,
            "erank_iso_r_eff_attn_mean": 0.0,
            "erank_iso_r_eff_mlp_mean": 0.0,
            "erank_iso_tau_mean": 0.0,
            "erank_iso_tau_min": 0.0,
            "erank_iso_tau_max": 0.0,
            "erank_iso_tau_attn_mean": 0.0,
            "erank_iso_tau_mlp_mean": 0.0,
            "erank_iso_tau_to_sv_mean_ratio": 0.0,
            "erank_iso_original_frob_norm": 0.0,
            "erank_iso_iso_target_frob_norm": 0.0,
            "erank_iso_transformed_frob_norm": 0.0,
            "erank_iso_relative_frob_match_error": 0.0,
            "erank_iso_max_matrix_rel_frob_match_error": 0.0,
            "erank_iso_sv_sum_before": 0.0,
            "erank_iso_sv_sum_after": 0.0,
            "erank_iso_nuclear_norm_retained_ratio": 0.0,
            "erank_iso_sv_min_before": 0.0,
            "erank_iso_sv_max_before": 0.0,
            "erank_iso_sv_mean_before": 0.0,
            "erank_iso_sv_min_after": 0.0,
            "erank_iso_sv_max_after": 0.0,
            "erank_iso_sv_mean_after": 0.0,
            "erank_iso_transformed_to_original_frob_ratio": 0.0,
            "erank_iso_relative_frobenius_change": 0.0,
            "erank_iso_correction_time_sec": 0.0,
            "erank_iso_step_time_sec": 0.0,
        }

    def get_diagnostics(self) -> Dict[str, float]:
        """Return diagnostic metrics from the most recent outer step."""
        return dict(self.last_metrics)

    def state_dict(self) -> Dict[str, Any]:
        if self.momentum > 0:
            for p, buf in zip(self.models_params[0], self.momentum_buffers):
                if buf is not None:
                    self.state[p]["momentum_buffer"] = buf
        sd = super().state_dict()
        sd["erank_iso_last_metrics"] = dict(self.last_metrics)
        return sd

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        metrics_copy = state_dict.get("erank_iso_last_metrics", None)
        super().load_state_dict(state_dict)
        if self.momentum > 0:
            new_buffers: List[Optional[torch.Tensor]] = []
            for p in self.models_params[0]:
                if p.requires_grad:
                    buf = self.state.get(p, {}).get("momentum_buffer", None)
                    if buf is None:
                        buf = torch.zeros_like(p)
                    else:
                        buf = buf.to(device=p.device, dtype=p.dtype)
                    self.state[p]["momentum_buffer"] = buf
                    new_buffers.append(buf)
                else:
                    new_buffers.append(None)
            self.momentum_buffers = new_buffers
        if isinstance(metrics_copy, dict):
            self.last_metrics.update(metrics_copy)

    def _all_gather_tensor(self, tensor: torch.Tensor) -> List[torch.Tensor]:
        """Gather tensor from all distributed workers."""
        if not self.is_distributed:
            return [tensor]

        world_size = dist.get_world_size(self.process_group)
        gathered = [torch.zeros_like(tensor) for _ in range(world_size)]
        dist.all_gather(gathered, tensor, group=self.process_group)
        return gathered

    def _should_apply_isoc(self, param_name: str, grad: torch.Tensor) -> bool:
        """Determine whether to apply Spectral-Ceiling L1 Effective-Rank Tikhonov IsoLoCo to this parameter."""
        for exclude_key in self.exclude_keys:
            if exclude_key in param_name:
                return False
        return grad.dim() == 2

    def _apply_isoc(self, grad: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """Apply Spectral-Ceiling L1 Effective-Rank Tikhonov IsoLoCo transformation and return (tensor, stats)."""
        res = rankiso_v2_transform(grad, return_stats=True)
        assert isinstance(res, tuple)
        return res

    @torch.no_grad()
    def step(self, closure=None):
        """Perform a single outer optimization step."""
        if closure is not None:
            closure()

        step_t0 = time.perf_counter()
        lr = self.lr

        s_before_chunks: List[torch.Tensor] = []
        s_after_chunks: List[torch.Tensor] = []
        r_eff_all: List[float] = []
        r_eff_attn: List[float] = []
        r_eff_mlp: List[float] = []
        tau_all: List[float] = []
        tau_attn: List[float] = []
        tau_mlp: List[float] = []

        total_orig_sq_norm = 0.0
        total_iso_target_sq_norm = 0.0
        total_trans_sq_norm = 0.0
        total_diff_sq_norm = 0.0
        total_sv_sum_before = 0.0
        total_sv_sum_after = 0.0
        max_matrix_rel_frob_match_error = 0.0
        total_correction_time_sec = 0.0

        def _record_matrix_stats(param_name: str, grad_shape: Tuple[int, ...], stats: Dict[str, Any]) -> None:
            nonlocal total_orig_sq_norm, total_iso_target_sq_norm, total_trans_sq_norm
            nonlocal total_diff_sq_norm, total_sv_sum_before, total_sv_sum_after
            nonlocal max_matrix_rel_frob_match_error, total_correction_time_sec

            if stats["s_before"] is not None:
                s_before_chunks.append(stats["s_before"])
            if stats["s_after"] is not None:
                s_after_chunks.append(stats["s_after"])
            if stats["r_eff"] > 0.0 and math.isfinite(stats["r_eff"]):
                r_eff_all.append(float(stats["r_eff"]))
                tau_all.append(float(stats["tau"]))
                is_attn = any(k in param_name for k in ("q_proj", "k_proj", "v_proj", "o_proj", "attn"))
                is_mlp = any(k in param_name for k in ("c_fc", "c_proj", "up_proj", "down_proj", "gate_proj", "mlp"))
                if is_attn or (not is_mlp and len(grad_shape) == 2 and grad_shape[0] == grad_shape[1]):
                    r_eff_attn.append(float(stats["r_eff"]))
                    tau_attn.append(float(stats["tau"]))
                else:
                    r_eff_mlp.append(float(stats["r_eff"]))
                    tau_mlp.append(float(stats["tau"]))

            total_orig_sq_norm += stats["orig_sq_norm"]
            total_iso_target_sq_norm += stats["iso_target_sq_norm"]
            total_trans_sq_norm += stats["trans_sq_norm"]
            total_diff_sq_norm += stats["diff_sq_norm"]
            total_sv_sum_before += stats["sv_sum_before"]
            total_sv_sum_after += stats["sv_sum_after"]
            total_correction_time_sec += stats["correction_time_sec"]
            if stats["rel_frob_match_error"] > max_matrix_rel_frob_match_error:
                max_matrix_rel_frob_match_error = stats["rel_frob_match_error"]

        for p_idx in range(len(self.models_params[0])):
            grads_local = []
            param_name = f"param_{p_idx}"

            for vmodel_idx, vmodel_params in enumerate(self.models_params):
                p = vmodel_params[p_idx]
                if not p.requires_grad or p.grad is None:
                    continue

                grads_local.append(p.grad)

                if hasattr(p, "param_name"):
                    param_name = p.param_name

            if not grads_local:
                continue

            local_stacked = torch.stack(grads_local, dim=0)
            gathered = self._all_gather_tensor(local_stacked)

            all_grads = []
            for worker_grads in gathered:
                for v in range(worker_grads.shape[0]):
                    all_grads.append(worker_grads[v])

            if self.orthogonalize_object == "gradient":
                grad_mean = torch.stack(all_grads, dim=0).mean(dim=0)
                if self._should_apply_isoc(param_name, grad_mean):
                    merged, stats = self._apply_isoc(grad_mean)
                    _record_matrix_stats(param_name, tuple(grad_mean.shape), stats)
                else:
                    merged = grad_mean

                if self.momentum > 0:
                    buf = self.momentum_buffers[p_idx]
                    assert buf is not None
                    buf.mul_(self.momentum).add_(merged)
                    if self.nesterov:
                        update = merged.add(buf, alpha=self.momentum)
                    else:
                        update = buf.clone()
                else:
                    update = merged
            elif self.orthogonalize_object == "update":
                grad_mean = torch.stack(all_grads, dim=0).mean(dim=0)

                if self.momentum > 0:
                    buf = self.momentum_buffers[p_idx]
                    assert buf is not None
                    buf.mul_(self.momentum).add_(grad_mean)
                    if self.nesterov:
                        momentum_update = grad_mean.add(buf, alpha=self.momentum)
                    else:
                        momentum_update = buf.clone()
                else:
                    momentum_update = grad_mean

                if self._should_apply_isoc(param_name, momentum_update):
                    update, stats = self._apply_isoc(momentum_update)
                    _record_matrix_stats(param_name, tuple(momentum_update.shape), stats)
                else:
                    update = momentum_update

            for vmodel_idx, vmodel_params in enumerate(self.models_params):
                p = vmodel_params[p_idx]
                if not p.requires_grad or p.grad is None:
                    continue

                p.grad = update.clone()
                p.data.add_(p.grad, alpha=-lr)

        step_time_sec = time.perf_counter() - step_t0

        if s_before_chunks and s_after_chunks:
            all_s_before = torch.cat(s_before_chunks, dim=0)
            all_s_after = torch.cat(s_after_chunks, dim=0)
            sv_min_before = float(all_s_before.min().item())
            sv_max_before = float(all_s_before.max().item())
            sv_mean_before = float(all_s_before.mean().item())
            sv_min_after = float(all_s_after.min().item())
            sv_max_after = float(all_s_after.max().item())
            sv_mean_after = float(all_s_after.mean().item())
        else:
            sv_min_before = 0.0
            sv_max_before = 0.0
            sv_mean_before = 0.0
            sv_min_after = 0.0
            sv_max_after = 0.0
            sv_mean_after = 0.0

        r_eff_mean = sum(r_eff_all) / len(r_eff_all) if r_eff_all else 0.0
        r_eff_min = min(r_eff_all) if r_eff_all else 0.0
        r_eff_max = max(r_eff_all) if r_eff_all else 0.0
        r_eff_attn_mean = sum(r_eff_attn) / len(r_eff_attn) if r_eff_attn else 0.0
        r_eff_mlp_mean = sum(r_eff_mlp) / len(r_eff_mlp) if r_eff_mlp else 0.0

        tau_mean = sum(tau_all) / len(tau_all) if tau_all else 0.0
        tau_min = min(tau_all) if tau_all else 0.0
        tau_max = max(tau_all) if tau_all else 0.0
        tau_attn_mean = sum(tau_attn) / len(tau_attn) if tau_attn else 0.0
        tau_mlp_mean = sum(tau_mlp) / len(tau_mlp) if tau_mlp else 0.0
        tau_to_sv_mean_ratio = (
            tau_mean / sv_mean_before if sv_mean_before > 0.0 else 0.0
        )

        orig_frob = math.sqrt(total_orig_sq_norm) if total_orig_sq_norm > 0.0 else 0.0
        iso_target_frob = (
            math.sqrt(total_iso_target_sq_norm)
            if total_iso_target_sq_norm > 0.0
            else 0.0
        )
        trans_frob = (
            math.sqrt(total_trans_sq_norm) if total_trans_sq_norm > 0.0 else 0.0
        )

        if orig_frob > 0.0:
            rel_frob_change = math.sqrt(total_diff_sq_norm) / orig_frob
            trans_to_orig_frob_ratio = trans_frob / orig_frob
        else:
            rel_frob_change = 0.0
            trans_to_orig_frob_ratio = 0.0

        if iso_target_frob > 0.0:
            rel_frob_match_error = abs(trans_frob - iso_target_frob) / iso_target_frob
        else:
            rel_frob_match_error = 0.0

        nuclear_norm_retained_ratio = (
            total_sv_sum_after / total_sv_sum_before
            if total_sv_sum_before > 0.0
            else 0.0
        )

        self.last_metrics = {
            "erank_iso_r_eff_mean": float(r_eff_mean),
            "erank_iso_r_eff_min": float(r_eff_min),
            "erank_iso_r_eff_max": float(r_eff_max),
            "erank_iso_r_eff_attn_mean": float(r_eff_attn_mean),
            "erank_iso_r_eff_mlp_mean": float(r_eff_mlp_mean),
            "erank_iso_tau_mean": float(tau_mean),
            "erank_iso_tau_min": float(tau_min),
            "erank_iso_tau_max": float(tau_max),
            "erank_iso_tau_attn_mean": float(tau_attn_mean),
            "erank_iso_tau_mlp_mean": float(tau_mlp_mean),
            "erank_iso_tau_to_sv_mean_ratio": float(tau_to_sv_mean_ratio),
            "erank_iso_original_frob_norm": float(orig_frob),
            "erank_iso_iso_target_frob_norm": float(iso_target_frob),
            "erank_iso_transformed_frob_norm": float(trans_frob),
            "erank_iso_relative_frob_match_error": float(rel_frob_match_error),
            "erank_iso_max_matrix_rel_frob_match_error": float(max_matrix_rel_frob_match_error),
            "erank_iso_sv_sum_before": float(total_sv_sum_before),
            "erank_iso_sv_sum_after": float(total_sv_sum_after),
            "erank_iso_nuclear_norm_retained_ratio": float(nuclear_norm_retained_ratio),
            "erank_iso_sv_min_before": float(sv_min_before),
            "erank_iso_sv_max_before": float(sv_max_before),
            "erank_iso_sv_mean_before": float(sv_mean_before),
            "erank_iso_sv_min_after": float(sv_min_after),
            "erank_iso_sv_max_after": float(sv_max_after),
            "erank_iso_sv_mean_after": float(sv_mean_after),
            "erank_iso_transformed_to_original_frob_ratio": float(trans_to_orig_frob_ratio),
            "erank_iso_relative_frobenius_change": float(rel_frob_change),
            "erank_iso_correction_time_sec": float(total_correction_time_sec),
            "erank_iso_step_time_sec": float(step_time_sec),
        }

        is_main_rank = (not self.is_distributed) or (
            dist.get_rank(self.process_group) == 0
        )
        if is_main_rank:
            logger.info(
                "[RankIsoV2] r_eff=[mean=%.2f, attn=%.2f, mlp=%.2f], "
                "tau=[mean=%.6f, attn=%.6f, mlp=%.6f, tau/sv_mean=%.3f], "
                "frob=[orig=%.6f, iso_target=%.6f, trans=%.6f, ratio=%.4f], "
                "nuc=[before=%.2f, after=%.2f, retained=%.4f], time=%.4fs",
                r_eff_mean,
                r_eff_attn_mean,
                r_eff_mlp_mean,
                tau_mean,
                tau_attn_mean,
                tau_mlp_mean,
                tau_to_sv_mean_ratio,
                orig_frob,
                iso_target_frob,
                trans_frob,
                trans_to_orig_frob_ratio,
                total_sv_sum_before,
                total_sv_sum_after,
                nuclear_norm_retained_ratio,
                total_correction_time_sec,
            )


RankIsoV2 = RankIsoV2LoCo
RANKIsoV2 = RankIsoV2LoCo
RANKIsoV2LoCo = RankIsoV2LoCo

