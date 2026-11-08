"""Soft Signal-Selective IsoLoCo (soft_signal_iso): Adaptive capped soft-thresholding of singular values."""

import logging
import math
from typing import Any, Dict, Iterable, List, Optional, Tuple, TypeAlias, Union

import torch
import torch.distributed as dist

ParamsT: TypeAlias = Union[Iterable[torch.Tensor], Iterable[dict[str, Any]]]

logger = logging.getLogger("loco")


def gavish_donoho_omega(beta: float) -> float:
    """
    Compute the Gavish-Donoho polynomial approximation omega(beta) for the
    optimal hard threshold coefficient under unknown noise variance.
    """
    return 0.56 * (beta**3) - 0.95 * (beta**2) + 1.82 * beta + 1.43


def soft_signal_iso_transform(
    grad: torch.Tensor,
    eps: float = 1e-12,
) -> Tuple[torch.Tensor, Dict[str, Any]]:
    """
    Apply adaptive capped soft-thresholding (soft_signal_iso) to a 2D pseudo-gradient matrix.

    Steps:
        1. Preserve original dtype and compute reduced SVD in float32:
           U, S, Vh = torch.linalg.svd(G.float(), full_matrices=False).
        2. beta = min(G.shape) / max(G.shape).
        3. omega = 0.56 * beta**3 - 0.95 * beta**2 + 1.82 * beta + 1.43.
        4. tau = omega * torch.quantile(S, 0.5).
        5. keep = S > tau.
        6. If no directions are selected, return original matrix unchanged.
        7. Otherwise compute:
           cap = S[keep].mean()
           S_new = torch.minimum(torch.clamp(S - tau, min=0), cap)
        8. Reconstruct G_new = (U * S_new.unsqueeze(0)) @ Vh.
        9. Restore original dtype and preserve device.

    Returns:
        Tuple of (transformed_tensor, per_matrix_diagnostics).
    """
    if grad.dim() != 2 or grad.numel() == 0:
        return grad, {
            "eligible": False,
            "k": 0,
            "capped_count": 0,
            "total_sv": 0,
            "discarded_energy": 0.0,
            "total_energy": 0.0,
            "diff_sq_norm": 0.0,
            "orig_sq_norm": 0.0,
            "corrected_sq_norm": 0.0,
            "isoloco_sq_norm": 0.0,
            "fallback": False,
        }

    orig_dtype = grad.dtype
    m, n = grad.shape
    r = min(m, n)

    grad_fp32 = grad.detach().float()

    if not torch.isfinite(grad_fp32).all():
        logger.warning(
            "[SoftSignalIsoLoCo] Non-finite values detected in 2D pseudo-gradient; "
            "preserving original non-finite tensor without zeroing."
        )
        return grad.clone(), {
            "eligible": True,
            "k": 0,
            "capped_count": 0,
            "total_sv": r,
            "discarded_energy": 0.0,
            "total_energy": 0.0,
            "diff_sq_norm": 0.0,
            "orig_sq_norm": float("nan"),
            "corrected_sq_norm": float("nan"),
            "isoloco_sq_norm": float("nan"),
            "fallback": True,
        }

    if not torch.any(grad_fp32 != 0):
        return torch.zeros_like(grad), {
            "eligible": True,
            "k": 0,
            "capped_count": 0,
            "total_sv": r,
            "discarded_energy": 0.0,
            "total_energy": 0.0,
            "diff_sq_norm": 0.0,
            "orig_sq_norm": 0.0,
            "corrected_sq_norm": 0.0,
            "isoloco_sq_norm": 0.0,
            "fallback": True,
        }

    orig_sq_norm = float(torch.sum(grad_fp32.to(dtype=torch.float64).square()).item())

    # 1. Reduced SVD in float32
    U, S, Vh = torch.linalg.svd(grad_fp32, full_matrices=False)

    # 2-4. Gavish-Donoho threshold
    beta = float(min(m, n)) / float(max(m, n))
    omega = gavish_donoho_omega(beta)
    tau = omega * torch.quantile(S.float(), 0.5)

    # 5. Select singular directions above threshold
    keep = S > tau
    k = int(keep.sum().item())
    total_sv = int(S.numel())
    s_fp64 = S.to(dtype=torch.float64)
    s_sq = s_fp64.square()
    total_energy = float(s_sq.sum().item())

    # Standard IsoLoCo squared Frobenius norm on the same input matrix: (sqrt(len(S)) * S.mean())^2
    s_mean = float(s_fp64.mean().item())
    isoloco_norm = math.sqrt(float(total_sv)) * s_mean
    isoloco_sq_norm = isoloco_norm * isoloco_norm

    # 6-9. Adaptive capped soft-thresholding or fallback
    if k > 0:
        cap = S[keep].mean()
        s_shifted = torch.clamp(S - tau, min=0.0)
        S_new = torch.minimum(s_shifted, cap)
        capped_count = int((s_shifted >= cap).sum().item())
        G_new_fp32 = (U * S_new.unsqueeze(0)) @ Vh
        discarded_energy = float(s_sq[~keep].sum().item())
        corrected_sq_norm = float(
            torch.sum(G_new_fp32.to(dtype=torch.float64).square()).item()
        )
        diff_fp64 = (G_new_fp32 - grad_fp32).to(dtype=torch.float64)
        diff_sq_norm = float(torch.sum(diff_fp64.square()).item())
        fallback = False
        transformed = G_new_fp32.to(dtype=orig_dtype)
    else:
        capped_count = 0
        discarded_energy = 0.0
        corrected_sq_norm = orig_sq_norm
        diff_sq_norm = 0.0
        fallback = True
        transformed = grad.clone()

    return transformed, {
        "eligible": True,
        "k": k,
        "capped_count": capped_count,
        "total_sv": total_sv,
        "discarded_energy": discarded_energy,
        "total_energy": total_energy,
        "diff_sq_norm": diff_sq_norm,
        "orig_sq_norm": orig_sq_norm,
        "corrected_sq_norm": corrected_sq_norm,
        "isoloco_sq_norm": isoloco_sq_norm,
        "fallback": fallback,
    }


def apply_soft_signal_iso(grad: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    """Apply soft_signal_iso transformation and return the transformed tensor."""
    transformed, _ = soft_signal_iso_transform(grad, eps=eps)
    return transformed


class SoftSignalIsoLoCo(torch.optim.Optimizer):
    """
    Soft Signal-Selective IsoLoCo outer optimizer (soft_signal_iso).

    Averages worker pseudo-gradients globally, applies adaptive capped soft-thresholding
    to eligible 2D matrices, and passes the corrected pseudo-gradient into Nesterov momentum.
    """

    def __init__(
        self,
        models_params: ParamsT,
        lr: float = 1.0,
        momentum: float = 0.0,
        nesterov: bool = False,
        orthogonalize_object: str = "gradient",
        exclude_keys: Optional[List[str]] = None,
        virtual_workers: int = 1,
        process_group: Optional[dist.ProcessGroup] = None,
        eps: float = 1e-12,
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
        self.eps = eps

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

        self.last_metrics: Dict[str, float] = {
            "soft_signal_iso_rank_fraction": 0.0,
            "soft_signal_iso_discarded_energy_fraction": 0.0,
            "soft_signal_iso_correction_ratio": 0.0,
            "soft_signal_iso_fallback_fraction": 0.0,
            "soft_signal_iso_cap_fraction": 0.0,
            "soft_signal_iso_corrected_to_original_norm_ratio": 0.0,
            "soft_signal_iso_corrected_to_isoloco_norm_ratio": 0.0,
            "signal_iso_rank_fraction": 0.0,
            "signal_iso_discarded_energy_fraction": 0.0,
            "signal_iso_correction_ratio": 0.0,
            "signal_iso_fallback_fraction": 0.0,
            "signal_iso_cap_fraction": 0.0,
            "signal_iso_corrected_to_original_norm_ratio": 0.0,
            "signal_iso_corrected_to_isoloco_norm_ratio": 0.0,
        }

    def state_dict(self) -> Dict[str, Any]:
        """Return optimizer state_dict including momentum buffers for checkpointing."""
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
        return sd

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        """Restore optimizer state_dict and synchronize momentum_buffers."""
        saved_buffers = state_dict.get("momentum_buffers", None)
        state_copy = {k: v for k, v in state_dict.items() if k != "momentum_buffers"}
        super().load_state_dict(state_copy)

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

    def _should_apply_soft_signal_iso(self, param_name: str, grad: torch.Tensor) -> bool:
        """Determine whether to apply soft_signal_iso to this parameter."""
        for exclude_key in self.exclude_keys:
            if exclude_key in param_name:
                return False
        return grad.dim() == 2

    def _apply_soft_signal_iso(
        self, grad: torch.Tensor
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """Apply soft_signal_iso transformation to a 2D tensor and return diagnostics."""
        return soft_signal_iso_transform(grad, eps=self.eps)

    def get_diagnostics(self) -> Dict[str, float]:
        """Return lightweight per-round aggregate spectral diagnostics."""
        return dict(self.last_metrics)

    @torch.no_grad()
    def step(self, closure=None):
        """Perform a single outer optimization step."""
        if closure is not None:
            closure()

        lr = self.lr

        total_retained_sv = 0
        total_capped_sv = 0
        total_examined_sv = 0
        total_discarded_energy = 0.0
        total_original_energy = 0.0
        total_diff_sq_norm = 0.0
        total_orig_sq_norm = 0.0
        total_corrected_sq_norm = 0.0
        total_isoloco_sq_norm = 0.0
        total_fallback_matrices = 0
        total_eligible_matrices = 0

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
                if self._should_apply_soft_signal_iso(param_name, grad_mean):
                    merged, stats = self._apply_soft_signal_iso(grad_mean)
                    total_eligible_matrices += 1
                    total_retained_sv += stats["k"]
                    total_capped_sv += stats["capped_count"]
                    total_examined_sv += stats["total_sv"]
                    total_discarded_energy += stats["discarded_energy"]
                    total_original_energy += stats["total_energy"]
                    total_diff_sq_norm += stats["diff_sq_norm"]
                    total_orig_sq_norm += stats["orig_sq_norm"]
                    total_corrected_sq_norm += stats["corrected_sq_norm"]
                    total_isoloco_sq_norm += stats["isoloco_sq_norm"]
                    if stats["fallback"]:
                        total_fallback_matrices += 1
                else:
                    merged = grad_mean

                if self.momentum > 0:
                    buf = self.momentum_buffers[p_idx]
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
                    buf.mul_(self.momentum).add_(grad_mean)
                    if self.nesterov:
                        momentum_update = grad_mean.add(buf, alpha=self.momentum)
                    else:
                        momentum_update = buf.clone()
                else:
                    momentum_update = grad_mean

                if self._should_apply_soft_signal_iso(param_name, momentum_update):
                    update, stats = self._apply_soft_signal_iso(momentum_update)
                    total_eligible_matrices += 1
                    total_retained_sv += stats["k"]
                    total_capped_sv += stats["capped_count"]
                    total_examined_sv += stats["total_sv"]
                    total_discarded_energy += stats["discarded_energy"]
                    total_original_energy += stats["total_energy"]
                    total_diff_sq_norm += stats["diff_sq_norm"]
                    total_orig_sq_norm += stats["orig_sq_norm"]
                    total_corrected_sq_norm += stats["corrected_sq_norm"]
                    total_isoloco_sq_norm += stats["isoloco_sq_norm"]
                    if stats["fallback"]:
                        total_fallback_matrices += 1
                else:
                    update = momentum_update

            for vmodel_idx, vmodel_params in enumerate(self.models_params):
                p = vmodel_params[p_idx]
                if not p.requires_grad or p.grad is None:
                    continue

                p.grad = update.clone()
                p.data.add_(p.grad, alpha=-lr)

        rank_fraction = (
            float(total_retained_sv) / float(total_examined_sv)
            if total_examined_sv > 0
            else 0.0
        )
        cap_fraction = (
            float(total_capped_sv) / float(total_examined_sv)
            if total_examined_sv > 0
            else 0.0
        )
        discarded_energy_fraction = (
            float(total_discarded_energy) / float(total_original_energy)
            if total_original_energy > 0.0
            else 0.0
        )
        if not math.isfinite(total_orig_sq_norm) or not math.isfinite(total_diff_sq_norm):
            correction_ratio = float("nan")
        elif total_orig_sq_norm > 0.0:
            correction_ratio = math.sqrt(total_diff_sq_norm) / (
                math.sqrt(total_orig_sq_norm) + self.eps
            )
        else:
            correction_ratio = 0.0

        if not math.isfinite(total_orig_sq_norm) or not math.isfinite(
            total_corrected_sq_norm
        ):
            corrected_to_orig_ratio = float("nan")
        elif total_orig_sq_norm > 0.0:
            corrected_to_orig_ratio = math.sqrt(total_corrected_sq_norm) / (
                math.sqrt(total_orig_sq_norm) + self.eps
            )
        else:
            corrected_to_orig_ratio = 0.0

        if not math.isfinite(total_isoloco_sq_norm) or not math.isfinite(
            total_corrected_sq_norm
        ):
            corrected_to_isoloco_ratio = float("nan")
        elif total_isoloco_sq_norm > 0.0:
            corrected_to_isoloco_ratio = math.sqrt(total_corrected_sq_norm) / (
                math.sqrt(total_isoloco_sq_norm) + self.eps
            )
        else:
            corrected_to_isoloco_ratio = 0.0

        fallback_fraction = (
            float(total_fallback_matrices) / float(total_eligible_matrices)
            if total_eligible_matrices > 0
            else 0.0
        )

        self.last_metrics = {
            "soft_signal_iso_rank_fraction": rank_fraction,
            "soft_signal_iso_discarded_energy_fraction": discarded_energy_fraction,
            "soft_signal_iso_correction_ratio": correction_ratio,
            "soft_signal_iso_fallback_fraction": fallback_fraction,
            "soft_signal_iso_cap_fraction": cap_fraction,
            "soft_signal_iso_corrected_to_original_norm_ratio": corrected_to_orig_ratio,
            "soft_signal_iso_corrected_to_isoloco_norm_ratio": corrected_to_isoloco_ratio,
            "signal_iso_rank_fraction": rank_fraction,
            "signal_iso_discarded_energy_fraction": discarded_energy_fraction,
            "signal_iso_correction_ratio": correction_ratio,
            "signal_iso_fallback_fraction": fallback_fraction,
            "signal_iso_cap_fraction": cap_fraction,
            "signal_iso_corrected_to_original_norm_ratio": corrected_to_orig_ratio,
            "signal_iso_corrected_to_isoloco_norm_ratio": corrected_to_isoloco_ratio,
        }

        is_main_rank = (not self.is_distributed) or (dist.get_rank(self.process_group) == 0)
        if is_main_rank:
            logger.info(
                "[SoftSignalIsoLoCo] rank_fraction=%.4f, cap_fraction=%.4f, "
                "discarded_energy_fraction=%.4f, correction_ratio=%.4f, "
                "corrected_to_orig_norm=%.4f, corrected_to_isoloco_norm=%.4f, "
                "fallback_fraction=%.4f",
                rank_fraction,
                cap_fraction,
                discarded_energy_fraction,
                correction_ratio,
                corrected_to_orig_ratio,
                corrected_to_isoloco_ratio,
                fallback_fraction,
            )




