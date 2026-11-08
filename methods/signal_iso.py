"""Signal-Selective IsoLoCo (signal_iso): Statistical singular-value selection with isotropic covariance correction."""

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


def signal_selective_iso_transform(
    grad: torch.Tensor,
    eps: float = 1e-12,
) -> Tuple[torch.Tensor, Dict[str, Any]]:
    """
    Apply Signal-Selective IsoLoCo spectral transformation to a 2D pseudo-gradient matrix.

    Steps:
        A. Reduced SVD in float32: G = U diag(sigma) V^T.
        B. Gavish-Donoho threshold: tau = omega(beta) * median(sigma).
        C. Retain singular directions with sigma_j > tau (count k).
        D. If k > 0, replace retained singular values with their mean mu and set
           discarded singular values to 0; if k == 0, return original matrix unchanged.

    Returns:
        Tuple of (transformed_tensor, per_matrix_diagnostics).
    """
    if grad.dim() != 2 or grad.numel() == 0:
        return grad, {
            "eligible": False,
            "k": 0,
            "total_sv": 0,
            "discarded_energy": 0.0,
            "total_energy": 0.0,
            "diff_sq_norm": 0.0,
            "orig_sq_norm": 0.0,
            "fallback": False,
        }

    orig_dtype = grad.dtype
    m, n = grad.shape
    r = min(m, n)

    grad_fp32 = grad.detach().to(dtype=torch.float32)

    if not torch.isfinite(grad_fp32).all():
        logger.warning(
            "[SignalIsoLoCo] Non-finite values detected in 2D pseudo-gradient; "
            "preserving original non-finite tensor without zeroing."
        )
        return grad.clone(), {
            "eligible": True,
            "k": 0,
            "total_sv": r,
            "discarded_energy": 0.0,
            "total_energy": 0.0,
            "diff_sq_norm": 0.0,
            "orig_sq_norm": float("nan"),
            "fallback": True,
        }

    if not torch.any(grad_fp32 != 0):
        return torch.zeros_like(grad), {
            "eligible": True,
            "k": 0,
            "total_sv": r,
            "discarded_energy": 0.0,
            "total_energy": 0.0,
            "diff_sq_norm": 0.0,
            "orig_sq_norm": 0.0,
            "fallback": True,
        }

    orig_sq_norm = float(torch.sum(grad_fp32.to(dtype=torch.float64).square()).item())

    # Step A: Reduced SVD in float32
    U, S, Vh = torch.linalg.svd(grad_fp32, full_matrices=False)

    # Step B: Gavish-Donoho statistical threshold
    beta = float(min(m, n)) / float(max(m, n))
    omega = gavish_donoho_omega(beta)
    tau = omega * torch.quantile(S.float(), 0.5)

    # Step C: Select singular directions
    retained_mask = S > tau
    k = int(retained_mask.sum().item())
    total_sv = int(S.numel())
    s_sq = S.to(dtype=torch.float64).square()
    total_energy = float(s_sq.sum().item())

    # Step D: Apply selective isotropy
    if k > 0:
        mu = S[retained_mask].mean()
        S_tilde = torch.where(retained_mask, mu, torch.zeros_like(S))
        grad_tilde_fp32 = (U * S_tilde.unsqueeze(0)) @ Vh
        discarded_energy = float(s_sq[~retained_mask].sum().item())
        diff_fp64 = (grad_tilde_fp32 - grad_fp32).to(dtype=torch.float64)
        diff_sq_norm = float(torch.sum(diff_fp64.square()).item())
        fallback = False
        transformed = grad_tilde_fp32.to(dtype=orig_dtype)
    else:
        discarded_energy = 0.0
        diff_sq_norm = 0.0
        fallback = True
        transformed = grad.clone()

    return transformed, {
        "eligible": True,
        "k": k,
        "total_sv": total_sv,
        "discarded_energy": discarded_energy,
        "total_energy": total_energy,
        "diff_sq_norm": diff_sq_norm,
        "orig_sq_norm": orig_sq_norm,
        "fallback": fallback,
    }


def apply_signal_iso(grad: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    """Apply Signal-Selective IsoLoCo transformation and return the transformed tensor."""
    transformed, _ = signal_selective_iso_transform(grad, eps=eps)
    return transformed


class SignalIsoLoCo(torch.optim.Optimizer):
    """
    Signal-Selective IsoLoCo outer optimizer.

    Averages worker pseudo-gradients globally, applies Gavish-Donoho singular-value
    selection followed by isotropic rescaling on retained singular directions for eligible
    2D matrices, and passes the corrected pseudo-gradient into Nesterov momentum.
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
                    self.momentum_buffers.append(torch.zeros_like(p))
                else:
                    self.momentum_buffers.append(None)

        self.last_metrics: Dict[str, float] = {
            "signal_iso_rank_fraction": 0.0,
            "signal_iso_discarded_energy_fraction": 0.0,
            "signal_iso_correction_ratio": 0.0,
            "signal_iso_fallback_fraction": 0.0,
        }

    def _all_gather_tensor(self, tensor: torch.Tensor) -> List[torch.Tensor]:
        """Gather tensor from all workers."""
        if not self.is_distributed:
            return [tensor]

        world_size = dist.get_world_size(self.process_group)
        gathered = [torch.zeros_like(tensor) for _ in range(world_size)]
        dist.all_gather(gathered, tensor, group=self.process_group)
        return gathered

    def _should_apply_signal_iso(self, param_name: str, grad: torch.Tensor) -> bool:
        """Determine whether to apply Signal-Selective IsoLoCo to this parameter."""
        for exclude_key in self.exclude_keys:
            if exclude_key in param_name:
                return False
        return grad.dim() == 2

    def _apply_signal_iso(self, grad: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """Apply Signal-Selective IsoLoCo transformation to a 2D tensor and return diagnostics."""
        return signal_selective_iso_transform(grad, eps=self.eps)

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
        total_examined_sv = 0
        total_discarded_energy = 0.0
        total_original_energy = 0.0
        total_diff_sq_norm = 0.0
        total_orig_sq_norm = 0.0
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
                if self._should_apply_signal_iso(param_name, grad_mean):
                    merged, stats = self._apply_signal_iso(grad_mean)
                    total_eligible_matrices += 1
                    total_retained_sv += stats["k"]
                    total_examined_sv += stats["total_sv"]
                    total_discarded_energy += stats["discarded_energy"]
                    total_original_energy += stats["total_energy"]
                    total_diff_sq_norm += stats["diff_sq_norm"]
                    total_orig_sq_norm += stats["orig_sq_norm"]
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

                if self._should_apply_signal_iso(param_name, momentum_update):
                    update, stats = self._apply_signal_iso(momentum_update)
                    total_eligible_matrices += 1
                    total_retained_sv += stats["k"]
                    total_examined_sv += stats["total_sv"]
                    total_discarded_energy += stats["discarded_energy"]
                    total_original_energy += stats["total_energy"]
                    total_diff_sq_norm += stats["diff_sq_norm"]
                    total_orig_sq_norm += stats["orig_sq_norm"]
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
        fallback_fraction = (
            float(total_fallback_matrices) / float(total_eligible_matrices)
            if total_eligible_matrices > 0
            else 0.0
        )

        self.last_metrics = {
            "signal_iso_rank_fraction": rank_fraction,
            "signal_iso_discarded_energy_fraction": discarded_energy_fraction,
            "signal_iso_correction_ratio": correction_ratio,
            "signal_iso_fallback_fraction": fallback_fraction,
        }

        is_main_rank = (not self.is_distributed) or (dist.get_rank(self.process_group) == 0)
        if is_main_rank:
            logger.info(
                "[SignalIsoLoCo] rank_fraction=%.4f, discarded_energy_fraction=%.4f, "
                "correction_ratio=%.4f, fallback_fraction=%.4f",
                rank_fraction,
                discarded_energy_fraction,
                correction_ratio,
                fallback_fraction,
            )


