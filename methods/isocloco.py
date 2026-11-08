"""ISOCLoCo: Isotropic Covariance merging-based gradient aggregation optimizer."""

from typing import Any, Iterable, List, Optional, TypeAlias, Union

import torch
import torch.distributed as dist

ParamsT: TypeAlias = Union[Iterable[torch.Tensor], Iterable[dict[str, Any]]]


class ISOCLoCo(torch.optim.Optimizer):
    """
    ISOCLoCo optimizer that aggregates gradients using ISO-C (Isotropic Covariance) merging.
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
    ):
        self.models_params = [list(model_param) for model_param in models_params]
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

        self.momentum_buffers = []
        if self.momentum > 0:
            for p in self.models_params[0]:
                if p.requires_grad:
                    self.momentum_buffers.append(torch.zeros_like(p))
                else:
                    self.momentum_buffers.append(None)

    def _all_gather_tensor(self, tensor: torch.Tensor) -> List[torch.Tensor]:
        """Gather tensor from all workers."""
        if not self.is_distributed:
            return [tensor]

        world_size = dist.get_world_size(self.process_group)
        gathered = [torch.zeros_like(tensor) for _ in range(world_size)]
        dist.all_gather(gathered, tensor, group=self.process_group)
        return gathered

    def _should_apply_isoc(self, param_name: str, grad: torch.Tensor) -> bool:
        """Determine whether to apply ISO-C merging to this parameter."""
        for exclude_key in self.exclude_keys:
            if exclude_key in param_name:
                return False
        return grad.dim() == 2

    def _apply_isoc(self, grad: torch.Tensor) -> torch.Tensor:
        """Apply ISO-C (Isotropic Covariance) transformation to a 2D tensor."""
        U, S, Vh = torch.linalg.svd(grad, full_matrices=False)
        S_iso = torch.ones_like(S) * S.mean()
        merged = torch.linalg.multi_dot((U, torch.diag(S_iso), Vh))
        return merged


    @torch.no_grad()
    def step(self, closure=None):
        """Perform a single optimization step."""
        if closure is not None:
            closure()

        lr = self.lr

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
                    merged = self._apply_isoc(grad_mean)
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

                if self._should_apply_isoc(param_name, momentum_update):
                    update = self._apply_isoc(momentum_update)
                else:
                    update = momentum_update

            for vmodel_idx, vmodel_params in enumerate(self.models_params):
                p = vmodel_params[p_idx]
                if not p.requires_grad or p.grad is None:
                    continue

                p.grad = update.clone()
                p.data.add_(p.grad, alpha=-lr)
