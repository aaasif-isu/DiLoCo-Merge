"""TSVLoCo: Task Singular Value merging-based gradient aggregation optimizer."""

from typing import Any, Iterable, List, Optional, TypeAlias, Union

import torch
import torch.distributed as dist

ParamsT: TypeAlias = Union[Iterable[torch.Tensor], Iterable[dict[str, Any]]]


class TSVLoCo(torch.optim.Optimizer):
    """
    TSVLoCo optimizer that aggregates gradients using SVD-based merging.

    This optimizer implements the TSV-Merge algorithm which:
    1. Computes SVD of each gradient and retains a fraction of singular components
    2. Concatenates the reduced singular matrices
    3. Optionally decorrelates by computing SVD of concatenated matrices
    4. Reconstructs the merged gradient from the (optionally orthogonalized) components
    """

    def __init__(
        self,
        models_params: ParamsT,
        lr: float = 1.0,
        singular_ratio: Optional[float] = None,
        decorrelate: bool = True,
        virtual_workers: int = 1,
        process_group: Optional[dist.ProcessGroup] = None,
    ):
        self.models_params = [list(model_param) for model_param in models_params]
        all_params = [p for model_params in self.models_params for p in model_params]
        defaults = dict(lr=lr)
        super().__init__(all_params, defaults)

        self.lr = lr
        self.virtual_workers = virtual_workers
        self.process_group = process_group
        self.is_distributed = dist.is_initialized() and process_group is not None

        if self.is_distributed:
            total_workers = virtual_workers * dist.get_world_size(self.process_group)
        else:
            total_workers = virtual_workers

        self.singular_ratio = (
            singular_ratio if singular_ratio is not None else 1.0 / total_workers
        )
        self.decorrelate = decorrelate

    def _all_gather_tensor(self, tensor: torch.Tensor) -> List[torch.Tensor]:
        """Gather tensor from all workers."""
        if not self.is_distributed:
            return [tensor]

        world_size = dist.get_world_size(self.process_group)
        gathered = [torch.zeros_like(tensor) for _ in range(world_size)]
        dist.all_gather(gathered, tensor, group=self.process_group)
        return gathered

    def _svd_low_rank(
        self, grad: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute SVD and retain only top singular_ratio components."""
        U, S, Vh = torch.linalg.svd(grad, full_matrices=False)
        V = Vh.T
        min_dim = min(grad.shape[0], grad.shape[1])
        k = max(1, int(self.singular_ratio * min_dim))

        U_k = U[:, :k]
        S_k = S[:k]
        V_k = V[:, :k]
        return U_k, S_k, V_k

    def _merge_gradients_svd(self, gradients_list: List[torch.Tensor]) -> torch.Tensor:
        """Merge a list of gradients using TSV-Merge algorithm."""
        if gradients_list[0].dim() == 1:
            stacked = torch.stack(gradients_list, dim=0)
            return stacked.mean(dim=0)

        U_list = []
        S_list = []
        V_list = []

        for grad in gradients_list:
            U_k, S_k, V_k = self._svd_low_rank(grad)
            U_list.append(U_k)
            S_list.append(S_k)
            V_list.append(V_k)

        U_concat = torch.cat(U_list, dim=1)
        V_concat = torch.cat(V_list, dim=1)

        total_k = sum(len(S) for S in S_list)
        Sigma = torch.zeros(
            total_k,
            total_k,
            device=gradients_list[0].device,
            dtype=gradients_list[0].dtype,
        )
        offset = 0
        for S in S_list:
            k = len(S)
            Sigma[offset : offset + k, offset : offset + k] = torch.diag(S)
            offset += k

        if self.decorrelate:
            P_u, D_u, Quh = torch.linalg.svd(U_concat, full_matrices=False)
            P_v, D_v, Qvh = torch.linalg.svd(V_concat, full_matrices=False)

            U_orth = P_u @ Quh
            V_orth = P_v @ Qvh

            merged = U_orth @ Sigma @ V_orth.T
        else:
            merged = U_concat @ Sigma @ V_concat.T

        return merged

    @torch.no_grad()
    def step(self, closure=None):
        """Perform a single optimization step."""
        if closure is not None:
            closure()

        lr = self.lr

        for p_idx in range(len(self.models_params[0])):
            grads_local = []

            for vmodel_idx, vmodel_params in enumerate(self.models_params):
                p = vmodel_params[p_idx]
                if not p.requires_grad or p.grad is None:
                    continue

                grads_local.append(p.grad)

            if not grads_local:
                continue

            local_stacked = torch.stack(grads_local, dim=0)
            gathered = self._all_gather_tensor(local_stacked)

            all_grads = []
            for worker_grads in gathered:
                for v in range(worker_grads.shape[0]):
                    all_grads.append(worker_grads[v])

            merged = self._merge_gradients_svd(all_grads)

            for vmodel_idx, vmodel_params in enumerate(self.models_params):
                p = vmodel_params[p_idx]
                if not p.requires_grad or p.grad is None:
                    continue

                p.grad = merged.clone()
                p.data.add_(p.grad, alpha=-lr)
