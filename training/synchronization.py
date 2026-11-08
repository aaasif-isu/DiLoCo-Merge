"""Worker synchronization and pseudo-gradient computation primitives."""

from abc import ABC, abstractmethod
from typing import List

import torch
import torch.distributed as dist
import torch.nn as nn

from models import unwrap_model


def offload_model_params(model: nn.Module) -> List[torch.Tensor]:
    """Detach and clone model parameters to CPU as the outer-step reference point."""
    actual_model = model.module if hasattr(model, "module") else model
    return [param.data.detach().clone().to("cpu") for param in actual_model.parameters()]


class WorkerSynchronizer(ABC):
    """
    Abstract interface for synchronizing worker pseudo-gradients and model parameters.
    Separates synchronization from outer aggregation so asynchronous execution
    can be implemented later without modifying aggregation methods.
    """

    @abstractmethod
    def prepare_pseudo_gradients(
        self,
        models: List[nn.Module],
        params_offloaded: List[torch.Tensor],
        handles_own_communication: bool,
    ) -> None:
        """Compute pseudo-gradients and synchronize across workers as needed."""
        pass

    @abstractmethod
    def sync_virtual_models(
        self,
        models: List[nn.Module],
        handles_own_communication: bool,
    ) -> None:
        """Broadcast updated master parameters to virtual workers on the same rank."""
        pass


class SynchronousSynchronizer(WorkerSynchronizer):
    """
    Synchronous distributed worker synchronizer.
    Preserves the exact pseudo-gradient computation and all-reduce / parameter reset
    semantics of the official IsoLoCo Diloco strategy.
    """

    def __init__(self, device: torch.device, world_size: int):
        self.device = device
        self.world_size = world_size

    def prepare_pseudo_gradients(
        self,
        models: List[nn.Module],
        params_offloaded: List[torch.Tensor],
        handles_own_communication: bool,
    ) -> None:
        master_params = list(models[0].parameters())
        vmodels_params = [list(vmodel.parameters()) for vmodel in models[1:]]

        for param_idx, param_offloaded in enumerate(params_offloaded):
            saved_param = param_offloaded.to(self.device)
            master_params[param_idx].grad = saved_param - master_params[param_idx].data
            for vmodel_params in vmodels_params:
                if handles_own_communication:
                    vmodel_params[param_idx].grad = (
                        saved_param - vmodel_params[param_idx].data
                    )
                else:
                    master_params[param_idx].grad.add_(
                        saved_param - vmodel_params[param_idx].data
                    )

            if not handles_own_communication:
                master_params[param_idx].grad.div_(len(models))
                if self.world_size > 1:
                    dist.all_reduce(
                        master_params[param_idx].grad, op=dist.ReduceOp.AVG
                    )

            master_params[param_idx].data.copy_(saved_param)
            if handles_own_communication:
                for vmodel_params in vmodels_params:
                    vmodel_params[param_idx].data.copy_(saved_param)

    def sync_virtual_models(
        self,
        models: List[nn.Module],
        handles_own_communication: bool,
    ) -> None:
        if not handles_own_communication and len(models) > 1:
            base_sd = unwrap_model(models[0]).state_dict()
            for vmodel in models[1:]:
                unwrap_model(vmodel).load_state_dict(base_sd, strict=True)
