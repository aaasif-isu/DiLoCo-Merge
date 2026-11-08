"""Inner and outer optimizer and learning-rate scheduler management."""

import logging
from typing import Any, List, Optional, Tuple

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.optim import AdamW, SGD
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR

from methods import build_outer_optimizer

logger = logging.getLogger("loco")


def resolve_warmup_steps(warmup_steps: float | int, max_steps: int) -> int:
    """Convert fractional warmup_steps (0 < x < 1) into integer step count."""
    if warmup_steps > 0 and warmup_steps < 1:
        return int(warmup_steps * max_steps)
    return int(warmup_steps)


def build_inner_optimizers_and_schedulers(
    virtual_models: List[nn.Module],
    config: Any,
    global_rank: int = 0,
) -> Tuple[List[Optional[torch.optim.Optimizer]], List[Optional[Any]]]:
    """
    Construct inner optimizers and LR schedulers for all virtual models.
    Preserves the exact scheduler mathematics from the official IsoLoCo train.py.
    """
    config.warmup_steps = resolve_warmup_steps(config.warmup_steps, config.max_steps)

    if config.inner_optimizer is None:
        return [None] * len(virtual_models), [None] * len(virtual_models)

    inner_optimizers: List[Optional[torch.optim.Optimizer]] = []
    inner_schedulers: List[Optional[Any]] = []

    if global_rank == 0:
        logger.info(
            f"Initializing {config.inner_optimizer} inner optimizers (lr={config.inner_learning_rate}) "
            f"for {len(virtual_models)} virtual worker(s)."
        )

    total_steps = config.max_steps * config.inner_steps
    warmup_steps = config.warmup_steps * config.inner_steps

    for vm in virtual_models:
        if config.inner_optimizer == "adamw":
            inner_opt = AdamW(
                vm.parameters(),
                lr=config.inner_learning_rate,
                weight_decay=config.weight_decay,
                betas=(0.9, config.inner_beta2),
            )
        elif config.inner_optimizer == "sgd":
            inner_opt = SGD(
                vm.parameters(),
                lr=config.inner_learning_rate,
                momentum=getattr(config, "inner_momentum", 0.0),
                weight_decay=getattr(config, "inner_weight_decay", 0.0),
                nesterov=getattr(config, "inner_nesterov", False),
            )
        else:
            raise ValueError(f"Unsupported inner_optimizer: {config.inner_optimizer}")

        inner_optimizers.append(inner_opt)

        if warmup_steps > 0:
            scheduler1 = LinearLR(
                inner_opt,
                start_factor=0.1,
                end_factor=1.0,
                total_iters=warmup_steps,
            )
            scheduler2 = CosineAnnealingLR(
                inner_opt,
                T_max=max(1, total_steps - warmup_steps),
                eta_min=config.inner_learning_rate * 0.1,
            )
            inner_Scheduler = SequentialLR(
                inner_opt,
                schedulers=[scheduler1, scheduler2],
                milestones=[warmup_steps],
            )
        else:
            inner_Scheduler = CosineAnnealingLR(
                inner_opt,
                T_max=max(1, total_steps),
                eta_min=config.inner_learning_rate * 0.1,
            )

        inner_schedulers.append(inner_Scheduler)

    return inner_optimizers, inner_schedulers


def build_outer_optimizer_and_scheduler(
    virtual_models: List[nn.Module],
    config: Any,
    virtual_workers: int = 1,
    process_group: Optional[dist.ProcessGroup] = None,
    global_rank: int = 0,
) -> Tuple[torch.optim.Optimizer, Optional[Any]]:
    """
    Construct the outer aggregation optimizer and optional outer LR scheduler.
    Preserves the exact outer optimizer and scheduler logic from train.py.
    """
    if global_rank == 0:
        logger.info(
            f"Initializing {config.outer_optimizer} outer optimizer (lr={config.outer_learning_rate})."
        )

    outer_optimizer = build_outer_optimizer(
        virtual_models=virtual_models,
        config=config,
        virtual_workers=virtual_workers,
        process_group=process_group,
    )

    outer_scheduler = None
    if config.use_outer_scheduler:
        warmup_steps = (
            config.outer_warmup_steps
            if config.outer_warmup_steps is not None
            else config.warmup_steps
        )
        warmup_steps = resolve_warmup_steps(warmup_steps, config.max_steps)

        if warmup_steps > 0:
            scheduler1 = LinearLR(
                outer_optimizer,
                start_factor=0.1,
                end_factor=1.0,
                total_iters=warmup_steps,
            )
            scheduler2 = CosineAnnealingLR(
                outer_optimizer,
                T_max=max(1, config.max_steps - warmup_steps),
                eta_min=config.outer_learning_rate * 0.1,
            )
            outer_scheduler = SequentialLR(
                outer_optimizer,
                schedulers=[scheduler1, scheduler2],
                milestones=[warmup_steps],
            )
        else:
            outer_scheduler = CosineAnnealingLR(
                outer_optimizer,
                T_max=max(1, config.max_steps),
                eta_min=config.outer_learning_rate * 0.1,
            )

    return outer_optimizer, outer_scheduler
