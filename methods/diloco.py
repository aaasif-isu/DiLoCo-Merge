"""Standard DiLoCo outer optimizers (Nesterov SGD, Vanilla SGD, and AdamW baseline)."""

from typing import Iterable
import torch
from torch.optim import AdamW, SGD


def build_diloco_nesterov(
    params: Iterable[torch.Tensor],
    lr: float = 0.7,
    momentum: float = 0.9,
) -> SGD:
    """Build standard DiLoCo outer optimizer (SGD with Nesterov momentum)."""
    return SGD(
        params,
        lr=lr,
        momentum=momentum,
        nesterov=True,
    )


def build_diloco_sgd(
    params: Iterable[torch.Tensor],
    lr: float = 1.0,
) -> SGD:
    """Build DiLoCo outer optimizer with vanilla SGD (no momentum)."""
    return SGD(
        params,
        lr=lr,
    )


def build_adamw_outer(
    params: Iterable[torch.Tensor],
    lr: float = 1e-3,
    weight_decay: float = 0.1,
    betas: tuple[float, float] = (0.9, 0.95),
) -> AdamW:
    """Build AdamW outer optimizer (used by the data-parallel AdamW baseline)."""
    return AdamW(
        params,
        lr=lr,
        weight_decay=weight_decay,
        betas=betas,
    )
