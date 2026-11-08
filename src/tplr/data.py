"""Backward-compatible re-export of data utilities."""

from data.dataset import ShardedGPUDataset, get_dataloader

__all__ = ["ShardedGPUDataset", "get_dataloader"]
