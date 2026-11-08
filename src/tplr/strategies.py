"""Backward-compatible re-export of training strategies."""

from training.strategies import Diloco, InnerOuterStrategy, SimpleAccum

__all__ = ["InnerOuterStrategy", "SimpleAccum", "Diloco"]
