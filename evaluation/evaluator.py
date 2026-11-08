"""Global and per-worker validation loss evaluation."""

import logging
from typing import Any, Dict, List, Optional

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
from torch.utils.data import DataLoader

logger = logging.getLogger("loco")


@torch.no_grad()
def evaluate_global(
    model: nn.Module,
    val_loader: Optional[DataLoader],
    tokenizer: Any,
    device: torch.device,
    world_size: int = 1,
    global_rank: int = 0,
) -> Optional[Dict[str, Any]]:
    """
    Run global validation evaluation on the validation dataset, computing
    token-weighted validation loss and perplexity across all evaluated tokens.
    """
    if val_loader is None:
        return None

    model.eval()
    total_weighted_loss = 0.0
    total_shifted_tokens = 0
    total_batches = 0
    total_tokens = 0

    for batch in val_loader:
        input_ids = batch.to(device, non_blocking=True)
        labels = input_ids.clone()
        labels[labels == tokenizer.pad_token_id] = -100

        with torch.amp.autocast(device_type=device.type, dtype=torch.bfloat16):
            outputs = model(input_ids=input_ids, labels=labels)

        shifted_valid = int((labels[..., 1:] != -100).sum().item())
        batch_valid_tokens = max(1, shifted_valid)
        total_weighted_loss += outputs.loss.item() * batch_valid_tokens
        total_shifted_tokens += batch_valid_tokens
        total_batches += 1
        total_tokens += int((labels != -100).sum().item())

    val_metrics = torch.tensor(
        [total_weighted_loss, total_shifted_tokens, total_batches, total_tokens],
        device=device,
        dtype=torch.float64,
    )
    if world_size > 1 and dist.is_initialized():
        dist.all_reduce(val_metrics, op=dist.ReduceOp.SUM)

    reduced_weighted_loss = val_metrics[0].item()
    reduced_shifted_tokens = int(val_metrics[1].item())
    reduced_batches = int(val_metrics[2].item())
    reduced_tokens = int(val_metrics[3].item())

    val_loss = (
        reduced_weighted_loss / reduced_shifted_tokens
        if reduced_shifted_tokens > 0
        else float("inf")
    )
    val_perplexity = (
        float(np.exp(min(val_loss, 20))) if val_loss < float("inf") else float("inf")
    )

    if global_rank == 0:
        logger.info(
            f"[Validation] Loss: {val_loss:.4f}, Perplexity: {val_perplexity:.2f}, "
            f"Tokens: {reduced_tokens:,}, Batches: {reduced_batches}"
        )

    model.train()

    return {
        "val_loss": val_loss,
        "val_perplexity": val_perplexity,
        "val_tokens": reduced_tokens,
        "val_shifted_tokens": reduced_shifted_tokens,
        "val_batches": reduced_batches,
    }


@torch.no_grad()
def evaluate_multilingual(
    model: nn.Module,
    lang_val_loaders: Dict[str, DataLoader],
    tokenizer: Any,
    device: torch.device,
    world_size: int = 1,
    global_rank: int = 0,
) -> Dict[str, Any]:
    """
    Evaluate model separately on each language's held-out validation loader (e.g. en, fr, de, es)
    and compute both per-language losses/perplexities and global token-weighted / equal-language
    validation loss.
    """
    if not lang_val_loaders:
        return {}

    result: Dict[str, Any] = {}
    total_weighted_loss = 0.0
    total_shifted_tokens = 0
    total_tokens = 0
    total_batches = 0
    per_lang_losses: List[float] = []

    for lang, loader in lang_val_loaders.items():
        lang_res = evaluate_global(
            model=model,
            val_loader=loader,
            tokenizer=tokenizer,
            device=device,
            world_size=world_size,
            global_rank=-1,  # Suppress duplicate per-call print; log summary below
        )
        if lang_res is None:
            continue

        lang_loss = lang_res["val_loss"]
        lang_ppl = lang_res["val_perplexity"]
        lang_tok = lang_res["val_tokens"]
        lang_shifted = lang_res["val_shifted_tokens"]
        lang_batches = lang_res["val_batches"]

        result[f"valid_loss_{lang}"] = lang_loss
        result[f"valid_perplexity_{lang}"] = lang_ppl
        result[f"valid_tokens_{lang}"] = lang_tok

        if np.isfinite(lang_loss) and lang_shifted > 0:
            total_weighted_loss += lang_loss * lang_shifted
            total_shifted_tokens += lang_shifted
            per_lang_losses.append(lang_loss)
        total_tokens += lang_tok
        total_batches += lang_batches

    global_loss = (
        total_weighted_loss / total_shifted_tokens
        if total_shifted_tokens > 0
        else float("inf")
    )
    equal_lang_loss = (
        float(np.mean(per_lang_losses)) if per_lang_losses else global_loss
    )
    global_ppl = (
        float(np.exp(min(global_loss, 20)))
        if global_loss < float("inf")
        else float("inf")
    )

    result["val_loss"] = global_loss
    result["val_loss_equal_language"] = equal_lang_loss
    result["val_perplexity"] = global_ppl
    result["val_tokens"] = total_tokens
    result["val_batches"] = total_batches

    if global_rank == 0:
        lang_summary = ", ".join(
            f"{lang.upper()}: {result[f'valid_loss_{lang}']:.4f} (ppl {result[f'valid_perplexity_{lang}']:.2f})"
            for lang in lang_val_loaders
            if f"valid_loss_{lang}" in result
        )
        logger.info(
            f"[Multilingual Validation] Global Loss: {global_loss:.4f} (ppl {global_ppl:.2f}) | {lang_summary}"
        )

    return result


@torch.no_grad()
def evaluate_per_worker(
    virtual_models: List[nn.Module],
    worker_val_loaders: List[DataLoader],
    tokenizer: Any,
    device: torch.device,
    world_size: int = 1,
    global_rank: int = 0,
    process_group: Optional[dist.ProcessGroup] = None,
) -> Dict[str, Any]:
    """
    Evaluate per-worker validation loss using fixed per-worker evaluation partitions.
    Returns a dictionary containing each worker's validation loss and summary statistics.
    """
    if not worker_val_loaders:
        return {}

    local_losses: List[float] = []
    local_tokens: List[int] = []

    for idx, vloader in enumerate(worker_val_loaders):
        vmodel = virtual_models[idx] if idx < len(virtual_models) else virtual_models[0]
        vmodel.eval()

        w_loss = 0.0
        w_batches = 0
        w_tokens = 0

        for batch in vloader:
            input_ids = batch.to(device, non_blocking=True)
            labels = input_ids.clone()
            labels[labels == tokenizer.pad_token_id] = -100

            with torch.amp.autocast(device_type=device.type, dtype=torch.bfloat16):
                outputs = vmodel(input_ids=input_ids, labels=labels)

            w_loss += outputs.loss.item()
            w_batches += 1
            w_tokens += (labels != -100).sum().item()

        vmodel.train()
        avg_loss = w_loss / w_batches if w_batches > 0 else float("inf")
        local_losses.append(avg_loss)
        local_tokens.append(w_tokens)

    local_tensor = torch.tensor(
        list(zip(local_losses, local_tokens)),
        device=device,
        dtype=torch.float64,
    )

    if world_size > 1 and dist.is_initialized():
        gathered = [torch.zeros_like(local_tensor) for _ in range(world_size)]
        dist.all_gather(gathered, local_tensor, group=process_group)
        all_worker_stats = torch.cat(gathered, dim=0).cpu().numpy()
    else:
        all_worker_stats = local_tensor.cpu().numpy()

    result: Dict[str, Any] = {}
    valid_losses: List[float] = []
    for wid, (loss_val, tok_val) in enumerate(all_worker_stats):
        loss_f = float(loss_val)
        result[f"worker_{wid}_val_loss"] = loss_f
        result[f"worker_{wid}_val_tokens"] = int(tok_val)
        if np.isfinite(loss_f):
            valid_losses.append(loss_f)

    if valid_losses:
        result["mean_worker_val_loss"] = float(np.mean(valid_losses))
        result["std_worker_val_loss"] = float(np.std(valid_losses))

    if global_rank == 0 and valid_losses:
        logger.info(
            f"[Per-Worker Validation] Mean Loss: {result['mean_worker_val_loss']:.4f} "
            f"(std: {result['std_worker_val_loss']:.4f}) across {len(valid_losses)} workers"
        )

    return result
