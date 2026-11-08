"""Calibration gradient and Generalized Gauss-Newton (GGN) computation for Joint Spectral Allocation."""

import contextlib
import copy
from dataclasses import dataclass, field
import glob
import logging
from pathlib import Path
import time
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.func import functional_call, jvp

from data.partitioning import discover_language_shards
from models import unwrap_model

logger = logging.getLogger("loco")


@contextlib.contextmanager
def _sdpa_math_context():
    """
    Explicitly enable only the PyTorch math SDPA backend during forward-mode AD
    (torch.func.jvp) so CUDA forward-mode differentiation does not dispatch to
    flash or memory-efficient attention kernels that lack JVP rules.
    """
    if (
        hasattr(torch.nn, "attention")
        and hasattr(torch.nn.attention, "sdpa_kernel")
        and hasattr(torch.nn.attention, "SDPBackend")
    ):
        with torch.nn.attention.sdpa_kernel(torch.nn.attention.SDPBackend.MATH):
            yield
    elif hasattr(torch.backends, "cuda") and hasattr(
        torch.backends.cuda, "sdp_kernel"
    ):
        with torch.backends.cuda.sdp_kernel(
            enable_flash=False,
            enable_mem_efficient=False,
            enable_math=True,
        ):
            yield
    else:
        yield



@dataclass
class CalibrationBatch:
    """Container for a calibration batch drawn from the training split."""

    input_ids: torch.Tensor
    language_slices: Dict[str, Tuple[int, int]] = field(default_factory=dict)


@dataclass
class CurvatureCalibrationStats:
    """Output of calibration gradient and GGN computation at the Nesterov look-ahead point."""

    g: np.ndarray
    K: np.ndarray
    loss_lookahead: float
    per_language_loss: Dict[str, float]
    valid_tokens: int
    total_tokens: int
    curvature_time_sec: float


class CalibrationStream:
    """
    Dedicated training-split calibration data stream.

    For multilingual training shards (e.g. train_00_en.npy, train_01_fr.npy, ...),
    draws an equal-language mixture on every step so non-IID experiments calibrate
    on a balanced cross-language sample from the training set (never validation/test).
    """

    def __init__(
        self,
        shards_path: str,
        sequence_length: int,
        batch_size: int = 8,
        seed: int = 42,
        languages: Optional[Sequence[str]] = None,
    ):
        self.shards_path = Path(shards_path)
        self.sequence_length = int(sequence_length)
        self.batch_size = max(1, int(batch_size))
        self.seed = int(seed)
        self.rng = np.random.default_rng(self.seed + 99991)
        self.eval_rng = np.random.default_rng(self.seed + 199993)

        self.lang_arrays: Dict[str, np.ndarray] = {}
        self.lang_num_seqs: Dict[str, int] = {}
        self.single_arrays: List[np.ndarray] = []
        self.single_num_seqs: List[int] = []
        self.cumulative_tokens: int = 0
        self.cumulative_batches: int = 0
        self.cumulative_eval_tokens: int = 0
        self.cumulative_eval_batches: int = 0

        if not self.shards_path.is_dir():
            raise FileNotFoundError(
                f"Calibration shards directory not found: {self.shards_path}"
            )

        discovered = discover_language_shards(self.shards_path, "train")
        if discovered:
            if languages:
                wanted = [lang.lower() for lang in languages]
                filtered = [item for item in discovered if item[0].lower() in wanted]
                if filtered:
                    discovered = filtered
            for lang, shard_name, _ in discovered:
                shard_file = self.shards_path / shard_name
                arr = np.load(str(shard_file), mmap_mode="r")
                n_seq = len(arr) // self.sequence_length
                if n_seq > 0:
                    self.lang_arrays[lang.lower()] = arr
                    self.lang_num_seqs[lang.lower()] = n_seq
        else:
            train_files = sorted(glob.glob(str(self.shards_path / "train_*.npy")))
            if not train_files:
                raise FileNotFoundError(
                    f"No training shards ('train_*.npy') found in {self.shards_path}"
                )
            for fpath in train_files:
                arr = np.load(str(fpath), mmap_mode="r")
                n_seq = len(arr) // self.sequence_length
                if n_seq > 0:
                    self.single_arrays.append(arr)
                    self.single_num_seqs.append(n_seq)

    def _sample_from_rng(
        self, rng: np.random.Generator, device: torch.device
    ) -> CalibrationBatch:
        """Sample an equal-language (or single-corpus) training batch using the given RNG."""
        sequences: List[np.ndarray] = []
        language_slices: Dict[str, Tuple[int, int]] = {}

        if self.lang_arrays:
            langs = list(self.lang_arrays.keys())
            num_langs = len(langs)
            base_per_lang = self.batch_size // num_langs
            remainder = self.batch_size % num_langs
            offset = 0
            for idx, lang in enumerate(langs):
                count = base_per_lang + (1 if idx < remainder else 0)
                if count <= 0:
                    continue
                arr = self.lang_arrays[lang]
                max_seq = self.lang_num_seqs[lang]
                seq_indices = rng.integers(0, max_seq, size=count)
                for s_idx in seq_indices:
                    start = int(s_idx) * self.sequence_length
                    end = start + self.sequence_length
                    sequences.append(np.asarray(arr[start:end], dtype=np.int64))
                language_slices[lang] = (offset, offset + count)
                offset += count
        else:
            for _ in range(self.batch_size):
                shard_idx = int(rng.integers(0, len(self.single_arrays)))
                arr = self.single_arrays[shard_idx]
                max_seq = self.single_num_seqs[shard_idx]
                s_idx = int(rng.integers(0, max_seq))
                start = s_idx * self.sequence_length
                end = start + self.sequence_length
                sequences.append(np.asarray(arr[start:end], dtype=np.int64))

        batch_np = np.stack(sequences, axis=0)
        input_ids = torch.from_numpy(batch_np).to(
            device=device, dtype=torch.long, non_blocking=True
        )
        return CalibrationBatch(input_ids=input_ids, language_slices=language_slices)

    def next_batch(self, device: torch.device) -> CalibrationBatch:
        """Draw calibration batch A from the training shards (used for gradient/GGN/QP)."""
        batch = self._sample_from_rng(self.rng, device=device)
        self.cumulative_tokens += int(batch.input_ids.numel())
        self.cumulative_batches += 1
        return batch

    def next_eval_batch(self, device: torch.device) -> CalibrationBatch:
        """
        Draw independent batch B from the training calibration distribution using a
        separate RNG stream so batch A and s* are never perturbed.
        """
        batch = self._sample_from_rng(self.eval_rng, device=device)
        self.cumulative_eval_tokens += int(batch.input_ids.numel())
        self.cumulative_eval_batches += 1
        return batch

    def state_dict(self) -> Dict[str, Any]:
        """Serialize RNG state and cumulative counters for checkpoint reproducibility."""
        return {
            "rng_state": copy.deepcopy(self.rng.bit_generator.state),
            "eval_rng_state": copy.deepcopy(self.eval_rng.bit_generator.state),
            "cumulative_tokens": int(self.cumulative_tokens),
            "cumulative_batches": int(self.cumulative_batches),
            "cumulative_eval_tokens": int(self.cumulative_eval_tokens),
            "cumulative_eval_batches": int(self.cumulative_eval_batches),
            "seed": int(self.seed),
        }

    def load_state_dict(self, state: Dict[str, Any]) -> None:
        """Restore RNG state and cumulative counters from a checkpoint."""
        if not isinstance(state, dict):
            return
        if "rng_state" in state and state["rng_state"] is not None:
            self.rng.bit_generator.state = copy.deepcopy(state["rng_state"])
        if "eval_rng_state" in state and state["eval_rng_state"] is not None:
            self.eval_rng.bit_generator.state = copy.deepcopy(state["eval_rng_state"])
        if "cumulative_tokens" in state and state["cumulative_tokens"] is not None:
            self.cumulative_tokens = int(state["cumulative_tokens"])
        if "cumulative_batches" in state and state["cumulative_batches"] is not None:
            self.cumulative_batches = int(state["cumulative_batches"])
        if (
            "cumulative_eval_tokens" in state
            and state["cumulative_eval_tokens"] is not None
        ):
            self.cumulative_eval_tokens = int(state["cumulative_eval_tokens"])
        if (
            "cumulative_eval_batches" in state
            and state["cumulative_eval_batches"] is not None
        ):
            self.cumulative_eval_batches = int(state["cumulative_eval_batches"])



def compute_nesterov_scaling_and_lookahead(
    named_params: List[Tuple[str, nn.Parameter]],
    momentum_buffers: List[Optional[torch.Tensor]],
    grad_rest: Dict[int, torch.Tensor],
    lr: float,
    momentum: float,
    nesterov: bool,
) -> Tuple[float, Dict[str, torch.Tensor]]:
    """
    Construct the Nesterov look-ahead parameter dictionary hat_theta:
        a = lr * (1 + momentum)     (when nesterov and momentum > 0)
        hat_theta = theta_t - lr * momentum^2 * B_{t-1} - a * G_rest
    where G_rest contains non-eligible pseudo-gradient tensors (and 0 for eligible 2D tensors).

    For vanilla momentum (nesterov=False, momentum > 0):
        a = lr,   hat_theta = theta_t - lr * momentum * B_{t-1} - a * G_rest.
    For no momentum (momentum == 0):
        a = lr,   hat_theta = theta_t - a * G_rest.
    """
    lr_f = float(lr)
    mu_f = float(momentum) if momentum > 0.0 else 0.0

    if nesterov and mu_f > 0.0:
        a_scale = lr_f * (1.0 + mu_f)
        mom_coeff = lr_f * (mu_f * mu_f)
    elif mu_f > 0.0:
        a_scale = lr_f
        mom_coeff = lr_f * mu_f
    else:
        a_scale = lr_f
        mom_coeff = 0.0

    theta_hat: Dict[str, torch.Tensor] = {}
    for p_idx, (name, param) in enumerate(named_params):
        p_val = param.detach().to(dtype=torch.float32)
        if mom_coeff != 0.0 and p_idx < len(momentum_buffers):
            buf = momentum_buffers[p_idx]
            if isinstance(buf, torch.Tensor):
                p_val = p_val - mom_coeff * buf.detach().to(dtype=torch.float32)
        if p_idx in grad_rest:
            g_r = grad_rest[p_idx].detach().to(dtype=torch.float32)
            p_val = p_val - a_scale * g_r
        theta_hat[name] = p_val

    return a_scale, theta_hat


def compute_decoupled_nesterov_scaling_and_lookahead(
    named_params: List[Tuple[str, nn.Parameter]],
    momentum_buffers: List[Optional[torch.Tensor]],
    g_tilde_ones: Dict[int, torch.Tensor],
    lr: float,
    momentum: float,
    nesterov: bool,
) -> Tuple[float, Dict[str, torch.Tensor]]:
    """
    Construct the decoupled Nesterov look-ahead parameter dictionary hat_theta:
        a = lr
        hat_theta = theta_t - lr * momentum^2 * B_{t-1} - lr * momentum * tilde_G(1)
    where tilde_G(1) is the original IsoLoCo spectral transformation on eligible 2D
    tensors and unchanged averaged pseudo-gradients on non-eligible tensors.

    Non-eligible tensors are included once in hat_theta via tilde_G(1) and excluded
    from the allocation-dependent term -a * sum_b s_b D_b.
    """
    lr_f = float(lr)
    mu_f = float(momentum) if momentum > 0.0 else 0.0
    a_scale = lr_f

    if nesterov and mu_f > 0.0:
        mom_buf_coeff = lr_f * (mu_f * mu_f)
        g_ones_coeff = lr_f * mu_f
    elif mu_f > 0.0:
        mom_buf_coeff = lr_f * mu_f
        g_ones_coeff = 0.0
    else:
        mom_buf_coeff = 0.0
        g_ones_coeff = 0.0

    theta_hat: Dict[str, torch.Tensor] = {}
    for p_idx, (name, param) in enumerate(named_params):
        p_val = param.detach().to(dtype=torch.float32)
        if mom_buf_coeff != 0.0 and p_idx < len(momentum_buffers):
            buf = momentum_buffers[p_idx]
            if isinstance(buf, torch.Tensor):
                p_val = p_val - mom_buf_coeff * buf.detach().to(dtype=torch.float32)
        if g_ones_coeff != 0.0 and p_idx in g_tilde_ones:
            g_iso = g_tilde_ones[p_idx].detach().to(dtype=torch.float32)
            p_val = p_val - g_ones_coeff * g_iso
        theta_hat[name] = p_val

    return a_scale, theta_hat



@torch.no_grad()
def evaluate_calibration_loss_at_params(
    model: nn.Module,
    params_fp32: Dict[str, torch.Tensor],
    calib_batch: CalibrationBatch,
    pad_token_id: Optional[int] = None,
    micro_batch_size: int = 4,
) -> Tuple[float, Dict[str, float]]:
    """
    Evaluate the repository's shifted language-model cross-entropy loss at a given
    parameter dictionary without modifying model state or gradients.
    """
    base_model = unwrap_model(model)
    was_training = base_model.training
    base_model.eval()

    try:
        param_dtypes = {name: p.dtype for name, p in base_model.named_parameters()}
        call_params = {
            name: val.to(dtype=param_dtypes.get(name, val.dtype))
            for name, val in params_fp32.items()
        }

        input_ids = calib_batch.input_ids
        labels = input_ids.clone()
        if pad_token_id is not None:
            labels[labels == pad_token_id] = -100

        n_seqs = input_ids.shape[0]
        mb_size = max(1, min(int(micro_batch_size), n_seqs))

        total_nll = 0.0
        total_valid = 0

        for start in range(0, n_seqs, mb_size):
            end = min(n_seqs, start + mb_size)
            mb_ids = input_ids[start:end]
            mb_labels = labels[start:end]

            with torch.amp.autocast(device_type=input_ids.device.type, enabled=False):
                out = functional_call(
                    base_model,
                    call_params,
                    args=(),
                    kwargs={"input_ids": mb_ids},
                )
                logits = out.logits if hasattr(out, "logits") else out[0]
                shift_logits = logits[..., :-1, :].contiguous().to(dtype=torch.float32)
                shift_labels = mb_labels[..., 1:].contiguous()
                valid_count = int((shift_labels != -100).sum().item())
                if valid_count > 0:
                    nll = F.cross_entropy(
                        shift_logits.view(-1, shift_logits.size(-1)),
                        shift_labels.view(-1),
                        ignore_index=-100,
                        reduction="sum",
                    )
                    total_nll += float(nll.item())
                    total_valid += valid_count

        global_loss = total_nll / float(total_valid) if total_valid > 0 else 0.0

        per_lang_loss: Dict[str, float] = {}
        for lang, (l_start, l_end) in calib_batch.language_slices.items():
            if l_start >= l_end or l_start >= n_seqs:
                continue
            lang_ids = input_ids[l_start : min(l_end, n_seqs)]
            lang_labels = labels[l_start : min(l_end, n_seqs)]
            with torch.amp.autocast(device_type=input_ids.device.type, enabled=False):
                out_l = functional_call(
                    base_model,
                    call_params,
                    args=(),
                    kwargs={"input_ids": lang_ids},
                )
                logits_l = out_l.logits if hasattr(out_l, "logits") else out_l[0]
                shift_logits_l = logits_l[..., :-1, :].contiguous().to(dtype=torch.float32)
                shift_labels_l = lang_labels[..., 1:].contiguous()
                valid_l = int((shift_labels_l != -100).sum().item())
                if valid_l > 0:
                    nll_l = F.cross_entropy(
                        shift_logits_l.view(-1, shift_logits_l.size(-1)),
                        shift_labels_l.view(-1),
                        ignore_index=-100,
                        reduction="sum",
                    )
                    per_lang_loss[lang] = float(nll_l.item()) / float(valid_l)

        return global_loss, per_lang_loss
    finally:
        if was_training:
            base_model.train()


@torch.no_grad()
def compute_calibration_gradient_and_ggn(
    model: nn.Module,
    theta_hat_fp32: Dict[str, torch.Tensor],
    bin_directions: List[Dict[str, torch.Tensor]],
    active_bins_mask: np.ndarray,
    calib_batch: CalibrationBatch,
    pad_token_id: Optional[int] = None,
    micro_batch_size: int = 2,
    token_chunk_size: int = 128,
) -> CurvatureCalibrationStats:
    """
    Compute the calibration directional gradient g_b = <grad L_cal(hat_theta), D_b>
    and GGN matrix K_{bc} = <D_b, GGN(hat_theta) D_c> using torch.func.jvp
    with token chunking and float64 accumulation.

    Matches the repository's shifted cross-entropy loss normalization over valid
    tokens (shift_labels != -100) and preserves model parameters, optimizer state,
    and training mode.
    """
    t0 = time.perf_counter()
    base_model = unwrap_model(model)
    was_training = base_model.training
    base_model.eval()

    num_bins = len(bin_directions)
    active_indices = [
        b for b in range(num_bins) if b < len(active_bins_mask) and bool(active_bins_mask[b])
    ]
    num_active = len(active_indices)

    g_accum = torch.zeros(num_bins, dtype=torch.float64, device="cpu")
    K_accum = torch.zeros((num_bins, num_bins), dtype=torch.float64, device="cpu")

    try:
        input_ids = calib_batch.input_ids
        device = input_ids.device
        labels = input_ids.clone()
        if pad_token_id is not None:
            labels[labels == pad_token_id] = -100

        n_seqs = input_ids.shape[0]
        seq_len = int(input_ids.shape[1]) if input_ids.dim() >= 2 else 1
        shifted_seq_len = max(1, seq_len - 1)
        t_chunk = max(1, int(token_chunk_size))
        max_seqs_for_token_budget = max(1, t_chunk // shifted_seq_len)
        mb_size = max(1, min(int(micro_batch_size), n_seqs, max_seqs_for_token_budget))

        primal_params: Dict[str, torch.Tensor] = {
            name: val.detach().to(device=device, dtype=torch.float32)
            for name, val in theta_hat_fp32.items()
        }
        zero_tangents: Dict[str, torch.Tensor] = {
            name: torch.zeros_like(val) for name, val in primal_params.items()
        }

        active_tangents: List[Dict[str, torch.Tensor]] = []
        for b in active_indices:
            dir_b = bin_directions[b]
            tan_b = {}
            for name, z_zero in zero_tangents.items():
                if name in dir_b:
                    tan_b[name] = dir_b[name].detach().to(
                        device=device, dtype=torch.float32
                    )
                else:
                    tan_b[name] = z_zero
            active_tangents.append(tan_b)

        total_nll = 0.0
        total_valid_tokens = 0
        g_act_gpu = torch.zeros(num_active, dtype=torch.float64, device=device)
        K_act_gpu = torch.zeros(
            (num_active, num_active), dtype=torch.float64, device=device
        )

        for start in range(0, n_seqs, mb_size):
            end = min(n_seqs, start + mb_size)
            mb_ids = input_ids[start:end]
            mb_labels = labels[start:end]
            shift_labels = mb_labels[..., 1:].contiguous().view(-1)
            valid_mask = shift_labels != -100
            valid_indices = torch.nonzero(valid_mask, as_tuple=False).squeeze(-1)
            num_valid_mb = int(valid_indices.numel())
            if num_valid_mb == 0:
                continue

            targets_valid = shift_labels.index_select(0, valid_indices)
            total_valid_tokens += num_valid_mb

            with torch.amp.autocast(
                device_type=device.type, enabled=False
            ), _sdpa_math_context():
                if num_active == 0:
                    out = functional_call(
                        base_model,
                        primal_params,
                        args=(),
                        kwargs={"input_ids": mb_ids},
                    )
                    logits = out.logits if hasattr(out, "logits") else out[0]
                    shift_logits = logits[..., :-1, :].contiguous().to(dtype=torch.float32)
                    logits_valid = shift_logits.view(
                        -1, shift_logits.size(-1)
                    ).index_select(0, valid_indices)
                    nll_mb = F.cross_entropy(
                        logits_valid,
                        targets_valid,
                        ignore_index=-100,
                        reduction="sum",
                    )
                    total_nll += float(nll_mb.item())
                else:
                    for c_start in range(0, num_valid_mb, t_chunk):
                        c_end = min(num_valid_mb, c_start + t_chunk)
                        chunk_valid_indices = valid_indices[c_start:c_end]
                        targets_c = targets_valid[c_start:c_end]

                        def _forward_chunk_logits(
                            params_dict: Dict[str, torch.Tensor],
                            _idx: torch.Tensor = chunk_valid_indices,
                        ) -> torch.Tensor:
                            out = functional_call(
                                base_model,
                                params_dict,
                                args=(),
                                kwargs={"input_ids": mb_ids},
                            )
                            logits = out.logits if hasattr(out, "logits") else out[0]
                            shift_logits = (
                                logits[..., :-1, :].contiguous().to(dtype=torch.float32)
                            )
                            return shift_logits.view(
                                -1, shift_logits.size(-1)
                            ).index_select(0, _idx)

                        logits_c_fp32: Optional[torch.Tensor] = None
                        z_chunk_list: List[torch.Tensor] = []
                        for a_idx in range(num_active):
                            primal_c, tangent_c = jvp(
                                _forward_chunk_logits,
                                (primal_params,),
                                (active_tangents[a_idx],),
                            )
                            if logits_c_fp32 is None:
                                logits_c_fp32 = primal_c
                            z_chunk_list.append(tangent_c)

                        assert logits_c_fp32 is not None
                        nll_c = F.cross_entropy(
                            logits_c_fp32,
                            targets_c,
                            ignore_index=-100,
                            reduction="sum",
                        )
                        total_nll += float(nll_c.item())

                        logits_c = logits_c_fp32.to(dtype=torch.float64)
                        p_c = torch.softmax(logits_c, dim=-1)

                        z_c = torch.stack(
                            [z_b.to(dtype=torch.float64) for z_b in z_chunk_list],
                            dim=0,
                        )
                        del z_chunk_list

                        mean_z = torch.einsum("tv,btv->bt", p_c, z_c)

                        target_idx = targets_c.view(1, -1, 1).expand(
                            num_active, c_end - c_start, 1
                        )
                        z_target = torch.gather(z_c, dim=2, index=target_idx).squeeze(-1)
                        g_act_gpu += torch.sum(mean_z - z_target, dim=1)

                        pz_c = p_c.unsqueeze(0) * z_c
                        second_moment = torch.einsum("btv,ctv->bc", pz_c, z_c)
                        first_moment_outer = torch.einsum("bt,ct->bc", mean_z, mean_z)
                        K_act_gpu += second_moment - first_moment_outer
                        del z_c

        if total_valid_tokens > 0 and num_active > 0:
            inv_T = 1.0 / float(total_valid_tokens)
            g_act_cpu = (g_act_gpu * inv_T).detach().cpu()
            K_act_cpu = (K_act_gpu * inv_T).detach().cpu()
            K_act_cpu = 0.5 * (K_act_cpu + K_act_cpu.T)

            for a_i, b_i in enumerate(active_indices):
                g_accum[b_i] = g_act_cpu[a_i]
                for a_j, b_j in enumerate(active_indices):
                    K_accum[b_i, b_j] = K_act_cpu[a_i, a_j]

        global_loss = (
            total_nll / float(total_valid_tokens) if total_valid_tokens > 0 else 0.0
        )

        per_lang_loss: Dict[str, float] = {}
        if calib_batch.language_slices:
            _, per_lang_loss = evaluate_calibration_loss_at_params(
                model=base_model,
                params_fp32=primal_params,
                calib_batch=calib_batch,
                pad_token_id=pad_token_id,
                micro_batch_size=mb_size,
            )

        return CurvatureCalibrationStats(
            g=g_accum.numpy(),
            K=K_accum.numpy(),
            loss_lookahead=global_loss,
            per_language_loss=per_lang_loss,
            valid_tokens=total_valid_tokens,
            total_tokens=int(input_ids.numel()),
            curvature_time_sec=time.perf_counter() - t0,
        )
    finally:
        if was_training:
            base_model.train()



