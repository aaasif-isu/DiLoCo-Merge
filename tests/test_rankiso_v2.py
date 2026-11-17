"""CPU-only mathematical correctness, optimizer integration, config, and checkpoint tests for Spectral-Ceiling L1 Effective-Rank Tikhonov IsoLoCo (methods/rankiso_v2.py)."""

import copy
import math
import os
import sys
import unittest
from pathlib import Path

import torch
import torch.nn as nn
import yaml

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from methods import (
    RankIsoV2,
    RankIsoV2LoCo,
    apply_rankiso_v2,
    apply_strategy_preset,
    build_outer_optimizer,
    compute_shannon_effective_rank,
    method_handles_own_communication,
    rankiso_v2_transform,
)
from training.config import load_experiment_config


def explicit_reference_rankiso_v2(
    G: torch.Tensor,
) -> tuple[torch.Tensor, float, int, float, torch.Tensor, torch.Tensor]:
    """
    Separate step-by-step reference implementation of Spectral-Ceiling L1 RankLoCo (rankiso_v2):
      p_r = sigma_r / sum_j sigma_j
      r_eff = exp(-sum_r p_r log p_r)
      tau_1 = sigma_{ceil(r_eff)}  (1-based index)
      psi_i = sigma_i / sqrt(sigma_i^2 + tau_1^2)
      sigma_tilde_i = sigma_bar * (psi_i / psi_1)
    """
    orig_dtype = G.dtype
    work_dtype = orig_dtype if orig_dtype in (torch.float32, torch.float64) else torch.float32
    G_work = G.detach().to(dtype=work_dtype)
    m, n = G_work.shape
    R = min(m, n)

    if not torch.any(G_work != 0):
        zeros = torch.zeros(R, dtype=torch.float32)
        return torch.zeros_like(G), 0.0, 0, 0.0, zeros, zeros.clone()

    U, S, Vh = torch.linalg.svd(G_work, full_matrices=False)
    s_list = [float(x) for x in S.to(dtype=torch.float64).tolist()]
    s_sum = sum(s_list)
    k_pos = sum(1 for x in s_list if x > 0.0)
    p_list = [x / s_sum for x in s_list]
    entropy = -sum(p * math.log(p) for p in p_list if p > 0.0)
    r_eff = min(float(k_pos), max(1.0, math.exp(entropy)))
    if abs(r_eff - round(r_eff)) <= 1e-12:
        ceil_idx = int(round(r_eff))
    else:
        ceil_idx = int(math.ceil(r_eff))
    ceil_idx = min(k_pos, max(1, ceil_idx))
    tau_1 = s_list[ceil_idx - 1]

    s_max = s_list[0]
    psi_list = [
        (x / s_max) / math.sqrt((x / s_max) ** 2 + (tau_1 / s_max) ** 2) for x in s_list
    ]
    psi_1 = psi_list[0]
    sigma_bar = s_sum / float(R)
    s_tilde_list = [sigma_bar * (z / psi_1) for z in psi_list]

    S_tilde = torch.tensor(s_tilde_list, dtype=S.dtype, device=G.device)
    G_tilde = U @ torch.diag(S_tilde) @ Vh
    return (
        G_tilde.to(dtype=orig_dtype),
        r_eff,
        ceil_idx,
        tau_1,
        S.detach().cpu().to(dtype=torch.float32),
        S_tilde.detach().cpu().to(dtype=torch.float32),
    )


class DummyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.embed_tokens = nn.Embedding(32, 16)
        self.wpe = nn.Embedding(16, 16)
        self.q_proj = nn.Linear(16, 16, bias=True)
        self.c_fc = nn.Linear(16, 32, bias=False)
        self.ln = nn.LayerNorm(16)
        self.lm_head = nn.Linear(16, 32, bias=False)

        for name, p in self.named_parameters():
            p.param_name = name



class TestRankIsoV2(unittest.TestCase):
    """CPU-only verification tests for rankiso_v2 (Spectral-Ceiling L1)."""

    def test_1_agreement_with_reference_and_spectral_ceiling_invariants(self):
        torch.manual_seed(42)
        for shape in ((32, 32), (64, 16), (16, 64), (17, 13), (1, 8), (8, 1)):
            for dtype in (torch.float64, torch.float32, torch.bfloat16):
                G = torch.randn(shape, dtype=dtype) * 0.4
                actual, stats = rankiso_v2_transform(G, return_stats=True)
                ref_G, ref_reff, ref_ceil, ref_tau, ref_sb, ref_sa = explicit_reference_rankiso_v2(G)

                self.assertEqual(actual.shape, G.shape)
                self.assertEqual(actual.dtype, G.dtype)
                self.assertFalse(torch.isnan(actual).any())
                self.assertFalse(torch.isinf(actual).any())

                self.assertAlmostEqual(stats["r_eff"], ref_reff, places=9)
                self.assertEqual(stats["ceil_idx"], ref_ceil)
                self.assertAlmostEqual(stats["tau"], ref_tau, places=7)
                self.assertTrue(torch.allclose(stats["s_before"], ref_sb, atol=1e-6, rtol=1e-6))
                self.assertTrue(torch.allclose(stats["s_after"], ref_sa, atol=1e-6, rtol=1e-6))

                tol = 1e-12 if dtype == torch.float64 else (1e-2 if dtype == torch.bfloat16 else 1e-5)
                self.assertTrue(torch.allclose(actual, ref_G, atol=tol, rtol=tol))

                # Spectral-Ceiling L1 invariant: sigma_tilde_1 == sigma_bar and sigma_tilde_i <= sigma_bar
                sigma_bar = float(ref_sb.double().mean().item())
                s_after = stats["s_after"].double()
                self.assertAlmostEqual(float(s_after[0].item()), sigma_bar, places=6)
                self.assertTrue(torch.all(s_after <= sigma_bar + 1e-6))
                # Monotonicity: descending singular values preserved
                if s_after.numel() > 1:
                    self.assertTrue(torch.all(s_after[:-1] >= s_after[1:] - 1e-7))
                # Frobenius norm <= IsoLoCo target Frobenius norm sqrt(R) * sigma_bar
                self.assertLessEqual(stats["trans_frob_norm"], stats["iso_target_frob_norm"] + 1e-5)

    def test_2_shannon_effective_rank_and_uniform_spectrum_identity(self):
        # Uniform spectrum: psi_i == psi_1 for all i -> sigma_tilde_i == sigma_bar for all i
        s_unif = torch.full((8,), 2.5, dtype=torch.float64)
        r_eff, ceil_idx, tau = compute_shannon_effective_rank(s_unif)
        self.assertAlmostEqual(r_eff, 8.0, places=12)
        self.assertEqual(ceil_idx, 8)
        self.assertAlmostEqual(tau, 2.5, places=12)

        q, _ = torch.linalg.qr(torch.randn(8, 8, dtype=torch.float64))
        v, _ = torch.linalg.qr(torch.randn(8, 8, dtype=torch.float64))
        G_unif = q @ torch.diag(s_unif) @ v.T
        out_unif, st_unif = rankiso_v2_transform(G_unif, return_stats=True)
        self.assertTrue(torch.allclose(out_unif, G_unif, atol=1e-12, rtol=1e-12))
        self.assertAlmostEqual(st_unif["trans_frob_norm"], st_unif["iso_target_frob_norm"], places=10)

    def test_3_zero_rank_deficient_and_scale_homogeneity(self):
        for dtype in (torch.float32, torch.float64, torch.bfloat16):
            z = torch.zeros(12, 8, dtype=dtype)
            out_z, stats_z = rankiso_v2_transform(z, return_stats=True)
            self.assertEqual(out_z.dtype, dtype)
            self.assertEqual(out_z.shape, (12, 8))
            self.assertFalse(torch.isnan(out_z).any())
            self.assertEqual(float(out_z.abs().max().item()), 0.0)
            self.assertEqual(stats_z["r_eff"], 0.0)

        torch.manual_seed(99)
        G_base = torch.randn(24, 16, dtype=torch.float64)
        out_base, stats_base = rankiso_v2_transform(G_base, return_stats=True)
        for alpha in (1e-12, 1e-6, 0.25, 10.0, 1e6, 1e12):
            out_scaled, stats_scaled = rankiso_v2_transform(G_base * alpha, return_stats=True)
            self.assertAlmostEqual(stats_scaled["r_eff"], stats_base["r_eff"], places=9)
            self.assertEqual(stats_scaled["ceil_idx"], stats_base["ceil_idx"])
            self.assertAlmostEqual(stats_scaled["tau"] / alpha, stats_base["tau"], places=7)
            self.assertTrue(
                torch.allclose(out_scaled, out_base * alpha, atol=1e-9 * alpha, rtol=1e-9)
            )

    def test_4_outer_nesterov_exclusions_and_diagnostics(self):
        torch.manual_seed(555)
        K = 4
        lr = 1.8
        mu = 0.8
        exclude_keys = ["embed_tokens", "wpe", "lm_head"]

        base_model = DummyModel()
        models = [copy.deepcopy(base_model) for _ in range(K)]
        for m in models:
            for name, p in m.named_parameters():
                p.param_name = name

        ref_params = [p.data.clone() for p in base_model.parameters()]
        ref_names = [n for n, _ in base_model.named_parameters()]
        ref_bufs = [torch.zeros_like(p) for p in ref_params]

        opt = RankIsoV2LoCo(
            [m.parameters() for m in models],
            lr=lr,
            momentum=mu,
            nesterov=True,
            exclude_keys=exclude_keys,
            virtual_workers=K,
        )

        for _ in range(3):
            step_grads = [
                [torch.randn_like(p) * 0.1 for p in base_model.parameters()]
                for _ in range(K)
            ]
            for v_idx, m in enumerate(models):
                for p_idx, p in enumerate(m.parameters()):
                    p.grad = step_grads[v_idx][p_idx].clone()

            opt.step()

            for p_idx, (name, ref_p, ref_buf) in enumerate(zip(ref_names, ref_params, ref_bufs)):
                g_mean = sum(step_grads[v][p_idx] for v in range(K)) / float(K)
                is_excluded_or_1d = (ref_p.dim() != 2) or any(k in name for k in exclude_keys)
                if is_excluded_or_1d:
                    merged_expected = g_mean
                else:
                    merged_expected, _, _, _, _, _ = explicit_reference_rankiso_v2(g_mean)

                ref_buf.mul_(mu).add_(merged_expected)
                upd_expected = merged_expected.add(ref_buf, alpha=mu)
                ref_p.add_(upd_expected, alpha=-lr)

                self.assertTrue(torch.allclose(opt.momentum_buffers[p_idx], ref_buf, atol=1e-6))
                for v_idx, m in enumerate(models):
                    p_actual = list(m.parameters())[p_idx]
                    self.assertTrue(torch.allclose(p_actual.grad, upd_expected, atol=1e-6))
                    self.assertTrue(torch.allclose(p_actual.data, ref_p, atol=1e-6))

        diag = opt.get_diagnostics()
        self.assertGreater(diag["erank_iso_r_eff_mean"], 1.0)
        self.assertGreater(diag["erank_iso_tau_mean"], 0.0)
        self.assertLessEqual(
            diag["erank_iso_transformed_frob_norm"],
            diag["erank_iso_iso_target_frob_norm"] + 1e-5,
        )

    def test_5_pilot_178m_config_matches_earlier_pilot_and_builds_optimizer(self):
        cfg_v2 = load_experiment_config(["--config", "configs/pilot_178m_rankiso_v2_50r.yaml"])
        apply_strategy_preset(cfg_v2)

        self.assertTrue(method_handles_own_communication(cfg_v2.outer_optimizer))
        models = [DummyModel() for _ in range(4)]
        opt = build_outer_optimizer(models, cfg_v2, virtual_workers=4)
        self.assertIsInstance(opt, RankIsoV2LoCo)
        self.assertIsInstance(opt, RankIsoV2)
        self.assertEqual(opt.lr, 1.8)
        self.assertEqual(opt.momentum, 0.8)
        self.assertTrue(opt.nesterov)
        self.assertEqual(opt.exclude_keys, ["embed_tokens", "lm_head"])

        self.assertEqual(cfg_v2.num_workers, 4)
        self.assertEqual(cfg_v2.virtual_workers, 4)
        self.assertEqual(cfg_v2.inner_steps, 30)
        self.assertEqual(cfg_v2.batch_size, 2)
        self.assertEqual(cfg_v2.micro_batch_size, 2)
        self.assertEqual(cfg_v2.sequence_length, 2048)
        self.assertEqual(cfg_v2.seed, 42)
        self.assertEqual(cfg_v2.max_steps, 50)
        self.assertEqual(cfg_v2.token_budget, 24576000)
        self.assertEqual(
            cfg_v2.shards_path,
            "/lustre/hdd/LAS/jannesar-lab/aaasif/datasets/dclm_178m_pilot_24p6m",
        )

        prev_pilot_path = (
            Path(_REPO_ROOT)
            / "outputs"
            / "dclm_178m_erank_iso_iid_pilot_w4_h30_r50_s42"
            / "experiment_config.yaml"
        )
        if prev_pilot_path.exists():
            prev_cfg = yaml.safe_load(prev_pilot_path.read_text())
            expected_diff_keys = {
                "config",
                "run_name",
                "save_path",
                "method",
                "strategy",
                "outer_optimizer",
                "resolved_model_hparams",
                "world_size",
                "total_workers",
                "dataset_metadata",
                "resolved_partition_info",
                "pad_token_id",
            }
            for k, v in prev_cfg.items():
                if k not in expected_diff_keys and hasattr(cfg_v2, k):
                    self.assertEqual(getattr(cfg_v2, k), v, msg=f"Mismatch on key {k}")
            self.assertNotEqual(cfg_v2.save_path, prev_cfg["save_path"])

    def test_6_checkpoint_state_dict_and_resume(self):
        torch.manual_seed(777)
        K = 4
        base_model = DummyModel()
        models_u = [copy.deepcopy(base_model) for _ in range(K)]
        for m in models_u:
            for name, p in m.named_parameters():
                p.param_name = name
        opt_u = RankIsoV2LoCo(
            [m.parameters() for m in models_u],
            lr=1.8,
            momentum=0.8,
            nesterov=True,
            virtual_workers=K,
        )

        all_grads = [
            [[torch.randn_like(p) * 0.05 for p in base_model.parameters()] for _ in range(K)]
            for _ in range(4)
        ]

        for s in range(2):
            for v_idx, m in enumerate(models_u):
                for p_idx, p in enumerate(m.parameters()):
                    p.grad = all_grads[s][v_idx][p_idx].clone()
            opt_u.step()

        saved_opt_sd = copy.deepcopy(opt_u.state_dict())
        saved_model_sds = [copy.deepcopy(m.state_dict()) for m in models_u]

        for s in range(2, 4):
            for v_idx, m in enumerate(models_u):
                for p_idx, p in enumerate(m.parameters()):
                    p.grad = all_grads[s][v_idx][p_idx].clone()
            opt_u.step()

        models_r = [DummyModel() for _ in range(K)]
        for m, sd in zip(models_r, saved_model_sds):
            m.load_state_dict(sd)
        opt_r = RankIsoV2LoCo(
            [m.parameters() for m in models_r],
            lr=1.8,
            momentum=0.8,
            nesterov=True,
            virtual_workers=K,
        )
        opt_r.load_state_dict(saved_opt_sd)

        for s in range(2, 4):
            for v_idx, m in enumerate(models_r):
                for p_idx, p in enumerate(m.parameters()):
                    p.grad = all_grads[s][v_idx][p_idx].clone()
            opt_r.step()

        for b_u, b_r in zip(opt_u.momentum_buffers, opt_r.momentum_buffers):
            self.assertTrue(torch.equal(b_u, b_r))
        for m_u, m_r in zip(models_u, models_r):
            for p_u, p_r in zip(m_u.parameters(), m_r.parameters()):
                self.assertTrue(torch.equal(p_u.data, p_r.data))
                self.assertTrue(torch.equal(p_u.grad, p_r.grad))


if __name__ == "__main__":
    unittest.main()


