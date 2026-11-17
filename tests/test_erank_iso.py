"""CPU-only mathematical correctness, optimizer integration, and checkpoint tests for Effective-Rank Tikhonov IsoLoCo (methods/erank_iso.py)."""

import copy
import math
import os
import sys
import unittest

import torch
import torch.nn as nn

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from methods import (
    ErankIsoLoCo,
    ISOCLoCo,
    apply_erank_iso,
    apply_strategy_preset,
    build_outer_optimizer,
    compute_shannon_effective_rank,
    erank_iso_transform,
    method_handles_own_communication,
)
from training.config import load_experiment_config


def explicit_reference_erank_iso(
    G: torch.Tensor,
) -> tuple[torch.Tensor, float, int, float, torch.Tensor, torch.Tensor]:
    """
    Separate step-by-step reference implementation of Effective-Rank Tikhonov IsoLoCo:
      p_r = sigma_r / sum_j sigma_j
      r_eff = exp(-sum_r p_r log p_r)
      tau = sigma_{ceil(r_eff)}  (1-based index)
      psi_r = sigma_r / sqrt(sigma_r^2 + tau^2)
      sigma_tilde_r = psi_r * (sqrt(R) * sigma_bar) / ||psi||_2
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
    tau = s_list[ceil_idx - 1]

    s_max = s_list[0]
    psi_list = [
        (x / s_max) / math.sqrt((x / s_max) ** 2 + (tau / s_max) ** 2) for x in s_list
    ]
    psi_norm = math.sqrt(sum(z * z for z in psi_list))
    sigma_bar = s_sum / float(R)
    f_iso = math.sqrt(float(R)) * sigma_bar
    s_tilde_list = [z * (f_iso / psi_norm) for z in psi_list]

    S_tilde = torch.tensor(s_tilde_list, dtype=S.dtype, device=G.device)
    G_tilde = U @ torch.diag(S_tilde) @ Vh
    return (
        G_tilde.to(dtype=orig_dtype),
        r_eff,
        ceil_idx,
        tau,
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


class TestErankIso(unittest.TestCase):
    """CPU-only verification tests for erank_iso."""

    def test_1_agreement_with_reference_and_frobenius_normalization(self):
        torch.manual_seed(42)
        for shape in ((32, 32), (64, 16), (16, 64), (17, 13), (1, 8), (8, 1)):
            for dtype in (torch.float64, torch.float32, torch.bfloat16):
                G = torch.randn(shape, dtype=dtype) * 0.4
                actual, stats = erank_iso_transform(G, return_stats=True)
                ref_G, ref_reff, ref_ceil, ref_tau, ref_sb, ref_sa = explicit_reference_erank_iso(G)

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
                if dtype in (torch.float64, torch.float32):
                    self.assertLess(stats["rel_frob_match_error"], 1e-6)

    def test_2_shannon_effective_rank_and_transition_scale(self):
        # Uniform spectrum of length 8 -> r_eff = 8.0, ceil_idx = 8, tau = sigma_8
        s_unif = torch.full((8,), 2.5, dtype=torch.float64)
        r_eff, ceil_idx, tau = compute_shannon_effective_rank(s_unif)
        self.assertAlmostEqual(r_eff, 8.0, places=12)
        self.assertEqual(ceil_idx, 8)
        self.assertAlmostEqual(tau, 2.5, places=12)

        # Rank-3 flat spectrum in R=8 -> r_eff = 3.0, ceil_idx = 3, tau = sigma_3 > 0 (not sigma_4 = 0)
        s_rank3 = torch.tensor([3.0, 3.0, 3.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=torch.float64)
        r_eff3, ceil_idx3, tau3 = compute_shannon_effective_rank(s_rank3)
        self.assertAlmostEqual(r_eff3, 3.0, places=12)
        self.assertEqual(ceil_idx3, 3)
        self.assertAlmostEqual(tau3, 3.0, places=12)

        # Rank-1 spectrum -> r_eff = 1.0, ceil_idx = 1, tau = sigma_1
        s_rank1 = torch.tensor([5.0, 0.0, 0.0, 0.0], dtype=torch.float64)
        r_eff1, ceil_idx1, tau1 = compute_shannon_effective_rank(s_rank1)
        self.assertAlmostEqual(r_eff1, 1.0, places=12)
        self.assertEqual(ceil_idx1, 1)
        self.assertAlmostEqual(tau1, 5.0, places=12)

        # Non-integer effective rank -> ceil_idx = ceil(r_eff)
        s_custom = torch.tensor([4.0, 2.0, 1.0, 1.0], dtype=torch.float64)
        # p = [0.5, 0.25, 0.125, 0.125], H = 0.5*ln2 + 0.25*ln4 + 0.25*ln8 = 1.75 * ln2 -> r_eff = 2^1.75 = 3.36358566...
        r_eff_c, ceil_idx_c, tau_c = compute_shannon_effective_rank(s_custom)
        self.assertAlmostEqual(r_eff_c, 2.0**1.75, places=12)
        self.assertEqual(ceil_idx_c, 4)
        self.assertAlmostEqual(tau_c, 1.0, places=12)

    def test_3_zero_and_rank_deficient_matrices(self):
        for dtype in (torch.float32, torch.float64, torch.bfloat16):
            z = torch.zeros(12, 8, dtype=dtype)
            out_z, stats_z = erank_iso_transform(z, return_stats=True)
            self.assertEqual(out_z.dtype, dtype)
            self.assertEqual(out_z.shape, (12, 8))
            self.assertFalse(torch.isnan(out_z).any())
            self.assertFalse(torch.isinf(out_z).any())
            self.assertEqual(int(torch.count_nonzero(out_z).item()), 0)
            self.assertEqual(stats_z["orig_frob_norm"], 0.0)
            self.assertEqual(stats_z["trans_frob_norm"], 0.0)

        # Rank-deficient matrix with exact zero rows/cols
        mat_def = torch.zeros(10, 6, dtype=torch.float32)
        mat_def[0, 0] = 3.0
        mat_def[1, 1] = 3.0
        out_def, stats_def = erank_iso_transform(mat_def, return_stats=True)
        self.assertFalse(torch.isnan(out_def).any())
        self.assertFalse(torch.isinf(out_def).any())
        self.assertAlmostEqual(stats_def["r_eff"], 2.0, places=6)
        self.assertEqual(stats_def["ceil_idx"], 2)
        self.assertAlmostEqual(stats_def["tau"], 3.0, places=6)
        self.assertLess(stats_def["rel_frob_match_error"], 1e-6)

    def test_4_numerical_scaling_equivariance(self):
        torch.manual_seed(314)
        G_base = torch.randn(16, 12, dtype=torch.float32)
        out_base, stats_base = erank_iso_transform(G_base, return_stats=True)

        for alpha in (1e-20, 1e-10, 1e-7, 2.0**-15, 2.0**10, 1e5):
            G_scaled = G_base * alpha
            out_scaled, stats_scaled = erank_iso_transform(G_scaled, return_stats=True)
            self.assertFalse(torch.isnan(out_scaled).any())
            self.assertFalse(torch.isinf(out_scaled).any())
            self.assertAlmostEqual(stats_scaled["r_eff"], stats_base["r_eff"], places=4)
            self.assertEqual(stats_scaled["ceil_idx"], stats_base["ceil_idx"])
            self.assertAlmostEqual(
                stats_scaled["tau"] / alpha, stats_base["tau"], places=4
            )
            self.assertTrue(
                torch.allclose(out_scaled, out_base * alpha, atol=1e-5 * alpha, rtol=1e-5)
            )

    def test_5_outer_nesterov_exclusions_and_diagnostics(self):
        torch.manual_seed(555)
        K = 4
        lr = 1.5
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

        opt = ErankIsoLoCo(
            [m.parameters() for m in models],
            lr=lr,
            momentum=mu,
            nesterov=True,
            exclude_keys=exclude_keys,
            virtual_workers=K,
        )

        for step_idx in range(3):
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
                    merged_expected, _, _, _, _, _ = explicit_reference_erank_iso(g_mean)

                ref_buf.mul_(mu).add_(merged_expected)
                upd_expected = merged_expected.add(ref_buf, alpha=mu)
                ref_p.add_(upd_expected, alpha=-lr)

                self.assertTrue(torch.allclose(opt.momentum_buffers[p_idx], ref_buf, atol=1e-6))
                for v_idx, m in enumerate(models):
                    p_actual = list(m.parameters())[p_idx]
                    self.assertTrue(torch.allclose(p_actual.grad, upd_expected, atol=1e-6))
                    self.assertTrue(torch.allclose(p_actual.data, ref_p, atol=1e-6))

        diag = opt.get_diagnostics()
        for key in (
            "erank_iso_r_eff_mean",
            "erank_iso_r_eff_attn_mean",
            "erank_iso_r_eff_mlp_mean",
            "erank_iso_tau_mean",
            "erank_iso_original_frob_norm",
            "erank_iso_iso_target_frob_norm",
            "erank_iso_transformed_frob_norm",
            "erank_iso_relative_frob_match_error",
            "erank_iso_sv_sum_before",
            "erank_iso_sv_sum_after",
            "erank_iso_nuclear_norm_retained_ratio",
            "erank_iso_transformed_to_original_frob_ratio",
            "erank_iso_correction_time_sec",
        ):
            self.assertIn(key, diag)
        self.assertLess(diag["erank_iso_relative_frob_match_error"], 1e-6)
        self.assertGreater(diag["erank_iso_r_eff_mean"], 1.0)
        self.assertGreater(diag["erank_iso_tau_mean"], 0.0)

    def test_6_registry_and_config_matches_isoloco_50r(self):
        cfg_iso = load_experiment_config(["--config", "configs/isoloco_iid_50r.yaml"])
        apply_strategy_preset(cfg_iso)
        cfg_erank = load_experiment_config(["--config", "configs/erank_iso_iid_50r.yaml"])
        apply_strategy_preset(cfg_erank)

        self.assertTrue(method_handles_own_communication(cfg_erank.outer_optimizer))
        models = [DummyModel() for _ in range(4)]
        opt = build_outer_optimizer(models, cfg_erank, virtual_workers=4)
        self.assertIsInstance(opt, ErankIsoLoCo)
        self.assertEqual(opt.lr, 1.5)
        self.assertEqual(opt.momentum, 0.8)
        self.assertTrue(opt.nesterov)
        self.assertEqual(opt.exclude_keys, ["embed_tokens", "wpe", "lm_head"])

        expected_diff_keys = {"config", "run_name", "save_path", "method", "strategy", "outer_optimizer"}
        all_keys = set(vars(cfg_iso).keys()) | set(vars(cfg_erank).keys())
        for k in sorted(all_keys):
            if k not in expected_diff_keys:
                self.assertEqual(getattr(cfg_iso, k), getattr(cfg_erank, k), msg=k)
        self.assertNotEqual(cfg_erank.save_path, cfg_iso.save_path)

    def test_7_checkpoint_state_dict_and_resume(self):
        torch.manual_seed(777)
        K = 4
        base_model = DummyModel()
        models_u = [copy.deepcopy(base_model) for _ in range(K)]
        for m in models_u:
            for name, p in m.named_parameters():
                p.param_name = name
        opt_u = ErankIsoLoCo([m.parameters() for m in models_u], lr=1.5, momentum=0.8, nesterov=True, virtual_workers=K)

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
        opt_r = ErankIsoLoCo([m.parameters() for m in models_r], lr=1.5, momentum=0.8, nesterov=True, virtual_workers=K)
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



