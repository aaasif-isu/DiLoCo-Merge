"""CPU-only unit tests for Joint Spectral Allocation (spectral_alloc)."""

import os
import sys
import unittest

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from methods.isocloco import ISOCLoCo
from methods.spectral_alloc import (
    SpectralAllocDecoupledLoCo,
    SpectralAllocLoCo,
    assign_rank_to_dyadic_bin,
    build_matrix_bin_direction,
    compute_dyadic_bin_slices,
    compute_e2_persistence_diagnostics,
    compute_num_dyadic_bins,
    decompose_matrix_dyadic,
    extract_compact_e2_state,
    reconstruct_matrix_from_allocation,
)
from methods.spectral_alloc_curvature import (
    CalibrationBatch,
    CalibrationStream,
    _sdpa_math_context,
    compute_calibration_gradient_and_ggn,
    compute_decoupled_nesterov_scaling_and_lookahead,
    compute_nesterov_scaling_and_lookahead,
)
from methods.spectral_alloc_qp import (
    compute_simplex_kkt_residual,
    solve_spectral_allocation_qp,
)
from models import TinyGPTConfig, TinyGPTForCausalLM, attach_param_names


class TestSpectralAllocBinsAndEquivalence(unittest.TestCase):
    """Tests for dyadic rank bins, zero-mass handling, and IsoLoCo equivalence."""

    def test_dyadic_bin_construction_r256_and_r1(self):
        self.assertEqual(compute_num_dyadic_bins(1), 1)
        self.assertEqual(compute_dyadic_bin_slices(1), [(0, 1)])
        self.assertEqual(assign_rank_to_dyadic_bin(1, 1), 0)

        self.assertEqual(compute_num_dyadic_bins(256), 8)
        slices_256 = compute_dyadic_bin_slices(256)
        expected_slices = [
            (0, 1),
            (1, 3),
            (3, 7),
            (7, 15),
            (15, 31),
            (31, 63),
            (63, 127),
            (127, 256),
        ]
        self.assertEqual(slices_256, expected_slices)

        for b_idx, (start, end) in enumerate(slices_256):
            for zero_idx in range(start, end):
                one_idx = zero_idx + 1
                self.assertEqual(assign_rank_to_dyadic_bin(one_idx, 8), b_idx)

    def test_isoloco_equivalence_when_s_equals_ones(self):
        torch.manual_seed(42)
        G = torch.randn(32, 16, dtype=torch.float32)

        U, S, Vh = torch.linalg.svd(G, full_matrices=False)
        S_iso = torch.ones_like(S) * S.mean()
        G_isoloco = torch.linalg.multi_dot((U, torch.diag(S_iso), Vh))

        decomp = decompose_matrix_dyadic(G, param_idx=0, param_name="layer.weight")
        s_ones = np.ones(len(decomp.bin_slices), dtype=np.float64)
        G_recon, fro_sq, max_spec = reconstruct_matrix_from_allocation(decomp, s_ones)

        self.assertTrue(torch.allclose(G_recon, G_isoloco, atol=1e-5, rtol=1e-5))
        self.assertAlmostEqual(
            fro_sq, float(torch.sum(G_isoloco.double() ** 2).item()), places=4
        )
        self.assertAlmostEqual(max_spec, float(S.mean().item()), places=5)

        sum_dirs = torch.zeros_like(G)
        for b_idx in range(len(decomp.bin_slices)):
            d_b = build_matrix_bin_direction(decomp, b_idx)
            if d_b is not None:
                sum_dirs = sum_dirs + d_b
        self.assertTrue(torch.allclose(sum_dirs, G_isoloco, atol=1e-5, rtol=1e-5))

    def test_zero_matrix_and_zero_mass_bins(self):
        G_zero = torch.zeros(16, 8, dtype=torch.bfloat16)
        decomp = decompose_matrix_dyadic(G_zero, param_idx=0, param_name="zero_w")
        self.assertTrue(decomp.is_zero)
        self.assertEqual(sum(decomp.bin_masses), 0.0)

        G_recon, fro_sq, max_spec = reconstruct_matrix_from_allocation(
            decomp, np.ones(len(decomp.bin_slices))
        )
        self.assertEqual(G_recon.dtype, torch.bfloat16)
        self.assertEqual(G_recon.shape, (16, 8))
        self.assertEqual(int(torch.count_nonzero(G_recon).item()), 0)
        self.assertEqual(fro_sq, 0.0)
        self.assertEqual(max_spec, 0.0)

        m_zero = np.zeros(4, dtype=np.float64)
        g_zero = np.ones(4, dtype=np.float64)
        K_zero = np.eye(4, dtype=np.float64)
        res = solve_spectral_allocation_qp(m=m_zero, g=g_zero, K=K_zero, a=2.7)
        self.assertEqual(res.active_bins, 0)
        self.assertFalse(res.fallback)
        self.assertTrue(np.allclose(res.s, np.zeros(4)))

        m_single = np.array([0.0, 5.0, 0.0, 0.0], dtype=np.float64)
        res_single = solve_spectral_allocation_qp(
            m=m_single, g=g_zero, K=K_zero, a=2.7
        )
        self.assertEqual(res_single.active_bins, 1)
        self.assertFalse(res_single.fallback)
        self.assertTrue(np.allclose(res_single.s, np.array([0.0, 1.0, 0.0, 0.0])))


class TestSpectralAllocQP(unittest.TestCase):
    """Tests for QP feasibility, KKT optimality, singular/indefinite curvature, and tie-breaking."""

    def test_qp_feasibility_and_nuclear_norm_preservation(self):
        rng = np.random.default_rng(123)
        m = np.array([2.0, 4.0, 8.0, 16.0], dtype=np.float64)
        g = rng.normal(size=4)
        A = rng.normal(size=(4, 4))
        K = A.T @ A + 0.1 * np.eye(4)
        a = 1.5 * 1.8

        res = solve_spectral_allocation_qp(m=m, g=g, K=K, a=a)
        self.assertFalse(res.fallback)
        self.assertTrue(np.all(res.s >= -1e-7))
        self.assertAlmostEqual(float(np.dot(m, res.s)), float(np.sum(m)), places=6)
        self.assertLessEqual(res.kkt_residual, 1e-5)
        self.assertLessEqual(res.pred_quad_change, res.pred_quad_change_isoloco + 1e-9)

    def test_singular_and_indefinite_curvature_matrices(self):
        m = np.array([1.0, 2.0, 4.0, 8.0], dtype=np.float64)
        g = np.array([0.5, -0.2, 0.1, 0.3], dtype=np.float64)

        v = np.array([1.0, -1.0, 0.5, 0.2], dtype=np.float64)
        u = np.array([0.0, 1.0, 1.0, -1.0], dtype=np.float64)
        K_indef = np.outer(v, v) - 1e-4 * np.outer(u, u)

        res = solve_spectral_allocation_qp(m=m, g=g, K=K_indef, a=2.0)
        self.assertTrue(np.all(res.s >= -1e-7))
        self.assertAlmostEqual(float(np.dot(m, res.s)), float(np.sum(m)), places=6)

        res_flat = solve_spectral_allocation_qp(
            m=m,
            g=np.zeros(4, dtype=np.float64),
            K=np.zeros((4, 4), dtype=np.float64),
            a=2.0,
        )
        self.assertFalse(res_flat.fallback)
        self.assertTrue(np.allclose(res_flat.s, np.ones(4), atol=1e-7))


class TestNesterovLookaheadAndCurvature(unittest.TestCase):
    """Tests for Nesterov look-ahead algebra and exactness of calibration gradient & GGN."""

    def test_nesterov_lookahead_algebra_matches_outer_step(self):
        torch.manual_seed(7)
        lr = 1.5
        mu = 0.8
        theta_t = [
            nn.Parameter(torch.randn(8, 8, dtype=torch.float32)),
            nn.Parameter(torch.randn(8, dtype=torch.float32)),
        ]
        B_prev = [
            torch.randn(8, 8, dtype=torch.float32),
            torch.randn(8, dtype=torch.float32),
        ]
        D_0 = torch.randn(8, 8, dtype=torch.float32)
        D_1 = torch.randn(8, 8, dtype=torch.float32)
        G_rest_1 = torch.randn(8, dtype=torch.float32)
        s = [0.6, 1.4]

        # Standard Nesterov step with tilde_G = [s_0 D_0 + s_1 D_1, G_rest_1]
        tilde_G_0 = s[0] * D_0 + s[1] * D_1
        B_t_0 = mu * B_prev[0] + tilde_G_0
        P_t_0 = tilde_G_0 + mu * B_t_0
        theta_next_0 = theta_t[0].detach() - lr * P_t_0

        B_t_1 = mu * B_prev[1] + G_rest_1
        P_t_1 = G_rest_1 + mu * B_t_1
        theta_next_1 = theta_t[1].detach() - lr * P_t_1

        # Look-ahead expansion: hat_theta - a * sum_b s_b D_b
        named_params = [("w", theta_t[0]), ("b", theta_t[1])]
        a_scale, theta_hat = compute_nesterov_scaling_and_lookahead(
            named_params=named_params,
            momentum_buffers=B_prev,
            grad_rest={1: G_rest_1},
            lr=lr,
            momentum=mu,
            nesterov=True,
        )
        self.assertAlmostEqual(a_scale, lr * (1.0 + mu), places=7)
        theta_from_lookahead_0 = theta_hat["w"] - a_scale * tilde_G_0
        theta_from_lookahead_1 = theta_hat["b"]

        self.assertTrue(
            torch.allclose(theta_next_0, theta_from_lookahead_0, atol=1e-6, rtol=1e-6)
        )
        self.assertTrue(
            torch.allclose(theta_next_1, theta_from_lookahead_1, atol=1e-6, rtol=1e-6)
        )

    def test_calibration_gradient_and_ggn_match_reference(self):
        torch.manual_seed(99)
        cfg = TinyGPTConfig(
            vocab_size=32,
            hidden_size=8,
            num_hidden_layers=1,
            num_attention_heads=2,
            intermediate_size=16,
            max_position_embeddings=16,
            tie_word_embeddings=True,
            bias=False,
        )
        model = TinyGPTForCausalLM(cfg)
        attach_param_names(model)
        model.train()

        named_params = list(model.named_parameters())
        theta_hat = {name: p.detach().clone().float() for name, p in named_params}

        # Two synthetic bin directions on q_proj.weight and k_proj.weight
        q_name = "layers.0.attn.q_proj.weight"
        k_name = "layers.0.attn.k_proj.weight"
        D_0 = {
            q_name: torch.randn_like(theta_hat[q_name]),
            k_name: torch.randn_like(theta_hat[k_name]),
        }
        D_1 = {
            q_name: torch.randn_like(theta_hat[q_name]),
            k_name: torch.randn_like(theta_hat[k_name]),
        }
        bin_directions = [D_0, D_1]
        active_mask = np.array([True, True])

        input_ids = torch.randint(1, 32, (2, 6), dtype=torch.long)
        input_ids[0, -1] = 0  # pad token to verify masked-token handling
        calib_batch = CalibrationBatch(
            input_ids=input_ids,
            language_slices={"en": (0, 1), "fr": (1, 2)},
        )

        stats = compute_calibration_gradient_and_ggn(
            model=model,
            theta_hat_fp32=theta_hat,
            bin_directions=bin_directions,
            active_bins_mask=active_mask,
            calib_batch=calib_batch,
            pad_token_id=0,
            micro_batch_size=1,
            token_chunk_size=3,
        )

        # Verify model training mode was preserved
        self.assertTrue(model.training)

        # Reference directional gradient via standard backward pass
        model.eval()
        labels = input_ids.clone()
        labels[labels == 0] = -100
        outputs = model(input_ids=input_ids, labels=labels)
        grads = torch.autograd.grad(
            outputs.loss, [dict(named_params)[q_name], dict(named_params)[k_name]]
        )
        grad_map = {q_name: grads[0].double(), k_name: grads[1].double()}
        for b_idx, d_dict in enumerate(bin_directions):
            g_ref = float(
                sum(
                    torch.sum(grad_map[name] * d_dict[name].double()).item()
                    for name in d_dict
                )
            )
            self.assertAlmostEqual(stats.g[b_idx], g_ref, places=5)

        # Verify GGN matrix is symmetric and positive semidefinite
        self.assertTrue(np.allclose(stats.K, stats.K.T, atol=1e-10))
        eigvals = np.linalg.eigvalsh(stats.K)
        self.assertGreaterEqual(float(np.min(eigvals)), -1e-10)
        self.assertIn("en", stats.per_language_loss)
        self.assertIn("fr", stats.per_language_loss)


class TestCheckpointMomentumAndStepPreservation(unittest.TestCase):
    """Tests for checkpoint momentum restoration and shape/dtype/device preservation."""

    def test_checkpoint_save_and_restore_momentum_buffers(self):
        torch.manual_seed(11)
        p_2d = nn.Parameter(torch.randn(8, 8, dtype=torch.bfloat16))
        p_2d.param_name = "layers.0.attn.q_proj.weight"
        p_1d = nn.Parameter(torch.randn(8, dtype=torch.float32))
        p_1d.param_name = "layers.0.ln_1.weight"

        opt1 = SpectralAllocLoCo(
            [[p_2d, p_1d]],
            lr=1.5,
            momentum=0.8,
            nesterov=True,
            exclude_keys=["embed_tokens", "wpe", "lm_head"],
        )
        p_2d.grad = torch.randn_like(p_2d)
        p_1d.grad = torch.randn_like(p_1d)
        opt1.step()

        self.assertEqual(p_2d.dtype, torch.bfloat16)
        self.assertEqual(p_2d.shape, (8, 8))
        self.assertEqual(p_1d.dtype, torch.float32)
        self.assertEqual(p_1d.shape, (8,))

        saved_sd = opt1.state_dict()
        buf_2d_expected = opt1.momentum_buffers[0].clone()
        buf_1d_expected = opt1.momentum_buffers[1].clone()

        p_2d_new = nn.Parameter(torch.randn(8, 8, dtype=torch.bfloat16))
        p_2d_new.param_name = "layers.0.attn.q_proj.weight"
        p_1d_new = nn.Parameter(torch.randn(8, dtype=torch.float32))
        p_1d_new.param_name = "layers.0.ln_1.weight"

        opt2 = SpectralAllocLoCo(
            [[p_2d_new, p_1d_new]],
            lr=1.5,
            momentum=0.8,
            nesterov=True,
            exclude_keys=["embed_tokens", "wpe", "lm_head"],
        )
        opt2.load_state_dict(saved_sd)

        self.assertTrue(torch.allclose(opt2.momentum_buffers[0], buf_2d_expected))
        self.assertTrue(torch.allclose(opt2.momentum_buffers[1], buf_1d_expected))

    def test_lazy_calibration_stream_rng_and_counter_checkpoint_restoration(self):
        import tempfile

        with tempfile.TemporaryDirectory() as tmpdir:
            for idx, lang in enumerate(["en", "fr", "de", "es"]):
                arr = np.arange(idx * 1000, idx * 1000 + 256, dtype=np.int32)
                np.save(os.path.join(tmpdir, f"train_0{idx}_{lang}.npy"), arr)

            p_2d = nn.Parameter(torch.randn(8, 8, dtype=torch.float32))
            p_2d.param_name = "layers.0.attn.q_proj.weight"

            opt1 = SpectralAllocLoCo(
                [[p_2d]],
                lr=1.5,
                momentum=0.8,
                nesterov=True,
                shards_path=tmpdir,
                sequence_length=8,
                calibration_batch_size=4,
                seed=42,
            )
            # Draw two batches so RNG state, batch count, and token count advance
            _ = opt1._get_next_calibration_batch(device=torch.device("cpu"))
            _ = opt1._get_next_calibration_batch(device=torch.device("cpu"))
            opt1.cumulative_calibration_tokens = opt1.calibration_stream.cumulative_tokens
            opt1.cumulative_calibration_batches = opt1.calibration_stream.cumulative_batches

            saved_sd = opt1.state_dict()
            expected_next_batch = opt1._get_next_calibration_batch(
                device=torch.device("cpu")
            )

            # Create opt2 with lazy CalibrationStream (self.calibration_stream is None before load_state_dict)
            p_2d_new = nn.Parameter(torch.randn(8, 8, dtype=torch.float32))
            p_2d_new.param_name = "layers.0.attn.q_proj.weight"
            opt2 = SpectralAllocLoCo(
                [[p_2d_new]],
                lr=1.5,
                momentum=0.8,
                nesterov=True,
                shards_path=tmpdir,
                sequence_length=8,
                calibration_batch_size=4,
                seed=42,
            )
            self.assertIsNone(opt2.calibration_stream)
            opt2.load_state_dict(saved_sd)
            self.assertIsNone(opt2.calibration_stream)
            self.assertEqual(opt2.cumulative_calibration_tokens, 64)
            self.assertEqual(opt2.cumulative_calibration_batches, 2)

            # Now lazily initialize CalibrationStream and draw next batch
            restored_next_batch = opt2._get_next_calibration_batch(
                device=torch.device("cpu")
            )
            self.assertIsNotNone(opt2.calibration_stream)
            self.assertEqual(opt2.calibration_stream.cumulative_batches, 3)
            self.assertEqual(opt2.calibration_stream.cumulative_tokens, 96)
            self.assertTrue(
                torch.equal(expected_next_batch.input_ids, restored_next_batch.input_ids)
            )

    def test_sdpa_math_context_executes_cleanly(self):
        q = torch.randn(1, 2, 4, 4, dtype=torch.float32)
        k = torch.randn(1, 2, 4, 4, dtype=torch.float32)
        v = torch.randn(1, 2, 4, 4, dtype=torch.float32)
        with _sdpa_math_context():
            out = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        self.assertEqual(out.shape, (1, 2, 4, 4))

    def test_independent_batch_b_does_not_influence_coefficients_or_updates(self):
        import tempfile

        with tempfile.TemporaryDirectory() as tmpdir:
            for idx, lang in enumerate(["en", "fr", "de", "es"]):
                arr = np.arange(1, 257, dtype=np.int32) % 31 + 1
                np.save(os.path.join(tmpdir, f"train_0{idx}_{lang}.npy"), arr)

            cfg = TinyGPTConfig(
                vocab_size=32,
                hidden_size=8,
                num_hidden_layers=1,
                num_attention_heads=2,
                intermediate_size=16,
                max_position_embeddings=16,
                tie_word_embeddings=True,
                bias=False,
            )

            torch.manual_seed(1234)
            model1 = TinyGPTForCausalLM(cfg)
            attach_param_names(model1)

            torch.manual_seed(1234)
            model2 = TinyGPTForCausalLM(cfg)
            attach_param_names(model2)

            # Give both models identical synthetic pseudo-gradients across two outer steps
            torch.manual_seed(5678)
            grads_step1 = [torch.randn_like(p) for p in model1.parameters()]
            grads_step2 = [torch.randn_like(p) for p in model1.parameters()]

            # Custom Batch B stream for opt1 vs completely different Batch B stream for opt2
            b_stream_1 = [torch.ones((2, 8), dtype=torch.long) * 2 for _ in range(4)]
            b_stream_2 = [torch.ones((2, 8), dtype=torch.long) * 15 for _ in range(4)]

            opt1 = SpectralAllocLoCo(
                [model1.parameters()],
                lr=1.5,
                momentum=0.8,
                nesterov=True,
                exclude_keys=["embed_tokens", "wpe", "lm_head"],
                model=model1,
                shards_path=tmpdir,
                eval_batch_b_stream=b_stream_1,
                sequence_length=8,
                calibration_batch_size=4,
                calibration_micro_batch_size=2,
                calibration_token_chunk_size=16,
                seed=42,
                eval_actual_calib_loss_change=True,
            )
            opt2 = SpectralAllocLoCo(
                [model2.parameters()],
                lr=1.5,
                momentum=0.8,
                nesterov=True,
                exclude_keys=["embed_tokens", "wpe", "lm_head"],
                model=model2,
                shards_path=tmpdir,
                eval_batch_b_stream=b_stream_2,
                sequence_length=8,
                calibration_batch_size=4,
                calibration_micro_batch_size=2,
                calibration_token_chunk_size=16,
                seed=42,
                eval_actual_calib_loss_change=True,
            )

            for grads in (grads_step1, grads_step2):
                for p1, p2, g in zip(model1.parameters(), model2.parameters(), grads):
                    p1.grad = g.clone()
                    p2.grad = g.clone()
                opt1.step()
                opt2.step()

                # Verify selected spectral coefficients s_b are bit-for-bit identical
                d1 = opt1.get_diagnostics()
                d2 = opt2.get_diagnostics()
                for k in d1:
                    if k.startswith("spectral_alloc_s_"):
                        self.assertEqual(d1[k], d2[k])

                # Verify all updated model parameters and momentum buffers are bit-for-bit identical
                for p1, p2 in zip(model1.parameters(), model2.parameters()):
                    self.assertTrue(torch.equal(p1.data, p2.data))
                for b1, b2 in zip(opt1.momentum_buffers, opt2.momentum_buffers):
                    if b1 is not None and b2 is not None:
                        self.assertTrue(torch.equal(b1, b2))

    def test_unified_experiment_yaml_matches_spectral_alloc_iid_and_isolates_max_steps(
        self,
    ):
        from methods import apply_strategy_preset
        from training.config import load_experiment_config

        cfg_unified = load_experiment_config(
            [
                "--config",
                "configs/experiment.yaml",
                "--method",
                "spectral_alloc",
                "--partition",
                "iid",
            ]
        )
        apply_strategy_preset(cfg_unified)

        cfg_dedicated = load_experiment_config(
            ["--config", "configs/spectral_alloc_iid.yaml"]
        )
        apply_strategy_preset(cfg_dedicated)

        for k, v in vars(cfg_dedicated).items():
            if k == "config":
                continue
            self.assertEqual(getattr(cfg_unified, k), v, msg=f"Mismatch on key '{k}'")

        cfg_diag20 = load_experiment_config(
            [
                "--config",
                "configs/experiment.yaml",
                "--method",
                "spectral_alloc",
                "--partition",
                "iid",
                "--max_steps",
                "20",
            ]
        )
        apply_strategy_preset(cfg_diag20)

        self.assertEqual(cfg_diag20.max_steps, 20)
        self.assertEqual(cfg_diag20.token_budget, 12288000)
        self.assertNotEqual(cfg_diag20.run_name, cfg_unified.run_name)
        self.assertNotEqual(cfg_diag20.save_path, cfg_unified.save_path)
        self.assertTrue(cfg_diag20.run_name.endswith("_w4_h30_r20_s42"))

    def test_e2_synthetic_positive_zero_negative_and_zero_norm_projections(self):
        import math

        # 4 orthogonal basis matrices in R^(4x4): E_00, E_11, E_22, and zero-norm bin 3
        D0 = torch.zeros((4, 4), dtype=torch.float32)
        D0[0, 0] = 2.0  # ||D0||_F^2 = 4.0

        D1 = torch.zeros((4, 4), dtype=torch.float32)
        D1[1, 1] = 3.0  # ||D1||_F^2 = 9.0

        D2 = torch.zeros((4, 4), dtype=torch.float32)
        D2[2, 2] = 4.0  # ||D2||_F^2 = 16.0

        D3 = torch.zeros((4, 4), dtype=torch.float32)  # zero-norm direction -> undefined

        # Construct G_avg^(t+1) = +1.5 * D0 + 0.0 * D1 - 0.5 * D2
        G_next = 1.5 * D0 + 0.0 * D1 - 0.5 * D2

        prev_state = {
            "max_bins": 4,
            "s": [1.8, 1.0, 0.4, 0.0],
            "a": 1.5 * (1.0 + 0.8),  # 2.7
            "bin_directions": [{0: D0}, {0: D1}, {0: D2}, {0: D3}],
        }

        e2 = compute_e2_persistence_diagnostics(
            eligible_grads={0: G_next},
            prev_state=prev_state,
            default_bins=4,
        )

        # Bin 0: positive projection r_0 = +1.5
        self.assertAlmostEqual(e2["spectral_alloc_e2_r_0"], 1.5, places=6)
        self.assertEqual(e2["spectral_alloc_e2_defined_0"], 1.0)
        self.assertEqual(e2["spectral_alloc_e2_status_0"], "defined")
        self.assertAlmostEqual(e2["spectral_alloc_e2_prev_s_0"], 1.8, places=6)
        self.assertAlmostEqual(
            e2["spectral_alloc_e2_prev_eff_step_0"], 2.7 * 1.8, places=6
        )

        # Bin 1: zero projection r_1 = 0.0
        self.assertAlmostEqual(e2["spectral_alloc_e2_r_1"], 0.0, places=6)
        self.assertAlmostEqual(e2["spectral_alloc_e2_cos_1"], 0.0, places=6)
        self.assertEqual(e2["spectral_alloc_e2_defined_1"], 1.0)

        # Bin 2: negative projection r_2 = -0.5
        self.assertAlmostEqual(e2["spectral_alloc_e2_r_2"], -0.5, places=6)
        self.assertLess(e2["spectral_alloc_e2_cos_2"], 0.0)
        self.assertEqual(e2["spectral_alloc_e2_defined_2"], 1.0)

        # Bin 3: zero-norm direction is logged as undefined (NaN), NOT 0.0
        self.assertTrue(math.isnan(e2["spectral_alloc_e2_r_3"]))
        self.assertTrue(math.isnan(e2["spectral_alloc_e2_cos_3"]))
        self.assertEqual(e2["spectral_alloc_e2_defined_3"], 0.0)
        self.assertEqual(e2["spectral_alloc_e2_status_3"], "undefined")

        # 3 defined bins (bins 0, 1, 2), of which 1 (bin 2) is negative -> 1/3
        self.assertEqual(e2["spectral_alloc_e2_defined_bins"], 3)
        self.assertAlmostEqual(
            e2["spectral_alloc_e2_negative_proj_fraction"], 1.0 / 3.0, places=6
        )

        # Head (bins 0, 1) mean r = (1.5 + 0.0) / 2 = 0.75; Tail (defined bin 2) mean r = -0.5
        self.assertAlmostEqual(e2["spectral_alloc_e2_head_mean_r"], 0.75, places=6)
        self.assertAlmostEqual(e2["spectral_alloc_e2_tail_mean_r"], -0.5, places=6)
        self.assertAlmostEqual(
            e2["spectral_alloc_e2_head_minus_tail_r"], 1.25, places=6
        )

    def test_e2_checkpoint_persistence_across_rounds(self):
        torch.manual_seed(2026)
        p_2d = nn.Parameter(torch.randn(8, 8, dtype=torch.float32))
        p_2d.param_name = "layers.0.attn.q_proj.weight"

        opt1 = SpectralAllocLoCo(
            [[p_2d]],
            lr=1.5,
            momentum=0.8,
            nesterov=True,
            exclude_keys=["embed_tokens", "wpe", "lm_head"],
        )
        p_2d.grad = torch.randn_like(p_2d)
        opt1.step()

        # Round 0 has no previous round -> e2_available == 0.0
        self.assertEqual(opt1.get_diagnostics()["spectral_alloc_e2_available"], 0.0)

        # Save checkpoint after round 0 and restore into opt2 before round 1
        sd = opt1.state_dict()
        p_2d_resumed = nn.Parameter(p_2d.data.clone())
        p_2d_resumed.param_name = "layers.0.attn.q_proj.weight"
        opt2 = SpectralAllocLoCo(
            [[p_2d_resumed]],
            lr=1.5,
            momentum=0.8,
            nesterov=True,
            exclude_keys=["embed_tokens", "wpe", "lm_head"],
        )
        opt2.load_state_dict(sd)

        next_grad = torch.randn_like(p_2d)
        p_2d.grad = next_grad.clone()
        p_2d_resumed.grad = next_grad.clone()
        opt1.step()
        opt2.step()

        d1 = opt1.get_diagnostics()
        d2 = opt2.get_diagnostics()
        self.assertEqual(d1["spectral_alloc_e2_available"], 1.0)
        self.assertEqual(d2["spectral_alloc_e2_available"], 1.0)
        for b in range(3):
            self.assertAlmostEqual(
                d1[f"spectral_alloc_e2_r_{b}"],
                d2[f"spectral_alloc_e2_r_{b}"],
                places=6,
            )


class TestSpectralAllocDecoupled(unittest.TestCase):
    """Tests for spectral_alloc_decoupled: Nesterov algebra, IsoLoCo parity at s=1, QP feasibility, and checkpointing."""

    def test_decoupled_nesterov_lookahead_and_update_algebra(self):
        torch.manual_seed(314)
        lr = 1.5
        mu = 0.8
        theta_2d = nn.Parameter(torch.randn(8, 8, dtype=torch.float32))
        theta_1d = nn.Parameter(torch.randn(8, dtype=torch.float32))
        B_prev_2d = torch.randn(8, 8, dtype=torch.float32)
        B_prev_1d = torch.randn(8, dtype=torch.float32)

        D_0 = torch.randn(8, 8, dtype=torch.float32)
        D_1 = torch.randn(8, 8, dtype=torch.float32)
        G_iso_2d = D_0 + D_1
        G_rest_1d = torch.randn(8, dtype=torch.float32)
        s = [0.5, 1.5]
        G_sel_2d = s[0] * D_0 + s[1] * D_1

        B_t_2d = mu * B_prev_2d + G_iso_2d
        P_t_2d = G_sel_2d + mu * B_t_2d
        theta_next_2d = theta_2d.detach() - lr * P_t_2d

        named_params = [("w", theta_2d), ("b", theta_1d)]
        g_tilde_ones = {0: G_iso_2d, 1: G_rest_1d}
        a_scale, theta_hat = compute_decoupled_nesterov_scaling_and_lookahead(
            named_params=named_params,
            momentum_buffers=[B_prev_2d, B_prev_1d],
            g_tilde_ones=g_tilde_ones,
            lr=lr,
            momentum=mu,
            nesterov=True,
        )
        self.assertAlmostEqual(a_scale, lr, places=7)

        expected_hat_2d = (
            theta_2d.detach() - lr * (mu**2) * B_prev_2d - lr * mu * G_iso_2d
        )
        expected_hat_1d = (
            theta_1d.detach() - lr * (mu**2) * B_prev_1d - lr * mu * G_rest_1d
        )
        self.assertTrue(torch.allclose(theta_hat["w"], expected_hat_2d, atol=1e-6))
        self.assertTrue(torch.allclose(theta_hat["b"], expected_hat_1d, atol=1e-6))
        self.assertTrue(
            torch.allclose(theta_hat["w"] - a_scale * G_sel_2d, theta_next_2d, atol=1e-6)
        )

    def test_decoupled_identity_allocation_reproduces_isoloco_full_updates(self):
        torch.manual_seed(2718)
        p_elig_1 = nn.Parameter(torch.randn(16, 8, dtype=torch.float32))
        p_elig_1.param_name = "layers.0.attn.q_proj.weight"
        p_excl_1 = nn.Parameter(torch.randn(32, 8, dtype=torch.float32))
        p_excl_1.param_name = "embed_tokens.weight"
        p_1d_1 = nn.Parameter(torch.randn(8, dtype=torch.float32))
        p_1d_1.param_name = "layers.0.ln_1.weight"

        p_elig_2 = nn.Parameter(p_elig_1.data.clone())
        p_elig_2.param_name = "layers.0.attn.q_proj.weight"
        p_excl_2 = nn.Parameter(p_excl_1.data.clone())
        p_excl_2.param_name = "embed_tokens.weight"
        p_1d_2 = nn.Parameter(p_1d_1.data.clone())
        p_1d_2.param_name = "layers.0.ln_1.weight"

        exclude_keys = ["embed_tokens", "wpe", "lm_head"]
        opt_iso = ISOCLoCo(
            [[p_elig_1, p_excl_1, p_1d_1]],
            lr=1.5,
            momentum=0.8,
            nesterov=True,
            exclude_keys=exclude_keys,
        )
        opt_dec = SpectralAllocDecoupledLoCo(
            [[p_elig_2, p_excl_2, p_1d_2]],
            lr=1.5,
            momentum=0.8,
            nesterov=True,
            exclude_keys=exclude_keys,
        )
        self.assertTrue(opt_dec.decoupled_momentum)

        for _ in range(3):
            g_elig = torch.randn_like(p_elig_1)
            g_excl = torch.randn_like(p_excl_1)
            g_1d = torch.randn_like(p_1d_1)

            p_elig_1.grad = g_elig.clone()
            p_excl_1.grad = g_excl.clone()
            p_1d_1.grad = g_1d.clone()

            p_elig_2.grad = g_elig.clone()
            p_excl_2.grad = g_excl.clone()
            p_1d_2.grad = g_1d.clone()

            opt_iso.step()
            opt_dec.step()

            self.assertTrue(torch.allclose(p_elig_1.data, p_elig_2.data, atol=1e-5))
            self.assertTrue(torch.allclose(p_excl_1.data, p_excl_2.data, atol=1e-6))
            self.assertTrue(torch.allclose(p_1d_1.data, p_1d_2.data, atol=1e-6))
            for b_iso, b_dec in zip(opt_iso.momentum_buffers, opt_dec.momentum_buffers):
                self.assertTrue(torch.allclose(b_iso, b_dec, atol=1e-5))

    def test_decoupled_qp_feasibility_and_checkpoint_restoration(self):
        rng = np.random.default_rng(999)
        m = np.array([1.5, 3.0, 6.0, 12.0], dtype=np.float64)
        g = rng.normal(size=4)
        A = rng.normal(size=(4, 4))
        K = A.T @ A + 0.05 * np.eye(4)
        res = solve_spectral_allocation_qp(m=m, g=g, K=K, a=1.5)
        self.assertFalse(res.fallback)
        self.assertTrue(np.all(res.s >= -1e-7))
        self.assertAlmostEqual(float(np.dot(m, res.s)), float(np.sum(m)), places=6)

        torch.manual_seed(404)
        p = nn.Parameter(torch.randn(8, 8, dtype=torch.float32))
        p.param_name = "layers.0.attn.q_proj.weight"
        opt1 = SpectralAllocDecoupledLoCo(
            [[p]],
            lr=1.5,
            momentum=0.8,
            nesterov=True,
            exclude_keys=["embed_tokens", "wpe", "lm_head"],
        )
        p.grad = torch.randn_like(p)
        opt1.step()

        sd = opt1.state_dict()
        p_restored = nn.Parameter(p.data.clone())
        p_restored.param_name = "layers.0.attn.q_proj.weight"
        opt2 = SpectralAllocDecoupledLoCo(
            [[p_restored]],
            lr=1.5,
            momentum=0.8,
            nesterov=True,
            exclude_keys=["embed_tokens", "wpe", "lm_head"],
        )
        opt2.load_state_dict(sd)
        self.assertTrue(torch.equal(opt1.momentum_buffers[0], opt2.momentum_buffers[0]))


if __name__ == "__main__":
    unittest.main()






