"""Constrained quadratic optimization for Joint Spectral Allocation (spectral_alloc)."""

from dataclasses import dataclass
import logging
import time
from typing import Optional, Tuple

import numpy as np

logger = logging.getLogger("loco")


@dataclass
class SpectralAllocationQPResult:
    """Result container for the spectral allocation quadratic program."""

    s: np.ndarray
    w: np.ndarray
    active_mask: np.ndarray
    active_bins: int
    status: str
    fallback: bool
    kkt_residual: float
    eig_min: float
    eig_max: float
    pred_quad_change: float
    pred_quad_change_isoloco: float
    solve_time_sec: float


def _project_onto_simplex(v: np.ndarray) -> np.ndarray:
    """Euclidean projection of a 1D numpy vector onto the probability simplex."""
    n = v.shape[0]
    if n == 0:
        return v.copy()
    u = np.sort(v)[::-1]
    cssv = np.cumsum(u) - 1.0
    ind = np.arange(1, n + 1)
    cond = u - cssv / ind > 0
    if not np.any(cond):
        return np.full(n, 1.0 / n, dtype=np.float64)
    rho = int(ind[cond][-1])
    theta = cssv[rho - 1] / float(rho)
    w = np.maximum(v - theta, 0.0)
    total = float(np.sum(w))
    if total > 0.0:
        w = w / total
    else:
        w = np.full(n, 1.0 / n, dtype=np.float64)
    return w


def compute_simplex_kkt_residual(
    w: np.ndarray,
    g_hat: np.ndarray,
    K_hat: np.ndarray,
) -> float:
    """
    Compute the KKT residual on the probability simplex for:
        min_{w >= 0, 1^T w = 1}  -g_hat^T w + 0.5 * w^T K_hat w.

    At a KKT point w* on the simplex:
      - Primal feasibility: w* >= 0 and sum(w*) == 1.
      - Stationarity & complementarity: grad(w*)^T (w* - u) <= 0 for all u in simplex,
        which is equivalent to grad(w*)^T w* - min_b grad(w*)_b == 0.
    """
    if w.size == 0:
        return 0.0
    grad = -g_hat + K_hat @ w
    primal_eq_err = abs(float(np.sum(w)) - 1.0)
    primal_ineq_err = float(np.sum(np.maximum(-w, 0.0)))
    dual_gap = max(0.0, float(np.dot(w, grad) - np.min(grad)))
    return float(dual_gap + primal_eq_err + primal_ineq_err)


def _solve_simplex_qp_cvxpy(
    g_hat: np.ndarray,
    K_hat: np.ndarray,
    w_iso: np.ndarray,
    min_tilde_eig: float,
) -> Tuple[Optional[np.ndarray], str]:
    """
    Solve the simplex-constrained QP in w using CVXPY and Clarabel:
        min_{w >= 0, sum(w) == 1}  -g_hat^T w + 0.5 * w^T K_hat w
    with deterministic tie-breaking toward w_iso when K_hat is singular.
    """
    try:
        import cvxpy as cp
    except Exception as exc:
        logger.warning(f"[SpectralAllocQP] CVXPY import failed: {exc}")
        return None, "cvxpy_unavailable"

    n = g_hat.shape[0]
    try:
        w_var = cp.Variable(n)
        K_sym = 0.5 * (K_hat + K_hat.T)
        quad_expr = 0.5 * cp.quad_form(w_var, cp.psd_wrap(K_sym))
        lin_expr = -g_hat @ w_var
        if min_tilde_eig <= 1e-10:
            # Infinitesimal lexicographic tie-breaking toward w_iso (s = 1) along flat directions
            tie_break = 0.5 * 1e-12 * cp.sum_squares(w_var - w_iso)
            objective = cp.Minimize(lin_expr + quad_expr + tie_break)
        else:
            objective = cp.Minimize(lin_expr + quad_expr)

        constraints = [w_var >= 0.0, cp.sum(w_var) == 1.0]
        prob = cp.Problem(objective, constraints)
        prob.solve(
            solver=cp.CLARABEL,
            tol_gap_abs=1e-10,
            tol_gap_rel=1e-10,
            tol_feas=1e-10,
            verbose=False,
        )
        status = str(prob.status) if prob.status is not None else "unknown"
        if w_var.value is None:
            return None, status
        return np.asarray(w_var.value, dtype=np.float64).reshape(n), status
    except Exception as exc:
        logger.warning(f"[SpectralAllocQP] Clarabel solve failed: {exc}")
        return None, "solver_exception"


def solve_spectral_allocation_qp(
    m: np.ndarray,
    g: np.ndarray,
    K: np.ndarray,
    a: float,
    kkt_tol: float = 1e-5,
    feas_tol: float = 1e-7,
    psd_eps: float = 0.0,
) -> SpectralAllocationQPResult:
    """
    Solve the Nesterov-aware spectral allocation QP:
        min_{s >= 0, m^T s = m^T 1}  -g^T s + (a / 2) s^T K s

    via the probability-simplex transformation:
        c = sum_b m_b,   w_b = (m_b * s_b) / c   (s_b = c * w_b / m_b).
    """
    t0 = time.perf_counter()

    m_arr = np.asarray(m, dtype=np.float64).reshape(-1)
    num_bins = m_arr.shape[0]
    g_arr = np.asarray(g, dtype=np.float64).reshape(num_bins)
    K_arr = np.asarray(K, dtype=np.float64).reshape(num_bins, num_bins)

    if not np.all(np.isfinite(m_arr)) or np.any(m_arr < 0.0):
        s_fallback = np.where(m_arr > 0.0, 1.0, 0.0)
        return SpectralAllocationQPResult(
            s=s_fallback,
            w=np.zeros(num_bins, dtype=np.float64),
            active_mask=m_arr > 0.0,
            active_bins=int(np.sum(m_arr > 0.0)),
            status="invalid_mass_fallback",
            fallback=True,
            kkt_residual=float("nan"),
            eig_min=float("nan"),
            eig_max=float("nan"),
            pred_quad_change=float("nan"),
            pred_quad_change_isoloco=float("nan"),
            solve_time_sec=time.perf_counter() - t0,
        )

    active_mask = m_arr > 0.0
    active_bins = int(np.sum(active_mask))

    if active_bins == 0:
        return SpectralAllocationQPResult(
            s=np.zeros(num_bins, dtype=np.float64),
            w=np.zeros(num_bins, dtype=np.float64),
            active_mask=active_mask,
            active_bins=0,
            status="zero_mass",
            fallback=False,
            kkt_residual=0.0,
            eig_min=0.0,
            eig_max=0.0,
            pred_quad_change=0.0,
            pred_quad_change_isoloco=0.0,
            solve_time_sec=time.perf_counter() - t0,
        )

    m_act = m_arr[active_mask]
    g_act = g_arr[active_mask]
    K_act = K_arr[np.ix_(active_mask, active_mask)]
    c_total = float(np.sum(m_act))
    w_iso_act = m_act / c_total

    def _build_full_s_w(w_active: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        s_full = np.zeros(num_bins, dtype=np.float64)
        w_full = np.zeros(num_bins, dtype=np.float64)
        s_full[active_mask] = (c_total * w_active) / m_act
        w_full[active_mask] = w_active
        return s_full, w_full

    if (
        not np.all(np.isfinite(g_act))
        or not np.all(np.isfinite(K_act))
        or not np.isfinite(a)
        or a <= 0.0
    ):
        s_iso, w_iso = _build_full_s_w(w_iso_act)
        return SpectralAllocationQPResult(
            s=s_iso,
            w=w_iso,
            active_mask=active_mask,
            active_bins=active_bins,
            status="nonfinite_input_fallback",
            fallback=True,
            kkt_residual=float("nan"),
            eig_min=float("nan"),
            eig_max=float("nan"),
            pred_quad_change=float("nan"),
            pred_quad_change_isoloco=float("nan"),
            solve_time_sec=time.perf_counter() - t0,
        )

    # Symmetrize curvature matrix K on active bins
    K_sym = 0.5 * (K_act + K_act.T)
    raw_eigvals, raw_eigvecs = np.linalg.eigh(K_sym)
    eig_min = float(np.min(raw_eigvals))
    eig_max = float(np.max(raw_eigvals))

    # Numerical PSD projection of K_sym when needed
    if eig_min < psd_eps:
        clamped_eigvals = np.maximum(raw_eigvals, psd_eps)
        K_psd = (raw_eigvecs * clamped_eigvals[None, :]) @ raw_eigvecs.T
        K_psd = 0.5 * (K_psd + K_psd.T)
    else:
        K_psd = K_sym

    def _eval_pred_quad_change(s_act: np.ndarray) -> float:
        obj_unscaled = -float(np.dot(g_act, s_act)) + 0.5 * float(a) * float(
            s_act @ (K_psd @ s_act)
        )
        return float(a) * obj_unscaled

    s_iso_act = np.ones(active_bins, dtype=np.float64)
    pred_iso = _eval_pred_quad_change(s_iso_act)

    if active_bins == 1:
        s_iso, w_iso = _build_full_s_w(w_iso_act)
        return SpectralAllocationQPResult(
            s=s_iso,
            w=w_iso,
            active_mask=active_mask,
            active_bins=1,
            status="single_active_bin",
            fallback=False,
            kkt_residual=0.0,
            eig_min=eig_min,
            eig_max=eig_max,
            pred_quad_change=pred_iso,
            pred_quad_change_isoloco=pred_iso,
            solve_time_sec=time.perf_counter() - t0,
        )

    # Transform to simplex variable w_b = (m_b * s_b) / c
    inv_scale = c_total / m_act
    g_tilde = inv_scale * g_act
    K_tilde = float(a) * (inv_scale[:, None] * K_psd * inv_scale[None, :])
    K_tilde = 0.5 * (K_tilde + K_tilde.T)

    tilde_eigvals, tilde_eigvecs = np.linalg.eigh(K_tilde)
    if np.min(tilde_eigvals) < 0.0:
        tilde_eigvals = np.maximum(tilde_eigvals, 0.0)
        K_tilde = (tilde_eigvecs * tilde_eigvals[None, :]) @ tilde_eigvecs.T
        K_tilde = 0.5 * (K_tilde + K_tilde.T)

    # Objective scaling for numerical stability
    g_norm_inf = float(np.max(np.abs(g_tilde)))
    K_norm_fro = float(np.linalg.norm(K_tilde, ord="fro"))
    obj_scale = max(g_norm_inf, K_norm_fro, 1e-12)

    # Degenerate zero-objective case: tie-break deterministically to s = 1
    if g_norm_inf <= 1e-15 and K_norm_fro <= 1e-15:
        s_iso, w_iso = _build_full_s_w(w_iso_act)
        return SpectralAllocationQPResult(
            s=s_iso,
            w=w_iso,
            active_mask=active_mask,
            active_bins=active_bins,
            status="optimal_tie_break_ones",
            fallback=False,
            kkt_residual=0.0,
            eig_min=eig_min,
            eig_max=eig_max,
            pred_quad_change=pred_iso,
            pred_quad_change_isoloco=pred_iso,
            solve_time_sec=time.perf_counter() - t0,
        )

    g_hat = g_tilde / obj_scale
    K_hat = K_tilde / obj_scale
    K_hat = 0.5 * (K_hat + K_hat.T)

    w_sol, status_str = _solve_simplex_qp_cvxpy(
        g_hat=g_hat,
        K_hat=K_hat,
        w_iso=w_iso_act,
        min_tilde_eig=float(np.min(tilde_eigvals)) / obj_scale,
    )

    if w_sol is None or not np.all(np.isfinite(w_sol)):
        s_iso, w_iso = _build_full_s_w(w_iso_act)
        return SpectralAllocationQPResult(
            s=s_iso,
            w=w_iso,
            active_mask=active_mask,
            active_bins=active_bins,
            status=f"fallback_{status_str}",
            fallback=True,
            kkt_residual=float("nan"),
            eig_min=eig_min,
            eig_max=eig_max,
            pred_quad_change=pred_iso,
            pred_quad_change_isoloco=pred_iso,
            solve_time_sec=time.perf_counter() - t0,
        )

    # Clean tiny numerical noise at simplex boundary
    if (
        np.min(w_sol) >= -feas_tol
        and abs(float(np.sum(w_sol)) - 1.0) <= feas_tol
    ):
        w_clean = _project_onto_simplex(w_sol)
    else:
        w_clean = w_sol

    # Deterministic tie-breaking toward w_iso (s = 1) when objective values are equivalent
    def _scaled_obj(w_vec: np.ndarray) -> float:
        return -float(np.dot(g_hat, w_vec)) + 0.5 * float(w_vec @ (K_hat @ w_vec))

    obj_sol = _scaled_obj(w_clean)
    obj_iso = _scaled_obj(w_iso_act)
    if abs(obj_iso - obj_sol) <= 1e-10 * max(1.0, abs(obj_sol)):
        kkt_iso = compute_simplex_kkt_residual(w_iso_act, g_hat, K_hat)
        if kkt_iso <= kkt_tol:
            w_clean = w_iso_act.copy()
            obj_sol = obj_iso

    kkt_res = compute_simplex_kkt_residual(w_clean, g_hat, K_hat)
    primal_feasible = (
        bool(np.all(np.isfinite(w_clean)))
        and float(np.min(w_clean)) >= -feas_tol
        and abs(float(np.sum(w_clean)) - 1.0) <= feas_tol
    )

    if (
        not primal_feasible
        or not np.isfinite(kkt_res)
        or kkt_res > kkt_tol
        or obj_sol > obj_iso + 1e-7
    ):
        s_iso, w_iso = _build_full_s_w(w_iso_act)
        return SpectralAllocationQPResult(
            s=s_iso,
            w=w_iso,
            active_mask=active_mask,
            active_bins=active_bins,
            status=f"fallback_verification_failed_{status_str}",
            fallback=True,
            kkt_residual=float(kkt_res),
            eig_min=eig_min,
            eig_max=eig_max,
            pred_quad_change=pred_iso,
            pred_quad_change_isoloco=pred_iso,
            solve_time_sec=time.perf_counter() - t0,
        )

    s_full, w_full = _build_full_s_w(w_clean)
    s_act = s_full[active_mask]
    pred_selected = _eval_pred_quad_change(s_act)

    return SpectralAllocationQPResult(
        s=s_full,
        w=w_full,
        active_mask=active_mask,
        active_bins=active_bins,
        status=status_str,
        fallback=False,
        kkt_residual=float(kkt_res),
        eig_min=eig_min,
        eig_max=eig_max,
        pred_quad_change=pred_selected,
        pred_quad_change_isoloco=pred_iso,
        solve_time_sec=time.perf_counter() - t0,
    )


