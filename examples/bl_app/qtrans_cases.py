"""Verifier inputs for the bl_app project: realistic data, kept out of the app's own files.

qtrans reads QTRANS_CASES from a qtrans_cases.py next to the modules (as well as from each
module itself): {module: {function: [(args...), ...]}}. Not translated, not part of the app.
500 days of simulated returns on 10 assets with 30% pairwise correlation, the annualised
covariance, random market weights, and three views (relative, absolute, a basket).
"""
import numpy as np


def _realistic():
    g = np.random.default_rng(7)
    n, T = 10, 500
    A = g.standard_normal((n, n)) * 0.01
    vols = np.linspace(0.008, 0.02, n)
    corr = 0.3 + 0.7 * np.eye(n)
    cov_d = np.outer(vols, vols) * corr
    R = g.multivariate_normal(np.full(n, 3e-4), cov_d, size=T)
    Sigma = np.cov(R, rowvar=False) * 252
    w = g.random(n)
    w /= w.sum()
    P = np.zeros((3, n))
    P[0, 0], P[0, 1] = 1, -1
    P[1, 2] = 1
    P[2, 3:6] = 1 / 3
    Q = np.array([0.02, 0.05, 0.03])
    tau, delta = 0.05, 2.5
    pi = delta * Sigma @ w
    Omega = np.diag(np.diag(P @ (tau * Sigma) @ P.T))
    Y = R[:, :4] - R[:, :4].mean(axis=0)
    return R, Sigma, w, P, Q, Omega, pi, tau, delta, Y, A


_R, _S, _w, _P, _Q, _O, _pi, _tau, _delta, _Y, _A = _realistic()
_port = _R @ _w  # the portfolio's daily returns

QTRANS_CASES = {
    "bl_core": {
        "nearest_psd": [(_S,), (_S + _A,)],
        "ledoit_wolf_identity": [(_R,)],
        "horizon_returns": [(_R, 5), (_R, 21)],
        "variance_ratio": [(_R, 5), (_R, 21), (_R, 1)],
        "apply_variance_ratio": [(_S, np.linspace(0.7, 1.3, 10))],
        "equilibrium_returns": [(_delta, _S, _w)],
        "implied_confidence_weights": [(_pi, _S, _delta)],
        "omega_he_litterman": [(_P, _S, _tau)],
        "posterior_mu": [(_pi, _S, _P, _Q, _O, _tau), (_pi, _S, None, _Q, _O, _tau)],
        "posterior_sigma": [(_S, _P, _O, _tau), (_S, None, _O, _tau)],
        "portfolio_stats": [(_w, _pi, _S, 0.04)],
        "_downside_dev": [(_w, _R, 0.0, 252.0)],
        "_cvar": [(_w, _R, 0.95)],
        "_ewma_weights": [(500, 60.0), (500, 0.0)],
    },
    "bl_inference": {
        "_ann_sharpe": [(_R[:, 0], 252.0)],
        "_ann_mean": [(_R[:, 0], 252.0)],
        "default_block_length": [(2000,), (1,)],
        "newey_west_lag": [(2000,), (30,)],
        "hit_rate_pvalue": [(0.56, 120), (None, 120), (0.56, 0)],
        "hac_cov": [(_Y, None), (_Y, 5)],
    },
    "bl_backtest": {
        "buy_and_hold_returns": [(_R[:60], _w)],
        "vol_target_scale": [(_port[:250], 0.10, 252.0, 63), (_port[:10], 0.10, 252.0, 63),
                             (_port, 0.12, 252.0, 126, 1.5, 0.25)],
    },
}
