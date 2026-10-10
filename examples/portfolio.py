"""Mean-variance portfolio construction: sample covariance, minimum-variance and tangency
portfolios, the efficient frontier, risk contributions, and Black-Litterman."""
import numpy as np

from qtrans.typing import Mat, Vec


def sample_cov(returns: Mat, ddof: int = 1) -> Mat:
    """Covariance of the columns of a T x n matrix of returns."""
    T = returns.shape[0]
    X = returns - returns.mean(axis=0)  # demean each asset (broadcasts across rows)
    return X.T @ X / (T - ddof)


def portfolio_vol(w: Vec, cov: Mat) -> float:
    return float(np.sqrt(w @ cov @ w))


def min_variance_weights(cov: Mat) -> Vec:
    """Fully invested minimum-variance portfolio: w proportional to inv(cov) @ 1."""
    x = np.linalg.solve(cov, np.ones(cov.shape[0]))
    return x / x.sum()


def tangency_weights(mu: Vec, cov: Mat, rf: float) -> Vec:
    """Maximum Sharpe ratio portfolio: w proportional to inv(cov) @ (mu - rf)."""
    x = np.linalg.solve(cov, mu - rf)
    return x / np.sum(x)


def frontier_weights(mu: Vec, cov: Mat, target: float) -> Vec:
    """Minimum-variance weights with expected return `target` (Merton's closed form)."""
    inv = np.linalg.inv(cov)
    ones = np.ones(len(mu))
    a = ones @ inv @ ones
    b = ones @ inv @ mu
    c = mu @ inv @ mu
    d = a * c - b * b
    lam = (c - b * target) / d
    gam = (a * target - b) / d
    return inv @ (lam * ones + gam * mu)


def risk_contributions(w: Vec, cov: Mat) -> Vec:
    """Each asset's share of portfolio volatility; they sum to the volatility."""
    marginal = cov @ w
    return w * marginal / np.sqrt(w @ marginal)


def black_litterman(cov: Mat, w_mkt: Vec, P: Mat, Q: Vec, omega_diag: Vec,
                    delta: float = 2.5, tau: float = 0.05) -> tuple[Vec, Mat]:
    """Posterior mean and covariance of returns given k views P @ mu = Q with independent
    uncertainties omega_diag (He and Litterman, 1999).

    Written in the Woodbury form, which only solves the small k x k system
    M = P tau cov P' + Omega. The textbook form inverts inv(tau cov) + P' inv(Omega) P, which
    becomes numerically singular as a view grows confident: against a 50-digit reference it is
    accurate to only about 6 digits at omega = 1e-14, while this form keeps full precision
    (tools/black_litterman_stability.py)."""
    if np.any(omega_diag <= 0):
        raise ValueError("view variances must be positive")
    pi = delta * cov @ w_mkt  # equilibrium returns implied by market weights
    tau_cov = tau * cov
    M = P @ tau_cov @ P.T + np.diag(omega_diag)
    post_mu = pi + tau_cov @ P.T @ np.linalg.solve(M, Q - P @ pi)
    post_cov = cov + tau_cov - tau_cov @ P.T @ np.linalg.solve(M, P @ tau_cov)
    return post_mu, post_cov


def bl_weights(cov: Mat, w_mkt: Vec, P: Mat, Q: Vec, omega_diag: Vec, delta: float = 2.5) -> Vec:
    """Unconstrained mean-variance weights under the Black-Litterman posterior."""
    mu, sigma = black_litterman(cov, w_mkt, P, Q, omega_diag, delta)
    return np.linalg.solve(delta * sigma, mu)


# --- test inputs for `qtrans verify` -------------------------------------------------
_rng = np.random.default_rng(42)
_R = _rng.normal(0.0005, 0.01, size=(250, 5)) @ np.linalg.cholesky(
    np.array([[1.0, 0.3, 0.2, 0.1, 0.0], [0.3, 1.0, 0.4, 0.2, 0.1], [0.2, 0.4, 1.0, 0.3, 0.2],
              [0.1, 0.2, 0.3, 1.0, 0.4], [0.0, 0.1, 0.2, 0.4, 1.0]])).T
_COV = np.cov(_R, rowvar=False) * 252
_MU = np.array([0.06, 0.08, 0.07, 0.10, 0.05])
_W = np.array([0.3, 0.25, 0.2, 0.15, 0.1])
_P = np.array([[1.0, -1.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.5, 0.5, -1.0]])  # 2 relative views
_Q = np.array([0.02, 0.01])
_OMEGA = np.array([0.0004, 0.0009])
_SINGULAR = np.ones((3, 3))

QTRANS_CASES = {
    "sample_cov": [(_R,), (_R[:20],), (_R[:, :2], 0)],
    "portfolio_vol": [(_W, _COV), (np.ones(5) / 5, _COV)],
    "min_variance_weights": [(_COV,), (_COV[:3, :3],), (_SINGULAR,)],
    "tangency_weights": [(_MU, _COV, 0.02), (_MU, _COV, 0.0)],
    "frontier_weights": [(_MU, _COV, t) for t in (0.05, 0.07, 0.09, 0.12)],
    "risk_contributions": [(_W, _COV), (np.ones(5) / 5, _COV)],
    "black_litterman": [(_COV, _W, _P, _Q, _OMEGA), (_COV, _W, _P[:1], _Q[:1], _OMEGA[:1], 3.0, 0.025),
                        (_COV, _W, _P, _Q, np.array([1e-14, 0.0009])), (_COV, _W, _P, _Q, -_OMEGA)],
    "bl_weights": [(_COV, _W, _P, _Q, _OMEGA), (_COV, _W, _P, _Q * 3, _OMEGA * 10)],
}
