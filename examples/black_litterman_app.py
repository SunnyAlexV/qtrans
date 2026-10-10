"""The numerical core of a real application, translated as it stands.

These 21 functions are copied from the Black-Litterman portfolio app
(github.com/SunnyAlexV/Black-Litterman-model: bl_core.py, bl_inference.py, bl_backtest.py).
The only change is the type hints, which qtrans needs and the app didn't have; every body is
the app's own code, comments included. `# from <file> line <n>` marks where each came from.

They exercise what quant code written without qtrans in mind actually uses: np.errstate,
np.clip with a None bound, boolean masks, keepdims and broadcasting, var/cumprod over an axis,
np.einsum, np.linalg.eigh / pinv, `try: solve(...) except LinAlgError: pinv(...)`,
`lag: int | None = None`, and functions that return None for "no answer".
Realistic inputs (500 days of returns on 10 assets, three views) are in bl_app_cases.py.
"""
import math

import numpy as np

from qtrans.typing import Mat, Vec

# from bl_core.py line 25
def nearest_psd(matrix: Mat, eps: float=1e-10) -> Mat:
    matrix = (matrix + matrix.T) / 2.0
    vals, vecs = np.linalg.eigh(matrix)
    vals = np.maximum(vals, eps)
    return vecs @ np.diag(vals) @ vecs.T

# from bl_core.py line 32
def ledoit_wolf_identity(X: Mat) -> Mat:
    """Ledoit-Wolf (2004) shrinkage of the sample covariance toward a scaled
    identity. X is a T x N matrix of (periodic) returns. Returns periodic cov."""
    X = np.asarray(X, dtype=float)
    t, n = X.shape
    Xc = X - X.mean(axis=0, keepdims=True)
    S = (Xc.T @ Xc) / t
    m = np.trace(S) / n
    d2 = np.sum((S - m * np.eye(n)) ** 2) / n

    # The textbook form of b_bar2 loops over every observation, building an
    # n x n outer product each time. With 15 years of daily data that is ~3,800
    # Python iterations per call, and a walk-forward backtest calls this once
    # per rebalance on an expanding window — twelve seconds of a fifteen-second
    # backtest went here. Expanding the Frobenius norm removes the loop:
    #
    #   sum_i ||x_i x_i' - S||_F^2
    #     = sum_i (x_i'x_i)^2  -  2 sum_i x_i' S x_i  +  t ||S||_F^2
    #
    # which is three vectorised reductions. Algebraically identical, no
    # approximation — verified against the loop to machine precision by a test.
    sq = np.einsum("ij,ij->i", Xc, Xc)              # ||x_i||^2 for each i
    term1 = float(np.dot(sq, sq))                   # sum_i (x_i'x_i)^2
    term2 = float(np.sum((Xc @ S) * Xc))            # sum_i x_i' S x_i
    term3 = t * float(np.sum(S * S))                # t ||S||_F^2
    b_bar2 = (term1 - 2.0 * term2 + term3) / (t ** 2) / n
    b2 = min(b_bar2, d2)
    a2 = d2 - b2
    shrink = b2 / d2 if d2 > 0 else 1.0
    sigma = shrink * m * np.eye(n) + (a2 / d2 if d2 > 0 else 0.0) * S
    return sigma

# from bl_core.py line 65
def _ewma_weights(t: int, halflife: float) -> Vec:
    lam = 0.5 ** (1.0 / max(halflife, 1e-9))
    w = lam ** np.arange(t - 1, -1, -1)
    return w / w.sum()

# from bl_core.py line 71
def horizon_returns(returns: Mat, h: int) -> Mat:
    """Overlapping h-period compounded returns from a period-return panel.

    Row k is the return between the price at k and the price at k + h, i.e. it
    compounds periods k+1 through k+h. The output therefore has len(R) - h rows,
    one per window that fully fits inside the sample.
    """
    R = np.asarray(returns, dtype=float)
    h = max(1, int(h))
    if h == 1 or len(R) <= h:
        return R
    g = np.cumprod(1.0 + R, axis=0)
    return g[h:] / g[:-h] - 1.0

# from bl_core.py line 86
def variance_ratio(returns: Mat, h: int, shrink_c: float=10.0, lo: float=0.25, hi: float=4.0) -> Vec:
    """Per-asset variance ratio VR(h) = Var(h-period) / (h x Var(1-period)).

    The whole reason a holding period should change a portfolio. Scaling daily
    variance by h assumes returns are independent from one day to the next. They
    are not: an asset that mean-reverts is LESS risky over a year than its daily
    volatility implies (VR < 1) and deserves more weight in a long hold, while a
    trending one is MORE risky (VR > 1) and deserves less. Under strict i.i.d.
    returns VR = 1 at every horizon and the holding period genuinely does not
    matter -- which is why the app used to give the same answer for a week and a
    year.

    Overlapping windows mean the estimate is far noisier than the raw sample
    size suggests: 3,780 daily observations contain only ~15 independent annual
    ones. So VR is shrunk toward 1 by the effective sample count, which leaves a
    weekly horizon essentially untouched and pulls a yearly one most of the way
    back unless the evidence is strong. Clipped to [lo, hi] so a single freak
    window cannot rescale a book.
    """
    R = np.asarray(returns, dtype=float)
    h = max(1, int(h))
    if R.ndim == 1:
        R = R[:, None]
    n = R.shape[1]
    if h == 1 or len(R) <= 2 * h:
        return np.ones(n)
    var1 = np.var(R, axis=0, ddof=1)
    Rh = horizon_returns(R, h)
    if len(Rh) < 3:
        return np.ones(n)
    varh = np.var(Rh, axis=0, ddof=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        vr = np.where(var1 > 1e-18, varh / (h * var1), 1.0)
    vr = np.where(np.isfinite(vr), vr, 1.0)
    # Effective independent observations at this horizon, not the overlapping count.
    n_eff = len(R) / float(h)
    k = n_eff / (n_eff + float(shrink_c))
    vr = 1.0 + (vr - 1.0) * k
    return np.clip(vr, lo, hi)

# from bl_core.py line 127
def apply_variance_ratio(cov_annual: Mat, vr: Vec) -> Mat:
    """Rescale each asset's variance by its variance ratio, keeping correlations.

    Correlations are estimated well from daily data and badly from a handful of
    overlapping annual windows, so only the volatilities are re-scaled here.
    That is a deliberate compromise and the reason it is stated in the app.
    """
    S = np.asarray(cov_annual, dtype=float)
    d = np.sqrt(np.clip(np.asarray(vr, dtype=float), 1e-12, None))
    return nearest_psd(S * np.outer(d, d))

# from bl_core.py line 298
def equilibrium_returns(delta: float, Sigma: Mat, w_mkt: Vec) -> Vec:
    """Reverse optimisation: pi = delta * Sigma * w_mkt.

    This is the whole trick of Black-Litterman. Instead of asking 'what returns
    do I forecast?' (noisy, and the optimiser will amplify the noise), we ask
    'what returns would make today's market portfolio optimal?' and treat that
    as the prior. Returned as EXCESS returns.
    """
    return float(delta) * (np.asarray(Sigma, dtype=float) @ np.asarray(w_mkt, dtype=float))

# from bl_core.py line 309
def implied_confidence_weights(mu_excess: Vec, Sigma: Mat, delta: float) -> Vec:
    """Unconstrained optimal weights for a given excess-return vector:
    w = (delta * Sigma)^-1 * mu.  Note w = w_mkt exactly when mu = pi."""
    Sigma = np.asarray(Sigma, dtype=float)
    try:
        return np.linalg.solve(delta * Sigma, np.asarray(mu_excess, dtype=float))
    except np.linalg.LinAlgError:
        return np.linalg.pinv(delta * Sigma) @ np.asarray(mu_excess, dtype=float)

# from bl_core.py line 595
def omega_he_litterman(P: Mat, Sigma: Mat, tau: float) -> Mat:
    """Omega = diag(P (tau Sigma) P').  The classic choice: a view is assumed to
    be exactly as uncertain as the prior is about that same combination of
    assets. Needs no user input at all."""
    P = np.asarray(P, dtype=float)
    if P.size == 0:
        return np.zeros((0, 0))
    M = P @ (tau * np.asarray(Sigma, dtype=float)) @ P.T
    d = np.clip(np.diag(M).astype(float), 1e-12, None)
    return np.diag(d)

# from bl_core.py line 671
def posterior_mu(pi: Vec, Sigma: Mat, P: Mat | None, Q: Vec, Omega: Mat, tau: float) -> Vec:
    """Black-Litterman posterior expected (excess) returns.

    Uses the Sherman-Morrison-Woodbury form
        mu = pi + tau*Sigma*P' (P tau*Sigma P' + Omega)^-1 (Q - P pi)
    rather than the textbook
        mu = [(tau Sigma)^-1 + P' Omega^-1 P]^-1 [(tau Sigma)^-1 pi + P' Omega^-1 Q]
    They are algebraically identical, but the first inverts a k x k matrix
    (k = number of views, usually 1-5) instead of an n x n one, and never
    inverts Omega — which matters because a high-confidence view drives Omega
    toward zero and makes Omega^-1 blow up.
    """
    pi = np.asarray(pi, dtype=float)
    if P is None or np.asarray(P).size == 0:
        return pi.copy()
    P = np.asarray(P, dtype=float)
    Q = np.asarray(Q, dtype=float)
    Omega = np.asarray(Omega, dtype=float)
    tauS = tau * np.asarray(Sigma, dtype=float)

    A = P @ tauS @ P.T + Omega
    resid = Q - P @ pi
    try:
        adj = tauS @ P.T @ np.linalg.solve(A, resid)
    except np.linalg.LinAlgError:
        adj = tauS @ P.T @ (np.linalg.pinv(A) @ resid)
    return pi + adj

# from bl_core.py line 700
def posterior_sigma(Sigma: Mat, P: Mat | None, Omega: Mat, tau: float) -> Mat:
    """Posterior covariance of returns: Sigma_p = Sigma + M, where M is the
    posterior variance of the mean. Using this instead of Sigma is the
    theoretically correct thing to do — it says 'I am not certain about the
    expected returns either' — and it pushes the optimiser toward slightly more
    diversified portfolios. Off by default because most practitioners, and most
    textbook worked examples, just use Sigma."""
    Sigma = np.asarray(Sigma, dtype=float)
    tauS = tau * Sigma
    if P is None or np.asarray(P).size == 0:
        return nearest_psd(Sigma + tauS)
    P = np.asarray(P, dtype=float)
    A = P @ tauS @ P.T + np.asarray(Omega, dtype=float)
    try:
        M = tauS - tauS @ P.T @ np.linalg.solve(A, P @ tauS)
    except np.linalg.LinAlgError:
        M = tauS - tauS @ P.T @ np.linalg.pinv(A) @ P @ tauS
    return nearest_psd(Sigma + M)

# from bl_core.py line 764
def portfolio_stats(w: Vec, mu: Vec, cov: Mat, rf: float) -> tuple[float, float, float]:
    w = np.asarray(w, dtype=float)
    ret = float(w @ mu)
    vol = math.sqrt(max(float(w @ cov @ w), 0.0))
    sharpe = (ret - rf) / vol if vol > 0 else np.nan
    return ret, vol, sharpe

# from bl_core.py line 772
def _downside_dev(w: Vec, R: Mat, target_period: float, freq_per_year: float) -> float:
    p = R @ w
    neg = np.minimum(p - target_period, 0.0)
    dd = math.sqrt(max(np.mean(neg ** 2), 1e-16))
    return dd * math.sqrt(freq_per_year)

# from bl_core.py line 779
def _cvar(w: Vec, R: Mat, alpha: float) -> float:
    p = R @ w
    losses = -p
    var = np.quantile(losses, alpha)
    tail = losses[losses >= var]
    return float(tail.mean()) if len(tail) else float(var)

# from bl_inference.py line 20
def default_block_length(T: int) -> int:
    """Mean block length ~ T^(1/3), the rate-optimal order for variance
    estimation. About 13 for eight years of daily data."""
    return max(2, int(round(T ** (1.0 / 3.0))))

# from bl_inference.py line 47
def _ann_mean(x: Vec, fpy: float) -> float:
    return np.mean(x, axis=-1) * fpy

# from bl_inference.py line 51
def _ann_sharpe(x: Vec, fpy: float) -> float:
    sd = np.std(x, axis=-1, ddof=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(sd > 0, np.mean(x, axis=-1) / sd * math.sqrt(fpy), np.nan)

# from bl_inference.py line 102
def newey_west_lag(T: int) -> int:
    """Newey-West (1994) plug-in bandwidth, floor(4 (T/100)^(2/9))."""
    return int(math.floor(4.0 * (T / 100.0) ** (2.0 / 9.0)))

# from bl_inference.py line 107
def hac_cov(Y: Mat, lag: int | None=None) -> Mat:
    """Newey-West long-run covariance of the columns of Y (already demeaned)."""
    Y = np.asarray(Y, dtype=float)
    T = Y.shape[0]
    L = newey_west_lag(T) if lag is None else int(lag)
    S = Y.T @ Y / T
    for j in range(1, L + 1):
        w = 1.0 - j / (L + 1.0)                     # Bartlett kernel
        G = Y[j:].T @ Y[:-j] / T
        S += w * (G + G.T)
    return S

# from bl_inference.py line 185
def hit_rate_pvalue(hit_rate: float | None, n_independent: int) -> float | None:
    """One-sided p-value for hit rate > 50%, normal approximation to the
    binomial on the number of independent checks."""
    if hit_rate is None or not n_independent:
        return None
    z = (float(hit_rate) - 0.5) / math.sqrt(0.25 / float(n_independent))
    return 0.5 * math.erfc(z / math.sqrt(2))

# from bl_backtest.py line 82
def buy_and_hold_returns(R: Mat, w: Vec) -> tuple[Vec, Vec]:
    """Period returns of a basket bought at weights `w` and then LEFT ALONE.

    The obvious-looking `R @ w` is not this. Applying a fixed weight vector to
    every period silently rebalances the portfolio back to `w` every single
    period — so a 'yearly rebalanced' strategy computed that way is really a
    daily-rebalanced one that happens to re-optimise once a year, and every
    rebalance frequency collapses to nearly the same result.

    Here the holdings drift: each asset's value compounds at its own return, and
    the portfolio return is the change in the total.
    """
    R = np.asarray(R, dtype=float)
    w = np.asarray(w, dtype=float)
    if R.size == 0:
        return np.zeros(0), w.copy()
    growth = np.cumprod(1.0 + R, axis=0)          # per-asset value path
    V = growth @ w                                # portfolio value path, V(0-)=1
    prev = np.concatenate(([float(np.sum(w))], V[:-1]))
    with np.errstate(divide="ignore", invalid="ignore"):
        r = np.where(np.abs(prev) > 1e-12, V / prev - 1.0, 0.0)
    # weights the portfolio has DRIFTED to by the end — what the next rebalance
    # actually trades away from, and therefore what turnover must be measured
    # against.
    end_val = w * growth[-1]
    tot = float(np.sum(end_val))
    w_end = end_val / tot if abs(tot) > 1e-12 else w.copy()
    return r, w_end

from bl_app_cases import QTRANS_CASES  # noqa: E402,F401  (verifier inputs)
