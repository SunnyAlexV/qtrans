"""Monte Carlo pricing and risk: antithetic variates, simulated paths, an Asian option with a
control variate, a barrier option, the Heston model, and portfolio VaR. Every simulation draws
from np.random.default_rng(seed), which qtrans reproduces exactly in C++, so a translated
simulation uses the same random numbers as the Python one."""
import math

import numpy as np
from scipy.stats import norm

from qtrans.typing import Mat, Vec


def european_mc(S0: float, K: float, r: float, sigma: float, T: float, n_paths: int, seed: int,
                is_call: bool = True) -> tuple[float, float]:
    """Black-Scholes price by simulation with antithetic variates: (price, standard error)."""
    if n_paths < 2:
        raise ValueError("need at least two paths")
    rng = np.random.default_rng(seed)
    z = rng.standard_normal(n_paths)
    drift = (r - 0.5 * sigma ** 2) * T
    vol = sigma * math.sqrt(T)
    up = S0 * np.exp(drift + vol * z)
    down = S0 * np.exp(drift - vol * z)  # the mirror-image path: cancels much of the noise
    if is_call:
        pay = 0.5 * (np.maximum(up - K, 0.0) + np.maximum(down - K, 0.0))
    else:
        pay = 0.5 * (np.maximum(K - up, 0.0) + np.maximum(K - down, 0.0))
    disc = math.exp(-r * T)
    return disc * np.mean(pay), disc * np.std(pay, ddof=1) / math.sqrt(n_paths)


def european_from_draws(z: Vec, S0: float, K: float, r: float, sigma: float, T: float) -> float:
    """The plain estimator with the normal draws passed in: the other way to make a simulation
    reproducible across languages."""
    ST = S0 * np.exp((r - 0.5 * sigma ** 2) * T + sigma * math.sqrt(T) * z)
    return math.exp(-r * T) * np.mean(np.maximum(ST - K, 0.0))


def gbm_paths(S0: float, mu: float, sigma: float, T: float, n_steps: int, n_paths: int,
              rng: np.random.Generator) -> Mat:
    """Geometric Brownian motion: an n_paths x (n_steps + 1) matrix, one row per path."""
    dt = T / n_steps
    growth = (mu - 0.5 * sigma ** 2) * dt
    shock = sigma * math.sqrt(dt)
    paths = np.zeros((n_paths, n_steps + 1))
    paths[:, 0] = S0
    for k in range(n_steps):
        z = rng.standard_normal(n_paths)  # one draw per path per step, in time order
        paths[:, k + 1] = paths[:, k] * np.exp(growth + shock * z)
    return paths


def geometric_asian_call(S0: float, K: float, r: float, sigma: float, T: float, n_fix: int) -> float:
    """Closed form for the discretely monitored geometric-average Asian call."""
    n = n_fix
    mu = math.log(S0) + (r - 0.5 * sigma ** 2) * T * (n + 1) / (2 * n)
    var = sigma ** 2 * T * (n + 1) * (2 * n + 1) / (6 * n * n)
    sd = math.sqrt(var)
    a = (mu - math.log(K) + var) / sd
    return math.exp(-r * T) * (math.exp(mu + 0.5 * var) * norm.cdf(a) - K * norm.cdf(a - sd))


def asian_call_mc(S0: float, K: float, r: float, sigma: float, T: float, n_steps: int, n_paths: int,
                  seed: int) -> tuple[float, float]:
    """Arithmetic-average Asian call, using the geometric-average option (known exactly) as a
    control variate: (price, standard error)."""
    if n_steps < 1 or n_paths < 2:
        raise ValueError("need at least one step and two paths")
    rng = np.random.default_rng(seed)
    paths = gbm_paths(S0, r, sigma, T, n_steps, n_paths, rng)
    fixings = paths[:, 1:]
    disc = math.exp(-r * T)
    pay_a = disc * np.maximum(fixings.mean(axis=1) - K, 0.0)
    pay_g = disc * np.maximum(np.exp(np.log(fixings).mean(axis=1)) - K, 0.0)
    da = pay_a - np.mean(pay_a)
    dg = pay_g - np.mean(pay_g)
    var_g = np.mean(dg * dg)
    beta = np.mean(da * dg) / var_g if var_g > 0 else 0.0
    adjusted = pay_a - beta * (pay_g - geometric_asian_call(S0, K, r, sigma, T, n_steps))
    return np.mean(adjusted), np.std(adjusted, ddof=1) / math.sqrt(n_paths)


def down_and_out_call_mc(S0: float, K: float, B: float, r: float, sigma: float, T: float,
                         n_steps: int, n_paths: int, seed: int) -> float:
    """Call that dies if the price is at or below the barrier B on any monitoring date."""
    rng = np.random.default_rng(seed)
    dt = T / n_steps
    S = np.full(n_paths, S0)
    alive = np.ones(n_paths)
    for _ in range(n_steps):
        z = rng.standard_normal(n_paths)
        S = S * np.exp((r - 0.5 * sigma ** 2) * dt + sigma * math.sqrt(dt) * z)
        alive = np.where(S <= B, 0.0, alive)
    return math.exp(-r * T) * np.mean(alive * np.maximum(S - K, 0.0))


def heston_call_mc(S0: float, K: float, r: float, v0: float, kappa: float, theta: float, xi: float,
                   rho: float, T: float, n_steps: int, n_paths: int, seed: int) -> float:
    """Heston stochastic volatility, full-truncation Euler scheme on log price."""
    if not -1.0 <= rho <= 1.0:
        raise ValueError("rho must lie in [-1, 1]")
    rng = np.random.default_rng(seed)
    dt = T / n_steps
    x = np.full(n_paths, math.log(S0))
    v = np.full(n_paths, v0)
    for _ in range(n_steps):
        z1 = rng.standard_normal(n_paths)
        z2 = rng.standard_normal(n_paths)
        w = rho * z1 + math.sqrt(1 - rho ** 2) * z2  # correlated with the price shock
        vp = np.maximum(v, 0.0)  # full truncation: negative variance is treated as zero
        x = x + (r - 0.5 * vp) * dt + np.sqrt(vp * dt) * z1
        v = v + kappa * (theta - vp) * dt + xi * np.sqrt(vp * dt) * w
    return math.exp(-r * T) * np.mean(np.maximum(np.exp(x) - K, 0.0))


def portfolio_var_mc(w: Vec, mu: Vec, cov: Mat, alpha: float, n_sims: int,
                     seed: int) -> tuple[float, float]:
    """One-period VaR and expected shortfall for multivariate normal asset returns."""
    rng = np.random.default_rng(seed)
    L = np.linalg.cholesky(cov)
    Z = rng.standard_normal((n_sims, len(w)))
    returns = Z @ L.T + mu  # each row is one scenario of correlated asset returns
    pnl = returns @ w
    var = -np.quantile(pnl, 1 - alpha)
    tail = pnl <= -var
    es = -np.sum(np.where(tail, pnl, 0.0)) / np.sum(tail)
    return var, es


_COV = np.array([[0.04, 0.012, 0.006], [0.012, 0.09, 0.018], [0.006, 0.018, 0.0225]]) / 252
_W = np.array([0.5, 0.3, 0.2])

QTRANS_CASES = {
    "european_mc": [(100.0, K, 0.05, 0.2, 1.0, n, s, c) for K in (80.0, 100.0, 120.0)
                    for n, s in ((1000, 1), (20000, 42)) for c in (True, False)] + [(100.0, 100.0, 0.05, 0.2, 1.0, 1, 7)],
    "european_from_draws": [(np.random.default_rng(3).standard_normal(5000), 100.0, K, 0.05, 0.2, 1.0)
                            for K in (90.0, 110.0)],
    "gbm_paths": [(100.0, 0.05, 0.2, 1.0, 12, 5, 11), (50.0, 0.0, 0.4, 0.5, 3, 8, 0)],
    "geometric_asian_call": [(100.0, K, 0.05, 0.3, 1.0, n) for K in (90.0, 110.0) for n in (1, 12, 252)],
    "asian_call_mc": [(100.0, K, 0.05, 0.3, 1.0, 12, 5000, 9) for K in (90.0, 100.0, 110.0)],
    "down_and_out_call_mc": [(100.0, 100.0, B, 0.05, 0.25, 1.0, 50, 4000, 5) for B in (70.0, 90.0, 99.0)],
    "heston_call_mc": [(100.0, K, 0.03, 0.04, 2.0, 0.04, 0.5, -0.7, 1.0, 50, 4000, 13) for K in (90.0, 100.0, 110.0)]
    + [(100.0, 100.0, 0.03, 0.04, 2.0, 0.04, 0.5, 1.5, 1.0, 10, 10, 1)],
    "portfolio_var_mc": [(_W, np.zeros(3), _COV, a, 20000, 21) for a in (0.95, 0.99)],
}
