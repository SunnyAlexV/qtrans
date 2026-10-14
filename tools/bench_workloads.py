"""Workloads for tools/benchmark.py: what a desk would run, not one call on one input.

The same book of options is priced three ways, so the C++ is compared with a fair Python
baseline: a plain loop over a scalar pricer (how much Python code starts out), the same loop
compiled, and NumPy-vectorised Python (how fast Python code is written) against its C++. Then a
vectorised Monte Carlo and a Newton yield solve over a book of bonds. Every function is checked
by qtrans verify like any other before it is timed.
"""
import math

import numpy as np
from scipy.special import ndtr
from scipy.stats import norm

from qtrans.typing import Vec


def bs_call(S: float, K: float, T: float, vol: float, r: float) -> float:
    """Black-Scholes call, one option."""
    sd = vol * math.sqrt(T)
    d1 = (math.log(S / K) + (r + 0.5 * vol * vol) * T) / sd
    return S * ndtr(d1) - K * math.exp(-r * T) * ndtr(d1 - sd)


def book_loop(S: Vec, K: Vec, T: Vec, vol: Vec, r: float) -> Vec:
    """The book, one option at a time."""
    out = np.zeros(len(S))
    for i in range(len(S)):
        out[i] = bs_call(float(S[i]), float(K[i]), float(T[i]), float(vol[i]), r)
    return out


def book_numpy(S: Vec, K: Vec, T: Vec, vol: Vec, r: float) -> Vec:
    """The book, vectorised: NumPy and SciPy work on whole arrays."""
    sd = vol * np.sqrt(T)
    d1 = (np.log(S / K) + (r + 0.5 * vol * vol) * T) / sd
    return S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d1 - sd)


def mc_call(s0: float, k: float, r: float, vol: float, T: float, n_paths: int, seed: int) -> float:
    """European call by Monte Carlo, vectorised over the paths."""
    rng = np.random.default_rng(seed)
    z = rng.standard_normal(n_paths)
    sT = s0 * np.exp((r - 0.5 * vol * vol) * T + vol * math.sqrt(T) * z)
    return math.exp(-r * T) * float(np.mean(np.maximum(sT - k, 0.0)))


def mc_draws(n_paths: int, seed: int) -> float:
    """mc_call's normal draws alone."""
    rng = np.random.default_rng(seed)
    return float(np.sum(rng.standard_normal(n_paths)))


def mc_exp(z: Vec) -> float:
    """mc_call's np.exp alone."""
    return float(np.sum(np.exp(z)))


def ytm(price: float, coupon: float, years: int) -> float:
    """Yield to maturity of an annual-coupon bond, face 100, by Newton's method."""
    y = 0.05
    for _ in range(50):
        pv = 0.0
        dpv = 0.0
        for t in range(1, years + 1):
            cf = coupon * 100.0 + (100.0 if t == years else 0.0)
            pv += cf / (1.0 + y) ** t
            dpv -= t * cf / (1.0 + y) ** (t + 1)
        step = (pv - price) / dpv
        y -= step
        if abs(step) < 1e-12:
            break
    return y


def ytm_book(prices: Vec, coupons: Vec, years: int) -> Vec:
    """Yields for a book of bonds."""
    out = np.zeros(len(prices))
    for i in range(len(prices)):
        out[i] = ytm(float(prices[i]), float(coupons[i]), years)
    return out


_g = np.random.default_rng(7)
_N = 10_000
_S = 100.0 * np.exp(0.1 * _g.standard_normal(_N))
_K = 100.0 * np.exp(0.1 * _g.standard_normal(_N))
_T = 0.25 + 2.0 * _g.random(_N)
_V = 0.1 + 0.3 * _g.random(_N)
_Z = _g.standard_normal(100_000)
_P = 90.0 + 20.0 * _g.random(1_000)
_C = 0.02 + 0.04 * _g.random(1_000)

QTRANS_CASES = {
    "bs_call": [(100.0, 95.0, 1.0, 0.2, 0.03)],
    "book_loop": [(_S, _K, _T, _V, 0.03)],
    "book_numpy": [(_S, _K, _T, _V, 0.03)],
    "mc_call": [(100.0, 100.0, 0.03, 0.2, 1.0, 100_000, 42)],
    "mc_draws": [(100_000, 42)],
    "mc_exp": [(_Z,)],
    "ytm": [(97.0, 0.04, 10)],
    "ytm_book": [(_P, _C, 10)],
}
# what each timed input is, for the report
WORKLOADS = {
    "bs_call": "one option",
    "book_loop": "10,000 options, a Python loop over bs_call",
    "book_numpy": "10,000 options, NumPy-vectorised",
    "mc_call": "100,000 paths, NumPy-vectorised",
    "mc_draws": "mc_call's 100,000 normal draws alone",
    "mc_exp": "mc_call's np.exp of 100,000 values alone",
    "ytm": "one 10-year bond, Newton",
    "ytm_book": "1,000 10-year bonds, a loop over ytm",
}
