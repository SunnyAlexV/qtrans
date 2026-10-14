"""A small option pricer: Black-Scholes, its vega, implied volatility, and a book of options."""
import math

import numpy as np
from scipy.special import ndtr

from qtrans.typing import Vec


def bs_call(S: float, K: float, T: float, vol: float, r: float) -> float:
    """Black-Scholes price of a European call."""
    if S <= 0.0 or K <= 0.0:
        raise ValueError("spot and strike must be positive")
    sd = vol * math.sqrt(T)
    d1 = (math.log(S / K) + (r + 0.5 * vol * vol) * T) / sd
    return S * ndtr(d1) - K * math.exp(-r * T) * ndtr(d1 - sd)


def vega(S: float, K: float, T: float, vol: float, r: float) -> float:
    """dPrice/dvol."""
    sd = vol * math.sqrt(T)
    d1 = (math.log(S / K) + (r + 0.5 * vol * vol) * T) / sd
    return S * math.sqrt(T) * math.exp(-0.5 * d1 * d1) / math.sqrt(2.0 * math.pi)


def implied_vol(price: float, S: float, K: float, T: float, r: float) -> float:
    """The volatility that reprices the call, by Newton's method from 20%."""
    vol = 0.2
    for _ in range(100):
        diff = bs_call(S, K, T, vol, r) - price
        if abs(diff) < 1e-12:
            return vol
        vol -= diff / vega(S, K, T, vol, r)
    raise ValueError("implied_vol did not converge")


def book(S: Vec, K: Vec, T: Vec, vol: Vec, r: float) -> Vec:
    """Prices for a book of calls."""
    out = np.zeros(len(S))
    for i in range(len(S)):
        out[i] = bs_call(float(S[i]), float(K[i]), float(T[i]), float(vol[i]), r)
    return out


_g = np.random.default_rng(3)
QTRANS_CASES = {
    "bs_call": [(100.0, 95.0, 1.0, 0.2, 0.03), (100.0, 95.0, 0.0, 0.2, 0.03), (-1.0, 95.0, 1.0, 0.2, 0.03)],
    "vega": [(100.0, 95.0, 1.0, 0.2, 0.03)],
    "implied_vol": [(10.0, 100.0, 95.0, 1.0, 0.03), (1e-9, 100.0, 95.0, 1.0, 0.03)],
    "book": [(100.0 + 10.0 * _g.standard_normal(1000), np.full(1000, 100.0), np.full(1000, 1.0),
              np.full(1000, 0.2), 0.03)],
}
