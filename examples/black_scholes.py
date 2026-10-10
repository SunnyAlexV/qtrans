"""Black-Scholes-Merton prices and Greeks for European options on a stock paying a
continuous dividend yield q."""
import math

import numpy as np
from scipy.stats import norm

TRADING_DAYS = 252


def d1(S: float, K: float, r: float, sigma: float, T: float, q: float = 0.0) -> float:
    """Standardised moneyness term d1."""
    return (math.log(S / K) + (r - q + 0.5 * sigma ** 2) * T) / (sigma * math.sqrt(T))


def d2(S: float, K: float, r: float, sigma: float, T: float, q: float = 0.0) -> float:
    return d1(S, K, r, sigma, T, q) - sigma * math.sqrt(T)


def bs_price(S: float, K: float, r: float, sigma: float, T: float, q: float = 0.0,
             is_call: bool = True) -> float:
    """Price of a European call (is_call=True) or put."""
    if S <= 0 or K <= 0:
        raise ValueError("spot and strike must be positive")
    if T <= 0 or sigma <= 0:
        # expired, or no uncertainty left: discounted intrinsic value of the forward
        fwd = S * np.exp(-q * max(T, 0.0))
        disc = K * np.exp(-r * max(T, 0.0))
        return max(fwd - disc, 0.0) if is_call else max(disc - fwd, 0.0)
    a = d1(S, K, r, sigma, T, q)
    b = a - sigma * np.sqrt(T)
    if is_call:
        return S * np.exp(-q * T) * norm.cdf(a) - K * np.exp(-r * T) * norm.cdf(b)
    return K * np.exp(-r * T) * norm.cdf(-b) - S * np.exp(-q * T) * norm.cdf(-a)


def delta(S: float, K: float, r: float, sigma: float, T: float, q: float = 0.0,
          is_call: bool = True) -> float:
    a = d1(S, K, r, sigma, T, q)
    if is_call:
        return math.exp(-q * T) * norm.cdf(a)
    return -math.exp(-q * T) * norm.cdf(-a)


def gamma(S: float, K: float, r: float, sigma: float, T: float, q: float = 0.0) -> float:
    a = d1(S, K, r, sigma, T, q)
    return math.exp(-q * T) * norm.pdf(a) / (S * sigma * math.sqrt(T))


def vega(S: float, K: float, r: float, sigma: float, T: float, q: float = 0.0) -> float:
    """Sensitivity to a one-unit change in volatility (not per 1%)."""
    a = d1(S, K, r, sigma, T, q)
    return S * math.exp(-q * T) * norm.pdf(a) * math.sqrt(T)


def theta_per_day(S: float, K: float, r: float, sigma: float, T: float, q: float = 0.0,
                  is_call: bool = True) -> float:
    a = d1(S, K, r, sigma, T, q)
    b = a - sigma * math.sqrt(T)
    decay = -S * math.exp(-q * T) * norm.pdf(a) * sigma / (2 * math.sqrt(T))
    if is_call:
        annual = decay - r * K * math.exp(-r * T) * norm.cdf(b) + q * S * math.exp(-q * T) * norm.cdf(a)
    else:
        annual = decay + r * K * math.exp(-r * T) * norm.cdf(-b) - q * S * math.exp(-q * T) * norm.cdf(-a)
    return annual / TRADING_DAYS


def rho(S: float, K: float, r: float, sigma: float, T: float, q: float = 0.0,
        is_call: bool = True) -> float:
    b = d2(S, K, r, sigma, T, q)
    if is_call:
        return K * T * math.exp(-r * T) * norm.cdf(b)
    return -K * T * math.exp(-r * T) * norm.cdf(-b)


def delta_gamma(S: float, K: float, r: float, sigma: float, T: float) -> tuple[float, float]:
    """Both first- and second-order spot sensitivities of a call in one pass."""
    a = d1(S, K, r, sigma, T)
    n = norm.pdf(a)
    return norm.cdf(a), n / (S * sigma * math.sqrt(T))


def parity_gap(S: float, K: float, r: float, sigma: float, T: float, q: float = 0.0) -> float:
    """Put-call parity error (C - P - (S e^{-qT} - K e^{-rT})) relative to S + K.

    The raw gap is a difference of numbers of size ~S and ~K whose exact value is 0, so it is
    rounding noise of order (S + K) * 1e-16. Dividing by S + K makes it dimensionless: a
    consistent pricer gives about 1e-16 whatever the scale of the inputs."""
    c = bs_price(S, K, r, sigma, T, q, True)
    p = bs_price(S, K, r, sigma, T, q, is_call=False)
    return (c - p - (S * math.exp(-q * T) - K * math.exp(-r * T))) / (S + K)


# Hand-picked cases for `qtrans verify`, on top of its random inputs: ATM, deep ITM/OTM,
# short and long maturities, high vol, dividends, and the error / expiry branches.
_GRID = [
    (100.0, 100.0, 0.05, 0.20, 1.0, 0.0),
    (100.0, 60.0, 0.03, 0.25, 0.5, 0.01),
    (100.0, 160.0, 0.03, 0.25, 0.5, 0.01),
    (42.0, 40.0, 0.10, 0.20, 0.5, 0.0),
    (100.0, 100.0, 0.0, 0.80, 5.0, 0.02),
    (100.0, 100.0, 0.05, 0.05, 1.0 / 365, 0.0),
    (2500.0, 2400.0, 0.045, 0.18, 30.0, 0.015),
]
QTRANS_CASES = {
    "bs_price": [g + (c,) for g in _GRID for c in (True, False)]
    + [(100.0, 100.0, 0.05, 0.2, 0.0, 0.0, True), (100.0, 90.0, 0.05, 0.0, 1.0, 0.0, False),
       (-1.0, 100.0, 0.05, 0.2, 1.0, 0.0, True)],
    "delta": [g + (c,) for g in _GRID for c in (True, False)],
    "gamma": _GRID,
    "vega": _GRID,
    "theta_per_day": [g + (c,) for g in _GRID for c in (True, False)],
    "rho": [g + (c,) for g in _GRID for c in (True, False)],
    "delta_gamma": [g[:5] for g in _GRID],
    "parity_gap": _GRID,
}
