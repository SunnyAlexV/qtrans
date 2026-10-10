"""Implied volatility of a European option: Newton-Raphson on vega, with a bisection
fallback when Newton leaves the bracket or vega is too small to divide by."""
import math

from scipy.stats import norm

SIGMA_LO = 1e-6
SIGMA_HI = 5.0
TOL = 1e-12
MAX_NEWTON = 50
MAX_BISECT = 200


def bs_price(S: float, K: float, r: float, sigma: float, T: float, is_call: bool) -> float:
    sq = sigma * math.sqrt(T)
    a = (math.log(S / K) + (r + 0.5 * sigma * sigma) * T) / sq
    b = a - sq
    if is_call:
        return S * norm.cdf(a) - K * math.exp(-r * T) * norm.cdf(b)
    return K * math.exp(-r * T) * norm.cdf(-b) - S * norm.cdf(-a)


def bs_vega(S: float, K: float, r: float, sigma: float, T: float) -> float:
    sq = sigma * math.sqrt(T)
    a = (math.log(S / K) + (r + 0.5 * sigma * sigma) * T) / sq
    return S * norm.pdf(a) * math.sqrt(T)


def implied_vol(price: float, S: float, K: float, r: float, T: float, is_call: bool = True) -> float:
    """Volatility sigma such that bs_price(..., sigma, ...) == price."""
    if S <= 0 or K <= 0 or T <= 0:
        raise ValueError("S, K and T must be positive")
    disc_k = K * math.exp(-r * T)
    lower = max(S - disc_k, 0.0) if is_call else max(disc_k - S, 0.0)
    upper = S if is_call else disc_k
    if not lower < price < upper:
        raise ValueError("price violates no-arbitrage bounds")

    # Brenner-Subrahmanyam starting point, kept inside the bracket
    sigma = math.sqrt(2 * math.pi / T) * price / S
    sigma = min(max(sigma, 0.05), 2.0)
    lo, hi = SIGMA_LO, SIGMA_HI
    for _ in range(MAX_NEWTON):
        diff = bs_price(S, K, r, sigma, T, is_call) - price
        if abs(diff) <= TOL * price:  # relative: an absolute test accepts anything for tiny prices
            return sigma
        # keep a bracket so bisection can take over if needed (price rises with sigma)
        if diff > 0:
            hi = sigma
        else:
            lo = sigma
        v = bs_vega(S, K, r, sigma, T)
        if v < 1e-10:
            break
        step = diff / v
        nxt = sigma - step
        if nxt <= lo or nxt >= hi:
            break  # Newton left the bracket
        sigma = nxt

    for _ in range(MAX_BISECT):
        mid = 0.5 * (lo + hi)
        diff = bs_price(S, K, r, mid, T, is_call) - price
        if abs(diff) <= TOL * price or hi - lo < 1e-15:
            return mid
        if diff > 0:
            hi = mid
        else:
            lo = mid
    raise RuntimeError("implied volatility did not converge")


# Prices generated from known vols, so a correct solver returns those vols. Cases with
# vega < 1e-3 are left out: there the price barely depends on sigma (deep in the money at
# low vol, or so far out of the money the price underflows), so the implied vol is not
# identifiable to many digits by any implementation. A one-ULP change in the input price
# then moves the answer in the 7th digit, in Python as much as in C++.
QTRANS_CASES = {
    "implied_vol": [(bs_price(100.0, K, 0.03, v, T, c), 100.0, K, 0.03, T, c)
                    for K in (50.0, 80.0, 100.0, 120.0, 200.0)
                    for v in (0.05, 0.2, 0.6, 1.5)
                    for T in (0.02, 0.5, 3.0)
                    for c in (True, False)
                    if bs_vega(100.0, K, 0.03, v, T) >= 1e-3],
    "bs_price": [(100.0, 100.0, 0.05, 0.2, 1.0, True), (100.0, 100.0, 0.05, 0.2, 1.0, False)],
}
