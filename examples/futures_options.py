"""Options on forwards and futures: Black-76 (lognormal) and Bachelier (normal),
plus the Margrabe exchange option."""
import math

from scipy.stats import norm


def black76(F: float, K: float, r: float, sigma: float, T: float, is_call: bool = True) -> float:
    """Black (1976): option on a forward F, discounted at rate r."""
    sq = sigma * math.sqrt(T)
    a = math.log(F / K) / sq + 0.5 * sq
    b = a - sq
    df = math.exp(-r * T)
    if is_call:
        return df * (F * norm.cdf(a) - K * norm.cdf(b))
    return df * (K * norm.cdf(-b) - F * norm.cdf(-a))


def bachelier(F: float, K: float, r: float, sigma_n: float, T: float, is_call: bool = True) -> float:
    """Normal model, as used for rates and spreads that can go negative.
    sigma_n is an absolute (not percentage) volatility."""
    sd = sigma_n * math.sqrt(T)
    if sd == 0:
        intrinsic = F - K if is_call else K - F
        return math.exp(-r * T) * max(intrinsic, 0.0)
    m = (F - K) / sd
    w = 1 if is_call else -1
    return math.exp(-r * T) * (w * (F - K) * norm.cdf(w * m) + sd * norm.pdf(m))


def margrabe(S1: float, S2: float, sigma1: float, sigma2: float, rho: float, T: float,
             q1: float = 0.0, q2: float = 0.0) -> float:
    """Right to exchange asset 2 for asset 1 at T: payoff max(S1 - S2, 0)."""
    if not -1.0 <= rho <= 1.0:
        raise ValueError("correlation must lie in [-1, 1]")
    sigma = math.sqrt(sigma1 ** 2 + sigma2 ** 2 - 2 * rho * sigma1 * sigma2)
    sq = sigma * math.sqrt(T)
    a = (math.log(S1 / S2) + (q2 - q1 + 0.5 * sigma ** 2) * T) / sq
    return S1 * math.exp(-q1 * T) * norm.cdf(a) - S2 * math.exp(-q2 * T) * norm.cdf(a - sq)


QTRANS_CASES = {
    "black76": [(F, 100.0, 0.04, 0.3, T, c) for F in (70.0, 100.0, 130.0) for T in (0.25, 2.0)
                for c in (True, False)],
    "bachelier": [(F, 0.02, 0.03, 0.008, T, c) for F in (-0.005, 0.015, 0.02, 0.04)
                  for T in (0.0, 0.5, 10.0) for c in (True, False)],
    "margrabe": [(100.0, 95.0, 0.3, 0.2, rho, 1.0) for rho in (-0.9, 0.0, 0.5, 0.99)]
    + [(100.0, 95.0, 0.3, 0.2, 1.5, 1.0)],
}
