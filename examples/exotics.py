"""Closed-form and lattice prices for a few non-vanilla payoffs."""
import math

from scipy.stats import norm


def digital_call(S: float, K: float, r: float, sigma: float, T: float, cash: float = 1.0) -> float:
    """Cash-or-nothing call: pays `cash` if S_T > K."""
    sq = sigma * math.sqrt(T)
    b = (math.log(S / K) + (r - 0.5 * sigma * sigma) * T) / sq
    return cash * math.exp(-r * T) * norm.cdf(b)


def geometric_asian_call(S: float, K: float, r: float, sigma: float, T: float, n_fix: int) -> float:
    """Discretely monitored geometric-average Asian call (Kemna-Vorst style closed form)
    with n_fix equally spaced fixings ending at T."""
    if n_fix < 1:
        raise ValueError("need at least one fixing")
    n = n_fix
    # mean and variance of the log of the geometric average
    mu = math.log(S) + (r - 0.5 * sigma ** 2) * T * (n + 1) / (2 * n)
    var = sigma ** 2 * T * (n + 1) * (2 * n + 1) / (6 * n * n)
    sd = math.sqrt(var)
    a = (mu - math.log(K) + var) / sd
    return math.exp(-r * T) * (math.exp(mu + 0.5 * var) * norm.cdf(a) - K * norm.cdf(a - sd))


def crr_european(S: float, K: float, r: float, sigma: float, T: float, steps: int,
                 is_call: bool = True) -> float:
    """Cox-Ross-Rubinstein binomial price of a European option, summed directly over
    terminal nodes. Binomial weights are built in log space so 1000+ steps don't overflow."""
    if steps < 1:
        raise ValueError("steps must be at least 1")
    dt = T / steps
    u = math.exp(sigma * math.sqrt(dt))
    d = 1 / u
    p = (math.exp(r * dt) - d) / (u - d)
    if not 0 < p < 1:
        raise ValueError("risk-neutral probability outside (0, 1): reduce the step size")
    log_p, log_q = math.log(p), math.log1p(-p)
    total = 0.0
    log_choose = 0.0  # log C(steps, k), updated incrementally
    for k in range(steps + 1):
        if k > 0:
            log_choose += math.log(steps - k + 1) - math.log(k)
        ST = S * u ** k * d ** (steps - k)
        payoff = max(ST - K, 0.0) if is_call else max(K - ST, 0.0)
        if payoff > 0:
            total += math.exp(log_choose + k * log_p + (steps - k) * log_q) * payoff
    return math.exp(-r * T) * total


QTRANS_CASES = {
    "digital_call": [(100.0, K, 0.05, 0.25, 1.0) for K in (60.0, 100.0, 150.0)],
    "geometric_asian_call": [(100.0, K, 0.05, 0.3, 1.0, n) for K in (90.0, 100.0, 110.0)
                             for n in (1, 12, 252)] + [(100.0, 100.0, 0.05, 0.3, 1.0, 0)],
    "crr_european": [(100.0, K, 0.05, 0.2, 1.0, n, c) for K in (80.0, 100.0, 120.0)
                     for n in (1, 50, 1000) for c in (True, False)],
}
