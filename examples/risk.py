"""Risk and performance statistics from a series of returns: historical VaR and expected
shortfall, EWMA (RiskMetrics) volatility, drawdown, Sharpe ratio, rolling means."""
import numpy as np

from qtrans.typing import Vec


def historical_var(returns: Vec, alpha: float = 0.99) -> float:
    """Loss not exceeded with probability alpha, from the empirical distribution."""
    return -np.quantile(returns, 1 - alpha)


def expected_shortfall(returns: Vec, alpha: float = 0.99) -> float:
    """Average loss over the returns at or below the (1 - alpha) quantile."""
    q = np.quantile(returns, 1 - alpha)
    tail = np.where(returns <= q, returns, 0.0)
    return -np.sum(tail) / np.sum(returns <= q)


def ewma_vol(returns: Vec, lam: float = 0.94) -> Vec:
    """RiskMetrics volatility: var_t = lam * var_{t-1} + (1 - lam) * r_{t-1}^2."""
    var = np.zeros(len(returns))
    if len(returns) == 0:
        return var
    var[0] = returns[0] ** 2
    for t in range(1, len(returns)):
        var[t] = lam * var[t - 1] + (1 - lam) * returns[t - 1] ** 2
    return np.sqrt(var)


def max_drawdown(prices: Vec) -> float:
    """Largest fall from a running peak, as a fraction of that peak."""
    peak = prices[0]
    worst = 0.0
    for p in prices:
        if p > peak:
            peak = p
        dd = (peak - p) / peak
        if dd > worst:
            worst = dd
    return worst


def sharpe_ratio(returns: Vec, rf: float = 0.0, periods: int = 252) -> float:
    excess = returns - rf / periods
    return np.mean(excess) / np.std(excess, ddof=1) * np.sqrt(periods)


def wealth_curve(returns: Vec, start: float = 1.0) -> Vec:
    return start * np.cumprod(1.0 + returns)


def rolling_mean(x: Vec, window: int) -> Vec:
    """Mean of every full window, from cumulative sums."""
    if window < 1 or window > len(x):
        raise ValueError("window must be between 1 and len(x)")
    c = np.cumsum(x)
    out = np.zeros(len(x) - window + 1)
    out[0] = c[window - 1] / window
    out[1:] = (c[window:] - c[:-window]) / window
    return out


def downside_deviation(returns: Vec, target: float = 0.0) -> float:
    shortfall = np.minimum(returns - target, 0.0)
    return np.sqrt(np.mean(shortfall ** 2))


_rng = np.random.default_rng(3)
_R = _rng.standard_t(4, size=1000) * 0.01           # fat-tailed daily returns
_PRICES = 100 * np.cumprod(1 + _R)

QTRANS_CASES = {
    "historical_var": [(_R, a) for a in (0.9, 0.95, 0.99, 0.999)] + [(_R[:7], 0.95)],
    "expected_shortfall": [(_R, a) for a in (0.9, 0.95, 0.99)],
    "ewma_vol": [(_R,), (_R[:1],), (np.array([]),)],
    "max_drawdown": [(_PRICES,), (np.array([1.0, 2.0, 3.0]),), (np.array([]),)],
    "sharpe_ratio": [(_R,), (_R, 0.03), (_R, 0.03, 12)],
    "wealth_curve": [(_R,), (_R[:10], 250.0)],
    "rolling_mean": [(_R, w) for w in (1, 5, 21, 252, 1000)] + [(_R, 0), (_R, 1001)],
    "downside_deviation": [(_R,), (_R, 0.001)],
}
