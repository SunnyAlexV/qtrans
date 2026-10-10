"""Instruments and a book as Python classes: what stage 5 translates.

Option and Bond are frozen dataclasses with validation in __post_init__ (Bond derives its
number of coupon periods there, with the frozen-dataclass idiom object.__setattr__). Curve is
a frozen dataclass holding arrays; Quote is a NamedTuple with a property. Portfolio is a
mutable dataclass whose methods change it, and an Account holds one. GBMEngine is a plain
class (its fields come from __init__) that owns its random generator. Functions take and
return these objects, change them in place, copy them, and configure behaviour with strings.
"""
import copy
import math
from dataclasses import dataclass, field, replace
from typing import NamedTuple

import numpy as np
from scipy.stats import norm

from qtrans.typing import Vec


@dataclass(frozen=True)
class Option:
    """A European option on a non-dividend-paying stock."""
    S: float
    K: float
    T: float
    vol: float
    r: float = 0.0
    is_call: bool = True

    def __post_init__(self):
        if self.S <= 0 or self.K <= 0 or self.T <= 0 or self.vol <= 0:
            raise ValueError("S, K, T and vol must be positive")

    @property
    def forward(self) -> float:
        return self.S * math.exp(self.r * self.T)

    def d1(self) -> float:
        return (math.log(self.forward / self.K) + 0.5 * self.vol ** 2 * self.T) / (self.vol * math.sqrt(self.T))

    def price(self) -> float:
        d1 = self.d1()
        d2 = d1 - self.vol * math.sqrt(self.T)
        df = math.exp(-self.r * self.T)
        if self.is_call:
            return df * (self.forward * norm.cdf(d1) - self.K * norm.cdf(d2))
        return df * (self.K * norm.cdf(-d2) - self.forward * norm.cdf(-d1))

    def delta(self) -> float:
        n = norm.cdf(self.d1())
        return n if self.is_call else n - 1.0

    @classmethod
    def atm(cls, S: float, T: float, vol: float, r: float = 0.0) -> "Option":
        return cls(S, S * math.exp(r * T), T, vol, r)


class Quote(NamedTuple):
    bid: float
    ask: float

    @property
    def mid(self) -> float:
        return 0.5 * (self.bid + self.ask)

    def spread_bps(self) -> float:
        return 1e4 * (self.ask - self.bid) / self.mid


@dataclass(frozen=True)
class Curve:
    """Zero rates at pillar times, linearly interpolated, flat outside the pillars."""
    times: Vec
    rates: Vec

    def __post_init__(self):
        if len(self.times) != len(self.rates):
            raise ValueError("times and rates differ in length")
        if len(self.times) == 0:
            raise ValueError("empty curve")

    def zero(self, t: float) -> float:
        ts = self.times
        rs = self.rates
        if t <= ts[0]:
            return rs[0]
        if t >= ts[-1]:
            return rs[-1]
        for i in range(1, len(ts)):
            if t <= ts[i]:
                w = (t - ts[i - 1]) / (ts[i] - ts[i - 1])
                return rs[i - 1] + w * (rs[i] - rs[i - 1])
        return rs[-1]

    def df(self, t: float) -> float:
        return math.exp(-self.zero(t) * t)


@dataclass(frozen=True)
class Bond:
    face: float
    coupon: float
    maturity: float
    freq: int = 2
    n_periods: int = field(init=False, default=0)

    def __post_init__(self):
        if self.freq <= 0 or self.maturity <= 0 or self.maturity > 100:
            raise ValueError("freq and maturity must be positive, maturity at most 100 years")
        object.__setattr__(self, "n_periods", int(round(self.maturity * self.freq)))

    def price(self, curve: Curve) -> float:
        c = self.face * self.coupon / self.freq
        pv = 0.0
        for k in range(1, self.n_periods + 1):
            pv += c * curve.df(k / self.freq)
        return pv + self.face * curve.df(self.n_periods / self.freq)


@dataclass
class Portfolio:
    weights: Vec
    cash: float = 0.0
    trades: int = field(default=0, init=False)

    def value(self, prices: Vec) -> float:
        return float(self.weights @ prices) + self.cash

    def rebalance(self, target: Vec, prices: Vec, cost_bps: float) -> float:
        """Move to the target weights, paying cost_bps on the turnover out of cash."""
        turnover = float(np.sum(np.abs(target - self.weights)))
        cost = turnover * cost_bps * 1e-4 * self.value(prices)
        self.weights = target.copy()
        self.cash -= cost
        self.trades += 1
        return cost

    def scale(self, k: float) -> None:
        self.weights *= k


@dataclass
class Account:
    port: Portfolio
    name: str = "main"
    stance: str = "Long only"

    def nav(self, prices: Vec) -> float:
        return self.port.value(prices)


class GBMEngine:
    """Monte Carlo under geometric Brownian motion; owns its random generator."""

    def __init__(self, s0: float, r: float, sigma: float, seed: int, steps_per_year: int = 252):
        if s0 <= 0 or sigma < 0 or steps_per_year <= 0:
            raise ValueError("need s0 > 0, sigma >= 0, steps_per_year > 0")
        self.s0 = s0
        self.r = r
        self.sigma = sigma
        self.dt = 1.0 / steps_per_year
        self.rng = np.random.default_rng(seed)
        self.paths_run = 0

    def terminal(self, n_paths: int, years: float) -> Vec:
        if years <= 0 or years > 10 or n_paths < 1 or n_paths > 100_000:
            raise ValueError("need 0 < years <= 10 and 1 <= n_paths <= 100000")
        steps = max(1, int(round(years / self.dt)))
        h = years / steps
        drift = (self.r - 0.5 * self.sigma ** 2) * h
        vol = self.sigma * math.sqrt(h)
        logs = np.zeros(n_paths)
        for _ in range(steps):
            z = self.rng.standard_normal(n_paths)
            logs += drift + vol * z
        self.paths_run += n_paths
        return self.s0 * np.exp(logs)

    def call(self, K: float, years: float, n_paths: int) -> float:
        st = self.terminal(n_paths, years)
        return math.exp(-self.r * years) * float(np.mean(np.maximum(st - K, 0.0)))


def mc_vs_black_scholes(seed: int, K: float, years: float) -> tuple[float, float, int]:
    """A Monte Carlo call price next to the closed form, and how many paths the engine ran."""
    eng = GBMEngine(100.0, 0.04, 0.2, seed, 52)
    mc = eng.call(K, years, 4000)
    exact = Option(100.0, K, years, 0.2, 0.04).price()
    return mc, exact, eng.paths_run


def reprice(eng: GBMEngine, K: float) -> float:
    """Draws from the caller's engine: its generator moves on, as in Python."""
    return eng.call(K, 0.25, 2000)


def weight_bounds(stance: str, cap: float) -> tuple[float, float]:
    """The box each weight must stay in, for a portfolio stance."""
    if stance == "Long only":
        return 0.0, cap
    if stance == "Short only":
        return -cap, 0.0
    if stance in ("Long-short", "Market neutral"):
        return -cap, cap
    raise ValueError("unknown stance")


def book_value(options: Option, qty: float) -> float:
    return qty * options.price()


def bumped_delta(o: Option, h: float) -> float:
    """Central-difference delta by repricing bumped copies: should match o.delta()."""
    up = replace(o, S=o.S + h)
    dn = replace(o, S=o.S - h)
    return (up.price() - dn.price()) / (2.0 * h)


def straddle(S: float, T: float, vol: float, r: float) -> float:
    c = Option.atm(S, T, vol, r)
    p = replace(c, is_call=False)
    return c.price() + p.price()


def quote_stats(q: Quote) -> tuple[float, float, float]:
    bid, ask = q
    return q.mid, q.spread_bps(), ask - bid


def bond_price(b: Bond, curve: Curve) -> float:
    return b.price(curve)


def parallel_shift(curve: Curve, h: float) -> Curve:
    return Curve(curve.times, curve.rates + h)


def dv01(b: Bond, curve: Curve) -> float:
    return (b.price(parallel_shift(curve, -1e-4)) - b.price(parallel_shift(curve, 1e-4))) / 2.0


def run_rebalances(p: Portfolio, target: Vec, prices: Vec, n: int) -> float:
    total = 0.0
    for _ in range(n):
        total += p.rebalance(target, prices, 5.0)
    return total


def scaled_copy(p: Portfolio, k: float) -> Portfolio:
    q = copy.deepcopy(p)
    q.scale(k)
    return q


def leveraged_nav(acc: Account, prices: Vec, k: float) -> float:
    lo, hi = weight_bounds(acc.stance, 1.0)
    a = copy.deepcopy(acc)
    a.port.scale(k)
    a.port.weights = np.clip(a.port.weights, lo, hi)
    return a.nav(prices)


# Verifier inputs (names starting QTRANS_ are not translated)
QTRANS_CURVE = Curve(np.array([0.5, 1.0, 2.0, 5.0, 10.0, 30.0]),
                     np.array([0.0450, 0.0435, 0.0410, 0.0395, 0.0405, 0.0430]))
QTRANS_W = np.array([0.35, 0.25, 0.20, 0.15, 0.05])
QTRANS_PX = np.array([101.0, 55.5, 230.0, 12.25, 87.0])
QTRANS_CASES = {
    "weight_bounds": [("Long only", 0.1), ("Short only", 0.1), ("Market neutral", 0.2), ("long only", 0.1)],
    "book_value": [(Option(100.0, 95.0, 0.5, 0.2, 0.04), 10.0), (Option(100.0, 120.0, 2.0, 0.35, 0.04, False), 3.0)],
    "bumped_delta": [(Option(100.0, 100.0, 1.0, 0.2, 0.04), 0.01)],
    "straddle": [(100.0, 0.25, 0.2, 0.04)],
    "quote_stats": [(Quote(99.98, 100.02),)],
    "bond_price": [(Bond(100.0, 0.05, 7.0), QTRANS_CURVE), (Bond(1000.0, 0.04, 30.0, 4), QTRANS_CURVE),
                   (Bond(100.0, 0.0, 0.5, 1), QTRANS_CURVE)],
    "dv01": [(Bond(100.0, 0.05, 10.0), QTRANS_CURVE)],
    "parallel_shift": [(QTRANS_CURVE, 0.0025)],
    "run_rebalances": [(Portfolio(QTRANS_W.copy(), 1000.0), np.full(5, 0.2), QTRANS_PX, 3)],
    "scaled_copy": [(Portfolio(QTRANS_W.copy(), 5.0), 1.5)],
    "mc_vs_black_scholes": [(1, 100.0, 1.0), (2, 110.0, 0.5)],
    "reprice": [(GBMEngine(100.0, 0.04, 0.3, 11), 100.0)],
    "leveraged_nav": [(Account(Portfolio(QTRANS_W.copy(), 50.0)), QTRANS_PX, 2.0),
                      (Account(Portfolio(QTRANS_W.copy()), stance="Long-short"), QTRANS_PX, -1.0)],
}
