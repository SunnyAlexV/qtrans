"""Option pricing on grids: an American binomial tree with early exercise, and an explicit
finite-difference solver for the Black-Scholes PDE. Both work on whole arrays per time
step, using slices of the previous step's values."""
import numpy as np

from qtrans.typing import Vec


def american_binomial(S0: float, K: float, r: float, sigma: float, T: float, steps: int,
                      is_call: bool = False) -> float:
    """Cox-Ross-Rubinstein tree; at each node the holder takes the larger of continuation
    and immediate exercise."""
    if steps < 1:
        raise ValueError("steps must be at least 1")
    dt = T / steps
    u = np.exp(sigma * np.sqrt(dt))
    d = 1.0 / u
    p = (np.exp(r * dt) - d) / (u - d)
    if not 0 < p < 1:
        # outside (0, 1) the tree admits arbitrage and values grow like p^steps: refuse
        raise ValueError("risk-neutral probability outside (0, 1): use more steps")
    disc = np.exp(-r * dt)
    j = np.arange(steps + 1)
    S = S0 * u ** j * d ** (steps - j)  # terminal stock prices, lowest first
    V = np.maximum(S - K, 0.0) if is_call else np.maximum(K - S, 0.0)
    for _ in range(steps):
        S = S[:-1] / d  # one step back: node j had price S_j(n) / d
        cont = disc * (p * V[1:] + (1 - p) * V[:-1])
        exercise = S - K if is_call else K - S
        V = np.maximum(cont, exercise)
    return V[0]


def early_exercise_premium(S0: float, K: float, r: float, sigma: float, T: float, steps: int) -> float:
    """American minus European put on the same tree."""
    dt = T / steps
    u = np.exp(sigma * np.sqrt(dt))
    d = 1.0 / u
    p = (np.exp(r * dt) - d) / (u - d)
    if not 0 < p < 1:
        raise ValueError("risk-neutral probability outside (0, 1): use more steps")
    disc = np.exp(-r * dt)
    j = np.arange(steps + 1)
    V = np.maximum(K - S0 * u ** j * d ** (steps - j), 0.0)
    for _ in range(steps):
        V = disc * (p * V[1:] + (1 - p) * V[:-1])
    return american_binomial(S0, K, r, sigma, T, steps) - V[0]


def bs_explicit_fd(K: float, r: float, sigma: float, T: float, S_max: float, n_s: int, n_t: int,
                   is_call: bool = True) -> Vec:
    """European option values at t = 0 on the grid S_i = i * S_max / n_s, by explicit
    finite differences backwards from the payoff. Stable when dt < 1 / (sigma^2 n_s^2)."""
    dt = T / n_t
    S = np.linspace(0.0, S_max, n_s + 1)
    V = np.maximum(S - K, 0.0) if is_call else np.maximum(K - S, 0.0)
    i = np.arange(1, n_s)
    a = 0.5 * dt * (sigma ** 2 * i ** 2 - r * i)
    b = 1.0 - dt * (sigma ** 2 * i ** 2 + r)
    c = 0.5 * dt * (sigma ** 2 * i ** 2 + r * i)
    for k in range(1, n_t + 1):
        tau = k * dt
        V[1:-1] = a * V[:-2] + b * V[1:-1] + c * V[2:]  # every interior node from the old values
        if is_call:
            V[0] = 0.0
            V[-1] = S_max - K * np.exp(-r * tau)
        else:
            V[0] = K * np.exp(-r * tau)
            V[-1] = 0.0
    return V


def fd_price(S0: float, K: float, r: float, sigma: float, T: float, S_max: float, n_s: int,
             n_t: int, is_call: bool = True) -> float:
    """Price at S0, interpolating linearly between grid nodes."""
    V = bs_explicit_fd(K, r, sigma, T, S_max, n_s, n_t, is_call)
    dS = S_max / n_s
    i = int(S0 // dS)
    if i < 0 or i >= n_s:
        raise ValueError("S0 must lie inside the grid")
    w = (S0 - i * dS) / dS
    return (1 - w) * V[i] + w * V[i + 1]


QTRANS_CASES = {
    "american_binomial": [(100.0, K, 0.05, 0.2, 1.0, n, c) for K in (80.0, 100.0, 120.0)
                          for n in (1, 10, 200) for c in (True, False)] + [(100.0, 100.0, 0.05, 0.2, 1.0, 0)],
    "early_exercise_premium": [(100.0, K, 0.05, 0.25, 1.0, 100) for K in (90.0, 100.0, 110.0)],
    "bs_explicit_fd": [(100.0, 0.05, 0.2, 1.0, 300.0, 60, 400, c) for c in (True, False)]
    + [(100.0, 0.03, 0.3, 0.5, 250.0, 40, 200, True)],
    "fd_price": [(S0, 100.0, 0.05, 0.2, 1.0, 300.0, 60, 400, c) for S0 in (80.0, 100.0, 123.4)
                 for c in (True, False)] + [(350.0, 100.0, 0.05, 0.2, 1.0, 300.0, 60, 400, True)],
}
