# qtrans report: `black_scholes.py`

**10 of 10 functions translated.**

## Translated

- `d1` (line 11)
- `d2` (line 16)
- `bs_price` (line 20)
- `delta` (line 37)
- `gamma` (line 45)
- `vega` (line 50)
- `theta_per_day` (line 56)
- `rho` (line 68)
- `delta_gamma` (line 76)
- `parity_gap` (line 83)
  - `(c - p - (S * math.exp(-q * T) - K * math.exp(-r * T))) / (S + K)` may hold a NumPy number or a Python float, depending on the run: Python then gives inf / NaN (NumPy) or raises (ZeroDivisionError, OverflowError); the C++ gives inf / NaN

## Module constants

- `TRADING_DAYS: int`
- `_GRID: list[tuple[float, float, float, float, float, float]]`
