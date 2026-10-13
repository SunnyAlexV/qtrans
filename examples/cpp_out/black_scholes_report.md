# qtrans report: `black_scholes.py`

**10 of 10 functions translated.**

## Translated

- `d1` (line 11)
  - math.log raises ValueError outside its domain in Python; C++ returns NaN
  - math.sqrt raises ValueError outside its domain in Python; C++ returns NaN
- `d2` (line 16)
  - math.sqrt raises ValueError outside its domain in Python; C++ returns NaN
- `bs_price` (line 20)
- `delta` (line 37)
  - math.exp raises OverflowError on overflow in Python; C++ returns inf
- `gamma` (line 45)
  - math.exp raises OverflowError on overflow in Python; C++ returns inf
  - math.sqrt raises ValueError outside its domain in Python; C++ returns NaN
- `vega` (line 50)
  - math.exp raises OverflowError on overflow in Python; C++ returns inf
  - math.sqrt raises ValueError outside its domain in Python; C++ returns NaN
- `theta_per_day` (line 56)
  - math.sqrt raises ValueError outside its domain in Python; C++ returns NaN
  - math.exp raises OverflowError on overflow in Python; C++ returns inf
- `rho` (line 68)
  - math.exp raises OverflowError on overflow in Python; C++ returns inf
- `delta_gamma` (line 76)
  - math.sqrt raises ValueError outside its domain in Python; C++ returns NaN
- `parity_gap` (line 83)
  - math.exp raises OverflowError on overflow in Python; C++ returns inf

## Differences on overflow and error paths

These only matter on inputs where Python would raise an exception or overflow.

- division by zero gives ±inf or NaN in C++ where Python raises ZeroDivisionError (applies to: d1, gamma, theta_per_day, delta_gamma, parity_gap)

## Module constants

- `TRADING_DAYS: int`
- `_GRID: list[tuple[float, float, float, float, float, float]]`
