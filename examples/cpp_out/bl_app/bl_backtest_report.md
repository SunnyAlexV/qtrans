# qtrans report: `bl_backtest.py`

**2 of 5 functions translated.**

## Translated

- `vol_target_scale` (line 53)
  - math.sqrt raises ValueError outside its domain in Python; C++ returns NaN
- `buy_and_hold_returns` (line 84)
  - np.errstate(...) blocks are translated as their contents: they only change which floating-point warnings NumPy prints, never the values

## Differences on overflow and error paths

These only matter on inputs where Python would raise an exception or overflow.

- division by zero gives ±inf or NaN in C++ where Python raises ZeroDivisionError (applies to: vol_target_scale)

## Not translated

- `perf_metrics` (line 14). line 14: parameter 'period_returns' has no type annotation, and no call in this file shows its type
  - Hint: write `period_returns: float` (or int / bool) for a number, or `period_returns: Vec` / `period_returns: Mat` for a 1-D / 2-D float array (from qtrans.typing import Vec, Mat)
- `_period_returns_of` (line 41). line 41: parameter 'series' has no type annotation, and no call in this file shows its type
  - Hint: write `series: float` (or int / bool) for a number, or `series: Vec` / `series: Mat` for a 1-D / 2-D float array (from qtrans.typing import Vec, Mat)
- `run_backtest` (line 114). line 114: parameter 'prices' has no type annotation, and no call in this file shows its type
  - Hint: write `prices: float` (or int / bool) for a number, or `prices: Vec` / `prices: Mat` for a 1-D / 2-D float array (from qtrans.typing import Vec, Mat)
