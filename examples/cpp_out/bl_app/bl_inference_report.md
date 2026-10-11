# qtrans report: `bl_inference.py`

**6 of 10 functions translated.**

## Translated

- `default_block_length` (line 22)
  - x ** y with a negative x and non-integer y gives a complex number for a Python float but NaN in C++ (and for np.float64)
  - float -> int beyond ±9.2e18 raises OverflowError in C++; Python returns a big integer
- `_ann_mean` (line 49)
- `_ann_sharpe` (line 53)
  - np.errstate(...) blocks are translated as their contents: they only change which floating-point warnings NumPy prints, never the values
  - math.sqrt raises ValueError outside its domain in Python; C++ returns NaN
- `newey_west_lag` (line 104)
  - x ** y with a negative x and non-integer y gives a complex number for a Python float but NaN in C++ (and for np.float64)
  - float -> int beyond ±9.2e18 raises OverflowError in C++; Python returns a big integer
- `hac_cov` (line 109)
- `hit_rate_pvalue` (line 187)
  - math.sqrt raises ValueError outside its domain in Python; C++ returns NaN

## Differences on overflow and error paths

These only matter on inputs where Python would raise an exception or overflow.

- float ** raises OverflowError in Python when the result overflows; C++ returns inf (applies to: default_block_length, newey_west_lag)
- division by zero gives ±inf or NaN in C++ where Python raises ZeroDivisionError (applies to: _ann_sharpe, hac_cov, hit_rate_pvalue)
- int arithmetic that could leave 64 bits is checked: where Python's unbounded int would exceed ±9.2e18, C++ raises OverflowError (applies to: hac_cov)

## Not translated

- `stationary_bootstrap_indices` (line 28). line 28: parameter 'T' has no type annotation
  - Hint: write `T: float` (or int / bool) for a number, or `T: Vec` / `T: Mat` for a 1-D / 2-D float array (from qtrans.typing import Vec, Mat)
- `paired_bootstrap` (line 59). line 59: parameter 'a' has no type annotation
  - Hint: write `a: float` (or int / bool) for a number, or `a: Vec` / `a: Mat` for a 1-D / 2-D float array (from qtrans.typing import Vec, Mat)
- `sharpe_diff_test` (line 122). line 122: parameter 'a' has no type annotation
  - Hint: write `a: float` (or int / bool) for a number, or `a: Vec` / `a: Mat` for a 1-D / 2-D float array (from qtrans.typing import Vec, Mat)
- `breakeven_cost_bps` (line 162). line 162: parameter 'bt_a' has no type annotation
  - Hint: write `bt_a: float` (or int / bool) for a number, or `bt_a: Vec` / `bt_a: Mat` for a 1-D / 2-D float array (from qtrans.typing import Vec, Mat)
