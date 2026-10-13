# qtrans report: `monte_carlo.py`

**8 of 8 functions translated.**

## Translated

- `european_mc` (line 13)
  - math.sqrt raises ValueError outside its domain in Python; C++ returns NaN
  - math.exp raises OverflowError on overflow in Python; C++ returns inf
- `european_from_draws` (line 32)
  - math.sqrt raises ValueError outside its domain in Python; C++ returns NaN
  - math.exp raises OverflowError on overflow in Python; C++ returns inf
- `gbm_paths` (line 39)
  - math.sqrt raises ValueError outside its domain in Python; C++ returns NaN
- `geometric_asian_call` (line 53)
  - math.log raises ValueError outside its domain in Python; C++ returns NaN
  - math.sqrt raises ValueError outside its domain in Python; C++ returns NaN
  - math.exp raises OverflowError on overflow in Python; C++ returns inf
- `asian_call_mc` (line 63)
  - math.exp raises OverflowError on overflow in Python; C++ returns inf
  - math.sqrt raises ValueError outside its domain in Python; C++ returns NaN
- `down_and_out_call_mc` (line 83)
  - math.sqrt raises ValueError outside its domain in Python; C++ returns NaN
  - math.exp raises OverflowError on overflow in Python; C++ returns inf
- `heston_call_mc` (line 97)
  - math.log raises ValueError outside its domain in Python; C++ returns NaN
  - math.sqrt raises ValueError outside its domain in Python; C++ returns NaN
  - math.exp raises OverflowError on overflow in Python; C++ returns inf
- `portfolio_var_mc` (line 116)

## Differences on overflow and error paths

These only matter on inputs where Python would raise an exception or overflow.

- float ** raises OverflowError in Python when the result overflows; C++ returns inf (applies to: european_mc, european_from_draws, gbm_paths, geometric_asian_call, down_and_out_call_mc, heston_call_mc)
- division by zero gives ±inf or NaN in C++ where Python raises ZeroDivisionError (applies to: european_mc, gbm_paths, geometric_asian_call, asian_call_mc, down_and_out_call_mc, heston_call_mc, portfolio_var_mc)
- int arithmetic that could leave 64 bits is checked: where Python's unbounded int would exceed ±9.2e18, C++ raises OverflowError (applies to: gbm_paths, geometric_asian_call)

## Module-level notes

- line 130: `_COV` not translated: Mat constants are not supported
- line 131: `_W` not translated: Vec constants are not supported
