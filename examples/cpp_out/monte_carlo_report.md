# qtrans report: `monte_carlo.py`

**8 of 8 functions translated.**

## Translated

- `european_mc` (line 13)
- `european_from_draws` (line 32)
- `gbm_paths` (line 39)
- `geometric_asian_call` (line 53)
- `asian_call_mc` (line 63)
- `down_and_out_call_mc` (line 83)
- `heston_call_mc` (line 97)
- `portfolio_var_mc` (line 116)

## Differences on overflow and error paths

These only matter on inputs where Python would raise an exception or overflow.

- int arithmetic that could leave 64 bits is checked: where Python's unbounded int would exceed ±9.2e18, C++ raises OverflowError (applies to: gbm_paths, geometric_asian_call)

## Module-level notes

- line 130: `_COV` not translated: Mat constants are not supported
- line 131: `_W` not translated: Vec constants are not supported
