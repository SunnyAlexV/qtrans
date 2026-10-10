# qtrans report: `portfolio.py`

**8 of 8 functions translated.**

## Translated

- `sample_cov` (line 8)
- `portfolio_vol` (line 15)
- `min_variance_weights` (line 19)
- `tangency_weights` (line 25)
- `frontier_weights` (line 31)
- `risk_contributions` (line 44)
- `black_litterman` (line 50)
- `bl_weights` (line 70)

## Differences on overflow and error paths

These only matter on inputs where Python would raise an exception or overflow.

- int arithmetic uses 64-bit long long; Python ints are unbounded, so results beyond ±9.2e18 would differ (applies to: sample_cov)
- division by zero gives ±inf or NaN in C++ where Python raises ZeroDivisionError (applies to: frontier_weights)

## Module-level notes

- line 77: `_rng` not translated: Generator constants are not supported
- line 78: `_R` not translated: module variable '_rng' was not translated: Generator constants are not supported
- line 81: `_COV` not translated: module variable '_R' was not translated: module variable '_rng' was not translated: Generator constants are not supported
- line 82: `_MU` not translated: Vec constants are not supported
- line 83: `_W` not translated: Vec constants are not supported
- line 84: `_P` not translated: Mat constants are not supported
- line 85: `_Q` not translated: Vec constants are not supported
- line 86: `_OMEGA` not translated: Vec constants are not supported
- line 87: `_SINGULAR` not translated: Mat constants are not supported
