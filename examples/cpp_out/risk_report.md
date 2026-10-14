# qtrans report: `risk.py`

**8 of 8 functions translated.**

## Translated

- `historical_var` (line 8)
- `expected_shortfall` (line 13)
- `ewma_vol` (line 20)
- `max_drawdown` (line 31)
- `sharpe_ratio` (line 44)
- `wealth_curve` (line 49)
- `rolling_mean` (line 53)
- `downside_deviation` (line 64)

## Differences on overflow and error paths

These only matter on inputs where Python would raise an exception or overflow.

- int arithmetic that could leave 64 bits is checked: where Python's unbounded int would exceed ±9.2e18, C++ raises OverflowError (applies to: rolling_mean)

## Module-level notes

- line 69: `_rng` not translated: Generator constants are not supported
- line 70: `_R` not translated: module variable '_rng' was not translated: Generator constants are not supported
- line 71: `_PRICES` not translated: module variable '_R' was not translated: module variable '_rng' was not translated: Generator constants are not supported
