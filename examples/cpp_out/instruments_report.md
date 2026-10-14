# qtrans report: `instruments.py`

**40 of 40 functions translated.**

## Translated

- `class Option` (line 22)
- `class Quote` (line 59)
- `class Curve` (line 72)
- `class Bond` (line 101)
- `class Portfolio` (line 122)
- `class Account` (line 144)
- `class GBMEngine` (line 153)
- `mc_vs_black_scholes` (line 185)
- `reprice` (line 193)
- `weight_bounds` (line 198)
- `book_value` (line 209)
- `bumped_delta` (line 213)
  - `(up.price() - dn.price()) / (2.0 * h)` may hold a NumPy number or a Python float, depending on the run: Python then gives inf / NaN (NumPy) or raises (ZeroDivisionError, OverflowError); the C++ gives inf / NaN
- `straddle` (line 220)
- `quote_stats` (line 226)
- `bond_price` (line 231)
- `parallel_shift` (line 235)
- `dv01` (line 239)
- `run_rebalances` (line 243)
- `scaled_copy` (line 250)
- `leveraged_nav` (line 256)
- `Option.__post_init__` (line 31)
- `Option.forward` (line 36)
- `Option.d1` (line 39)
  - `self.forward / self.K` may hold a NumPy number or a Python float, depending on the run: Python then gives inf / NaN (NumPy) or raises (ZeroDivisionError, OverflowError); the C++ gives inf / NaN
- `Option.price` (line 42)
- `Option.delta` (line 50)
- `Option.atm` (line 55)
- `Quote.mid` (line 64)
- `Quote.spread_bps` (line 67)
- `Curve.__post_init__` (line 77)
- `Curve.zero` (line 83)
  - `(t - ts[i - 1]) / (ts[i] - ts[i - 1])` may hold a NumPy number or a Python float, depending on the run: Python then gives inf / NaN (NumPy) or raises (ZeroDivisionError, OverflowError); the C++ gives inf / NaN
- `Curve.df` (line 96)
- `Bond.__post_init__` (line 108)
  - float -> int beyond ±9.2e18 raises OverflowError in C++; Python returns a big integer
- `Bond.price` (line 113)
- `Portfolio.value` (line 127)
- `Portfolio.rebalance` (line 130)
- `Portfolio.scale` (line 139)
- `Account.nav` (line 149)
- `GBMEngine.__init__` (line 156)
- `GBMEngine.terminal` (line 166)
  - `years / self.dt` may hold a NumPy number or a Python float, depending on the run: Python then gives inf / NaN (NumPy) or raises (ZeroDivisionError, OverflowError); the C++ gives inf / NaN
  - float -> int beyond ±9.2e18 raises OverflowError in C++; Python returns a big integer
- `GBMEngine.call` (line 180)

## Differences on overflow and error paths

These only matter on inputs where Python would raise an exception or overflow.

- int arithmetic that could leave 64 bits is checked: where Python's unbounded int would exceed ±9.2e18, C++ raises OverflowError (applies to: Curve.zero, Bond.price, Portfolio.rebalance, GBMEngine.terminal)
