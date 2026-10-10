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
  - math.exp raises OverflowError on overflow in Python; C++ returns inf
- `Option.d1` (line 39)
  - math.log raises ValueError outside its domain in Python; C++ returns NaN
  - math.sqrt raises ValueError outside its domain in Python; C++ returns NaN
- `Option.price` (line 42)
  - math.sqrt raises ValueError outside its domain in Python; C++ returns NaN
  - math.exp raises OverflowError on overflow in Python; C++ returns inf
- `Option.delta` (line 50)
- `Option.atm` (line 55)
  - math.exp raises OverflowError on overflow in Python; C++ returns inf
- `Quote.mid` (line 64)
- `Quote.spread_bps` (line 67)
- `Curve.__post_init__` (line 77)
- `Curve.zero` (line 83)
- `Curve.df` (line 96)
  - math.exp raises OverflowError on overflow in Python; C++ returns inf
- `Bond.__post_init__` (line 108)
  - float -> int beyond ±9.2e18 raises OverflowError in C++; Python returns a big integer
- `Bond.price` (line 113)
- `Portfolio.value` (line 127)
- `Portfolio.rebalance` (line 130)
- `Portfolio.scale` (line 139)
- `Account.nav` (line 149)
- `GBMEngine.__init__` (line 156)
- `GBMEngine.terminal` (line 166)
  - float -> int beyond ±9.2e18 raises OverflowError in C++; Python returns a big integer
  - math.sqrt raises ValueError outside its domain in Python; C++ returns NaN
- `GBMEngine.call` (line 180)
  - math.exp raises OverflowError on overflow in Python; C++ returns inf

## Differences on overflow and error paths

These only matter on inputs where Python would raise an exception or overflow.

- division by zero gives ±inf or NaN in C++ where Python raises ZeroDivisionError (applies to: bumped_delta, Option.d1, Quote.spread_bps, Curve.zero, Bond.price, GBMEngine.__init__, GBMEngine.terminal)
- int arithmetic uses 64-bit long long; Python ints are unbounded, so results beyond ±9.2e18 would differ (applies to: Curve.zero, Bond.price, Portfolio.rebalance, GBMEngine.terminal)
