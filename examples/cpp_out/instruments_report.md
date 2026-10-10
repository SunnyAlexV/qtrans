# qtrans report: `instruments.py`

**34 of 34 functions translated.**

## Translated

- `class Option` (line 21)
- `class Quote` (line 58)
- `class Curve` (line 71)
- `class Bond` (line 100)
- `class Portfolio` (line 121)
- `class Account` (line 143)
- `weight_bounds` (line 152)
- `book_value` (line 163)
- `bumped_delta` (line 167)
- `straddle` (line 174)
- `quote_stats` (line 180)
- `bond_price` (line 185)
- `parallel_shift` (line 189)
- `dv01` (line 193)
- `run_rebalances` (line 197)
- `scaled_copy` (line 204)
- `leveraged_nav` (line 210)
- `Option.__post_init__` (line 30)
- `Option.forward` (line 35)
  - math.exp raises OverflowError on overflow in Python; C++ returns inf
- `Option.d1` (line 38)
  - math.log raises ValueError outside its domain in Python; C++ returns NaN
  - math.sqrt raises ValueError outside its domain in Python; C++ returns NaN
- `Option.price` (line 41)
  - math.sqrt raises ValueError outside its domain in Python; C++ returns NaN
  - math.exp raises OverflowError on overflow in Python; C++ returns inf
- `Option.delta` (line 49)
- `Option.atm` (line 54)
  - math.exp raises OverflowError on overflow in Python; C++ returns inf
- `Quote.mid` (line 63)
- `Quote.spread_bps` (line 66)
- `Curve.__post_init__` (line 76)
- `Curve.zero` (line 82)
- `Curve.df` (line 95)
  - math.exp raises OverflowError on overflow in Python; C++ returns inf
- `Bond.__post_init__` (line 107)
  - float -> int beyond ±9.2e18 raises OverflowError in C++; Python returns a big integer
- `Bond.price` (line 112)
- `Portfolio.value` (line 126)
- `Portfolio.rebalance` (line 129)
- `Portfolio.scale` (line 138)
- `Account.nav` (line 148)

## Differences on overflow and error paths

These only matter on inputs where Python would raise an exception or overflow.

- division by zero gives ±inf or NaN in C++ where Python raises ZeroDivisionError (applies to: bumped_delta, Option.d1, Quote.spread_bps, Curve.zero, Bond.price)
- int arithmetic uses 64-bit long long; Python ints are unbounded, so results beyond ±9.2e18 would differ (applies to: Curve.zero, Bond.price, Portfolio.rebalance)
