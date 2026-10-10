# qtrans report: `bonds.py`

**6 of 6 functions translated.**

## Translated

- `bond_price` (line 5)
  - 'price' holds an int in some assignments and a float in others; declared double (same values)
- `macaulay_duration` (line 19)
- `modified_duration` (line 34)
- `convexity` (line 38)
- `ytm_newton` (line 52)
- `payments_remaining` (line 68)

## Differences on overflow and error paths

These only matter on inputs where Python would raise an exception or overflow.

- int arithmetic uses 64-bit long long; Python ints are unbounded, so results beyond ±9.2e18 would differ (applies to: bond_price, macaulay_duration, convexity, payments_remaining)
- division by zero gives ±inf or NaN in C++ where Python raises ZeroDivisionError (applies to: bond_price, macaulay_duration, modified_duration, convexity, ytm_newton)
- float ** raises OverflowError in Python when the result overflows; C++ returns inf (applies to: bond_price, macaulay_duration, convexity)
