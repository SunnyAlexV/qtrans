# qtrans report: `exotics.py`

**3 of 3 functions translated.**

## Translated

- `digital_call` (line 7)
  - math.sqrt raises ValueError outside its domain in Python; C++ returns NaN
  - math.log raises ValueError outside its domain in Python; C++ returns NaN
  - math.exp raises OverflowError on overflow in Python; C++ returns inf
- `geometric_asian_call` (line 14)
  - math.log raises ValueError outside its domain in Python; C++ returns NaN
  - math.sqrt raises ValueError outside its domain in Python; C++ returns NaN
  - math.exp raises OverflowError on overflow in Python; C++ returns inf
- `crr_european` (line 28)
  - math.sqrt raises ValueError outside its domain in Python; C++ returns NaN
  - math.exp raises OverflowError on overflow in Python; C++ returns inf
  - math.log raises ValueError outside its domain in Python; C++ returns NaN
  - math.log1p raises ValueError outside its domain in Python; C++ returns NaN

## Differences on overflow and error paths

These only matter on inputs where Python would raise an exception or overflow.

- division by zero gives ±inf or NaN in C++ where Python raises ZeroDivisionError (applies to: digital_call, geometric_asian_call, crr_european)
- int arithmetic uses 64-bit long long; Python ints are unbounded, so results beyond ±9.2e18 would differ (applies to: geometric_asian_call, crr_european)
- float ** raises OverflowError in Python when the result overflows; C++ returns inf (applies to: crr_european)
