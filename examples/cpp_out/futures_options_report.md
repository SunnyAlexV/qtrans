# qtrans report: `futures_options.py`

**3 of 3 functions translated.**

## Translated

- `black76` (line 8)
  - math.sqrt raises ValueError outside its domain in Python; C++ returns NaN
  - math.log raises ValueError outside its domain in Python; C++ returns NaN
  - math.exp raises OverflowError on overflow in Python; C++ returns inf
- `bachelier` (line 19)
  - math.sqrt raises ValueError outside its domain in Python; C++ returns NaN
  - math.exp raises OverflowError on overflow in Python; C++ returns inf
- `margrabe` (line 31)
  - math.sqrt raises ValueError outside its domain in Python; C++ returns NaN
  - math.log raises ValueError outside its domain in Python; C++ returns NaN
  - math.exp raises OverflowError on overflow in Python; C++ returns inf

## Differences on overflow and error paths

These only matter on inputs where Python would raise an exception or overflow.

- division by zero gives ±inf or NaN in C++ where Python raises ZeroDivisionError (applies to: black76, bachelier, margrabe)
- float ** raises OverflowError in Python when the result overflows; C++ returns inf (applies to: margrabe)
