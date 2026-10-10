# qtrans report: `implied_vol.py`

**3 of 3 functions translated.**

## Translated

- `bs_price` (line 14)
  - math.sqrt raises ValueError outside its domain in Python; C++ returns NaN
  - math.log raises ValueError outside its domain in Python; C++ returns NaN
  - math.exp raises OverflowError on overflow in Python; C++ returns inf
- `bs_vega` (line 23)
  - math.sqrt raises ValueError outside its domain in Python; C++ returns NaN
  - math.log raises ValueError outside its domain in Python; C++ returns NaN
- `implied_vol` (line 29)
  - math.exp raises OverflowError on overflow in Python; C++ returns inf
  - math.sqrt raises ValueError outside its domain in Python; C++ returns NaN

## Differences on overflow and error paths

These only matter on inputs where Python would raise an exception or overflow.

- division by zero gives ±inf or NaN in C++ where Python raises ZeroDivisionError (applies to: bs_price, bs_vega, implied_vol)

## Module constants

- `SIGMA_LO: float`
- `SIGMA_HI: float`
- `TOL: float`
- `MAX_NEWTON: int`
- `MAX_BISECT: int`
