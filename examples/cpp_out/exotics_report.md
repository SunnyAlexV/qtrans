# qtrans report: `exotics.py`

**3 of 3 functions translated.**

## Translated

- `digital_call` (line 7)
- `geometric_asian_call` (line 14)
- `crr_european` (line 28)

## Differences on overflow and error paths

These only matter on inputs where Python would raise an exception or overflow.

- int arithmetic that could leave 64 bits is checked: where Python's unbounded int would exceed ±9.2e18, C++ raises OverflowError (applies to: geometric_asian_call, crr_european)
