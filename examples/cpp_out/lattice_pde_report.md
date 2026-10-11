# qtrans report: `lattice_pde.py`

**4 of 4 functions translated.**

## Translated

- `american_binomial` (line 9)
  - np.arange of ints is stored as doubles (exact for |values| < 2^53)
- `early_exercise_premium` (line 34)
  - np.arange of ints is stored as doubles (exact for |values| < 2^53)
- `bs_explicit_fd` (line 50)
  - np.arange of ints is stored as doubles (exact for |values| < 2^53)
- `fd_price` (line 73)
  - float -> int beyond ±9.2e18 raises OverflowError in C++; Python returns a big integer

## Differences on overflow and error paths

These only matter on inputs where Python would raise an exception or overflow.

- division by zero gives ±inf or NaN in C++ where Python raises ZeroDivisionError (applies to: american_binomial, early_exercise_premium, bs_explicit_fd, fd_price)
- int arithmetic that could leave 64 bits is checked: where Python's unbounded int would exceed ±9.2e18, C++ raises OverflowError (applies to: american_binomial, early_exercise_premium, bs_explicit_fd, fd_price)
