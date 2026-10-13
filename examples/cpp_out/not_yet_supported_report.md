# qtrans report: `not_yet_supported.py`

**2 of 10 functions translated.**

## Translated

- `discount` (line 8)
  - math.exp raises OverflowError on overflow in Python; C++ returns inf
- `forward` (line 51)

## Differences on overflow and error paths

These only matter on inputs where Python would raise an exception or overflow.

- division by zero gives ±inf or NaN in C++ where Python raises ZeroDivisionError (applies to: forward)

## Not translated

- `annuity` (line 12). line 13: map() is not supported
  - Hint: write a comprehension instead: [f(x) for x in xs], or sum(f(x) for x in xs)
- `portfolio_value` (line 16). line 16: type annotation 'np.ndarray' doesn't say whether the array is 1-D or 2-D
  - Hint: annotate with Vec (1-D) or Mat (2-D): `from qtrans.typing import Vec, Mat`
- `running_totals` (line 20). line 20: generators (yield) are not supported
- `lookup` (line 27). line 28: sets are not supported in stage 1
- `no_annotations` (line 32). line 32: parameter 'x' has no type annotation, and no call in this file shows its type
  - Hint: write `x: float` (or int / bool) for a number, or `x: Vec` / `x: Mat` for a 1-D / 2-D float array (from qtrans.typing import Vec, Mat)
- `uses_annuity` (line 36). line 37: calls annuity(), which could not be translated
- `last_index` (line 40). line 43: loop variable 'i' is used outside its for-loop
  - Hint: Python keeps its last value after the loop; copy it into another variable inside the loop if you need it
- `sometimes_none` (line 46). line 46: sometimes_none() can reach its end without returning (Python would return None)
  - Hint: add a final return or raise
