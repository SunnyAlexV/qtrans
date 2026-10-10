# qtrans report: `bl_core.py`

**14 of 37 functions translated.**

## Translated

- `nearest_psd` (line 27)
  - np.linalg.eigh: eigenvalues agree to rounding; each eigenvector is defined only up to sign (and rotation within a repeated eigenvalue), which LAPACK and Eigen may choose differently. Results like V @ diag(f(w)) @ V.T agree.
- `ledoit_wolf_identity` (line 34)
  - np.einsum is computed by direct summation; NumPy's kernels add in a different order, so results agree to rounding rather than bit for bit
- `_ewma_weights` (line 67)
  - x ** y with a negative x and non-integer y gives a complex number for a Python float but NaN in C++ (and for np.float64)
  - np.arange of ints is stored as doubles (exact for |values| < 2^53)
- `horizon_returns` (line 73)
- `variance_ratio` (line 88)
  - np.errstate(...) blocks are translated as their contents: they only change which floating-point warnings NumPy prints, never the values
  - line 109: `R.ndim == 1` is always False for the declared types, so the other branch is left out
- `apply_variance_ratio` (line 129)
- `equilibrium_returns` (line 300)
- `implied_confidence_weights` (line 311)
  - np.linalg.pinv agrees with NumPy to rounding; singular values within rounding of the rcond cutoff can be kept by one and dropped by the other
- `omega_he_litterman` (line 597)
- `posterior_mu` (line 673)
  - np.linalg.pinv agrees with NumPy to rounding; singular values within rounding of the rcond cutoff can be kept by one and dropped by the other
- `posterior_sigma` (line 702)
  - np.linalg.pinv agrees with NumPy to rounding; singular values within rounding of the rcond cutoff can be kept by one and dropped by the other
- `portfolio_stats` (line 766)
  - math.sqrt raises ValueError outside its domain in Python; C++ returns NaN
- `_downside_dev` (line 774)
  - math.sqrt raises ValueError outside its domain in Python; C++ returns NaN
- `_cvar` (line 781)

## Differences on overflow and error paths

These only matter on inputs where Python would raise an exception or overflow.

- division by zero gives ±inf or NaN in C++ where Python raises ZeroDivisionError (applies to: ledoit_wolf_identity, _ewma_weights, variance_ratio, portfolio_stats)
- int arithmetic uses 64-bit long long; Python ints are unbounded, so results beyond ±9.2e18 would differ (applies to: ledoit_wolf_identity, _ewma_weights, variance_ratio)
- float ** raises OverflowError in Python when the result overflows; C++ returns inf (applies to: _ewma_weights)

## Not translated

- `to_returns` (line 17). line 17: parameter 'prices' has no type annotation
  - Hint: write `prices: float` (or int / bool) for a number, or `prices: Vec` / `prices: Mat` for a 1-D / 2-D float array (from qtrans.typing import Vec, Mat)
- `estimate_cov` (line 141). line 141: parameter 'returns' has no type annotation
  - Hint: write `returns: float` (or int / bool) for a number, or `returns: Vec` / `returns: Mat` for a 1-D / 2-D float array (from qtrans.typing import Vec, Mat)
- `estimate_mu` (line 160). line 160: parameter 'returns' has no type annotation
  - Hint: write `returns: float` (or int / bool) for a number, or `returns: Vec` / `returns: Mat` for a 1-D / 2-D float array (from qtrans.typing import Vec, Mat)
- `_scalar` (line 198). line 198: parameter 'x' has no type annotation
  - Hint: write `x: float` (or int / bool) for a number, or `x: Vec` / `x: Mat` for a 1-D / 2-D float array (from qtrans.typing import Vec, Mat)
- `market_weights` (line 205). line 205: parameter 'tickers' has no type annotation
  - Hint: write `tickers: float` (or int / bool) for a number, or `tickers: Vec` / `tickers: Mat` for a 1-D / 2-D float array (from qtrans.typing import Vec, Mat)
- `backcast_caps` (line 240). line 240: parameter 'caps_today' has no type annotation
  - Hint: write `caps_today: float` (or int / bool) for a number, or `caps_today: Vec` / `caps_today: Mat` for a 1-D / 2-D float array (from qtrans.typing import Vec, Mat)
- `implied_risk_aversion` (line 278). line 278: parameter 'w_mkt' has no type annotation
  - Hint: write `w_mkt: float` (or int / bool) for a number, or `w_mkt: Vec` / `w_mkt: Mat` for a 1-D / 2-D float array (from qtrans.typing import Vec, Mat)
- `blank_views` (line 324). line 324: parameter 'tickers' has no type annotation
  - Hint: write `tickers: float` (or int / bool) for a number, or `tickers: Vec` / `tickers: Mat` for a 1-D / 2-D float array (from qtrans.typing import Vec, Mat)
- `build_pq` (line 338). line 338: parameter 'views_df' has no type annotation
  - Hint: write `views_df: float` (or int / bool) for a number, or `views_df: Vec` / `views_df: Mat` for a 1-D / 2-D float array (from qtrans.typing import Vec, Mat)
- `_engine_scores` (line 418). line 418: parameter 'prices' has no type annotation
  - Hint: write `prices: float` (or int / bool) for a number, or `prices: Vec` / `prices: Mat` for a 1-D / 2-D float array (from qtrans.typing import Vec, Mat)
- `systematic_views` (line 458). line 458: parameter 'prices' has no type annotation
  - Hint: write `prices: float` (or int / bool) for a number, or `prices: Vec` / `prices: Mat` for a 1-D / 2-D float array (from qtrans.typing import Vec, Mat)
- `engine_hit_rate` (line 516). line 516: parameter 'prices' has no type annotation
  - Hint: write `prices: float` (or int / bool) for a number, or `prices: Vec` / `prices: Mat` for a 1-D / 2-D float array (from qtrans.typing import Vec, Mat)
- `confidence_from_hit_rate` (line 573). line 573: parameter 'hit_rate' has no type annotation
  - Hint: write `hit_rate: float` (or int / bool) for a number, or `hit_rate: Vec` / `hit_rate: Mat` for a 1-D / 2-D float array (from qtrans.typing import Vec, Mat)
- `omega_idzorek` (line 609). line 609: parameter 'P' has no type annotation
  - Hint: write `P: float` (or int / bool) for a number, or `P: Vec` / `P: Mat` for a 1-D / 2-D float array (from qtrans.typing import Vec, Mat)
- `black_litterman` (line 722). line 722: parameter 'Sigma' has no type annotation
  - Hint: write `Sigma: float` (or int / bool) for a number, or `Sigma: Vec` / `Sigma: Mat` for a 1-D / 2-D float array (from qtrans.typing import Vec, Mat)
- `_stance_bounds` (line 789). line 789: parameter 'stance' has no type annotation
  - Hint: write `stance: float` (or int / bool) for a number, or `stance: Vec` / `stance: Mat` for a 1-D / 2-D float array (from qtrans.typing import Vec, Mat)
- `_solve` (line 797). line 797: parameter 'obj_fun' has no type annotation
  - Hint: write `obj_fun: float` (or int / bool) for a number, or `obj_fun: Vec` / `obj_fun: Mat` for a 1-D / 2-D float array (from qtrans.typing import Vec, Mat)
- `_optimize_once` (line 832). line 832: parameter 'objective' has no type annotation
  - Hint: write `objective: float` (or int / bool) for a number, or `objective: Vec` / `objective: Mat` for a 1-D / 2-D float array (from qtrans.typing import Vec, Mat)
- `resample_stack` (line 871). line 871: parameter 'objective' has no type annotation
  - Hint: write `objective: float` (or int / bool) for a number, or `objective: Vec` / `objective: Mat` for a 1-D / 2-D float array (from qtrans.typing import Vec, Mat)
- `optimize_portfolio` (line 896). line 896: parameter 'objective' has no type annotation
  - Hint: write `objective: float` (or int / bool) for a number, or `objective: Vec` / `objective: Mat` for a 1-D / 2-D float array (from qtrans.typing import Vec, Mat)
- `_min_var_at_target` (line 908). line 908: parameter 'mu' has no type annotation
  - Hint: write `mu: float` (or int / bool) for a number, or `mu: Vec` / `mu: Mat` for a 1-D / 2-D float array (from qtrans.typing import Vec, Mat)
- `efficient_frontier` (line 934). line 934: parameter 'mu' has no type annotation
  - Hint: write `mu: float` (or int / bool) for a number, or `mu: Vec` / `mu: Mat` for a 1-D / 2-D float array (from qtrans.typing import Vec, Mat)
- `monte_carlo_cloud` (line 969). line 969: parameter 'mu' has no type annotation
  - Hint: write `mu: float` (or int / bool) for a number, or `mu: Vec` / `mu: Mat` for a 1-D / 2-D float array (from qtrans.typing import Vec, Mat)
