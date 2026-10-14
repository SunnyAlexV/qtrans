# qtrans report: `black_litterman_app.py`

**21 of 21 functions translated.**

## Translated

- `nearest_psd` (line 21)
  - np.linalg.eigh: eigenvalues agree to rounding; each eigenvector is defined only up to sign (and rotation within a repeated eigenvalue), which LAPACK and Eigen may choose differently. Results like V @ diag(f(w)) @ V.T agree.
- `ledoit_wolf_identity` (line 28)
  - np.einsum is computed by direct summation; NumPy's kernels add in a different order, so results agree to rounding rather than bit for bit
- `_ewma_weights` (line 61)
- `horizon_returns` (line 67)
- `variance_ratio` (line 82)
  - np.errstate(...) blocks are translated as their contents: they only change which floating-point warnings NumPy prints, never the values
  - line 103: `R.ndim == 1` is always False for the declared types, so the other branch is left out
- `apply_variance_ratio` (line 123)
- `equilibrium_returns` (line 135)
- `implied_confidence_weights` (line 146)
  - np.linalg.pinv agrees with NumPy to rounding; singular values within rounding of the rcond cutoff can be kept by one and dropped by the other
- `omega_he_litterman` (line 156)
- `posterior_mu` (line 168)
  - np.linalg.pinv agrees with NumPy to rounding; singular values within rounding of the rcond cutoff can be kept by one and dropped by the other
- `posterior_sigma` (line 197)
  - np.linalg.pinv agrees with NumPy to rounding; singular values within rounding of the rcond cutoff can be kept by one and dropped by the other
- `portfolio_stats` (line 217)
- `_downside_dev` (line 225)
- `_cvar` (line 232)
- `default_block_length` (line 240)
  - float -> int beyond ±9.2e18 raises OverflowError in C++; Python returns a big integer
- `_ann_mean` (line 246)
- `_ann_sharpe` (line 250)
  - np.errstate(...) blocks are translated as their contents: they only change which floating-point warnings NumPy prints, never the values
- `newey_west_lag` (line 256)
  - float -> int beyond ±9.2e18 raises OverflowError in C++; Python returns a big integer
- `hac_cov` (line 261)
- `hit_rate_pvalue` (line 274)
- `buy_and_hold_returns` (line 283)
  - np.errstate(...) blocks are translated as their contents: they only change which floating-point warnings NumPy prints, never the values

## Differences on overflow and error paths

These only matter on inputs where Python would raise an exception or overflow.

- int arithmetic that could leave 64 bits is checked: where Python's unbounded int would exceed ±9.2e18, C++ raises OverflowError (applies to: ledoit_wolf_identity, _ewma_weights, variance_ratio, hac_cov)
