# qtrans report: `bl_config.py`

**0 of 1 functions translated.**

## Not translated

- `market_default` (line 91). line 91: parameter 'market_name' has no type annotation, and no call in this file shows its type
  - Hint: write `market_name: float` (or int / bool) for a number, or `market_name: Vec` / `market_name: Mat` for a 1-D / 2-D float array (from qtrans.typing import Vec, Mat)

## Module constants

- `MIN_HISTORY_COVERAGE: float`
- `TARGET_OBS_PER_ASSET: int`
- `AUTO_HOLDINGS: int`
- `DEFAULT_RF: float`
- `DEFAULT_MAX_WEIGHT_PCT: int`
- `DEFAULT_ASSET_CLASS_CAP_PCT: int`
- `DEFAULT_TRAIN_FRAC: float`
- `DEFAULT_VIEW_CONFIDENCE: float`
- `DEFAULT_VOL_TARGET_PCT: float`
- `DEFAULT_VOL_TARGET_FRAC_PCT: int`

## Module-level notes

- line 5: `FREQ_PER_YEAR` not translated: dicts are not supported in stage 1
- line 6: `FREQ_RULE` not translated: dicts are not supported in stage 1
- line 28: `HORIZON_CFG` not translated: dicts are not supported in stage 1
- line 51: `BENCHMARKS` not translated: dicts are not supported in stage 1
- line 67: `MARKET_DEFAULTS` not translated: dicts are not supported in stage 1
- line 97: `MARKETS` not translated: dicts are not supported in stage 1
- line 233: `VIEW_TYPES` not translated: lists are not supported in stage 1
- line 252: `VIEW_ENGINES` not translated: dicts are not supported in stage 1
- line 260: `ENGINE_NOTES` not translated: dicts are not supported in stage 1
