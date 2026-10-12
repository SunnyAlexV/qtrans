# qtrans report: `bl_config.py`

**0 of 1 functions translated.**

## Not translated

- `market_default` (line 91). line 91: parameter 'market_name' has no type annotation, and no call in this file shows its type
  - Hint: write `market_name: float` (or int / bool) for a number, or `market_name: Vec` / `market_name: Mat` for a 1-D / 2-D float array (from qtrans.typing import Vec, Mat)

## Module constants

- `FREQ_PER_YEAR: dict[str, int]`
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
- `VIEW_TYPES: list[str]`
- `ENGINE_NOTES: dict[str, str]`

## Module-level notes

- line 6: `FREQ_RULE` not translated: a dict holding None among its values is not supported (yet)
- line 28: `HORIZON_CFG` not translated: a dict whose values have several types (int, str): Python keeps each one's type (an int prints as 1, a float as 1.0), the C++ dict has one
- line 51: `BENCHMARKS` not translated: dict values of type tuple[str, str] are not supported
- line 67: `MARKET_DEFAULTS` not translated: dict values of type dict[str, float] are not supported
- line 97: `MARKETS` not translated: a dict whose values have several types (bool, list[str], str): Python keeps each one's type (an int prints as 1, a float as 1.0), the C++ dict has one
- line 252: `VIEW_ENGINES` not translated: a dict holding None among its values is not supported (yet)
