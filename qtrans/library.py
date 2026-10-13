"""Mapping from Python library functions to C++.

Keys are fully qualified Python names. The translator resolves whatever alias the
user imported (`import numpy as np`, `from math import exp`, `from scipy.stats import norm`)
back to these names, so `np.exp`, `numpy.exp` and a bare imported `exp` all land here.
"""
from __future__ import annotations

# --- constants -------------------------------------------------------------------
CONSTANTS = {}
for _mod in ("math", "numpy"):
    for _name in ("pi", "e", "inf", "nan"):
        CONSTANTS[f"{_mod}.{_name}"] = f"qt::{_name}"
CONSTANTS["math.tau"] = "qt::tau"

# --- f(x) -> float -----------------------------------------------------------------
_MATH_SAME = [
    "sqrt", "exp", "log10", "log2", "log1p", "expm1", "sin", "cos", "tan", "asin", "acos",
    "atan", "sinh", "cosh", "tanh", "asinh", "acosh", "atanh", "erf", "erfc", "cbrt", "exp2",
    "fabs", "lgamma",
]
UNARY_FLOAT = {f"math.{n}": f"std::{n}" for n in _MATH_SAME}
UNARY_FLOAT["math.gamma"] = "std::tgamma"

_NUMPY_SAME = [
    "sqrt", "exp", "log", "log10", "log2", "log1p", "expm1", "sin", "cos", "tan", "sinh",
    "cosh", "tanh", "cbrt", "exp2", "fabs", "floor", "ceil", "trunc",
]
UNARY_FLOAT.update({f"numpy.{n}": f"std::{n}" for n in _NUMPY_SAME})
UNARY_FLOAT.update({
    "numpy.arcsin": "std::asin", "numpy.arccos": "std::acos", "numpy.arctan": "std::atan",
    "numpy.arcsinh": "std::asinh", "numpy.arccosh": "std::acosh", "numpy.arctanh": "std::atanh",
    "numpy.rint": "std::nearbyint",  # half-to-even, like numpy
    "scipy.special.erf": "std::erf", "scipy.special.erfc": "std::erfc",
    "scipy.special.gamma": "std::tgamma", "scipy.special.gammaln": "std::lgamma",
    "scipy.special.ndtr": "qt::norm_cdf", "scipy.special.ndtri": "qt::norm_ppf",
})

# Functions that raise ValueError in Python's math module outside their domain, where
# C++ (and numpy) return NaN instead. Used for the report's semantic notes.
MATH_DOMAIN = {"math.sqrt", "math.log", "math.log10", "math.log2", "math.log1p", "math.asin",
               "math.acos", "math.acosh", "math.atanh", "math.gamma", "math.lgamma", "math.pow",
               "math.fmod"}
MATH_RANGE = {"math.exp", "math.exp2", "math.expm1", "math.sinh", "math.cosh", "math.gamma",
              "math.pow"}

# --- f(x, y) -> float ----------------------------------------------------------------
BINARY_FLOAT = {
    "math.atan2": "std::atan2", "math.hypot": "std::hypot", "math.copysign": "std::copysign",
    "math.fmod": "std::fmod",
    "numpy.arctan2": "std::atan2", "numpy.hypot": "std::hypot",
    "numpy.copysign": "std::copysign", "numpy.fmod": "std::fmod",
}

# --- f(x) -> bool --------------------------------------------------------------------
PREDICATES = {f"{m}.{n}": f"std::{n}" for m in ("math", "numpy") for n in ("isnan", "isinf", "isfinite")}

# --- scipy.stats.norm ------------------------------------------------------------------
NORM = {"scipy.stats.norm.cdf", "scipy.stats.norm.pdf", "scipy.stats.norm.ppf", "scipy.stats.norm.sf"}

# --- decorators that can be dropped without changing results -------------------------
DROPPABLE_DECORATORS = {
    "numba.njit": "@njit dropped: the C++ is already compiled",
    "numba.jit": "@jit dropped: the C++ is already compiled",
    "functools.cache": "@cache dropped: results are not memoised in C++",
    "functools.lru_cache": "@lru_cache dropped: results are not memoised in C++",
}

# --- exceptions: Python class -> C++ class -------------------------------------------
EXCEPTIONS = {  # the classes of qtrans_runtime.hpp, shaped as Python's
    n: f"qt::{n}" for n in ("Exception", "ArithmeticError", "ZeroDivisionError", "OverflowError", "FloatingPointError",
                            "AssertionError", "LookupError", "IndexError", "KeyError", "RuntimeError",
                            "NotImplementedError", "TypeError", "ValueError")
}
# the ones a float can raise in Python where a C++ double carries on (1.0 / 0.0, math.sqrt(-1),
# math.exp(1000)): code a handler for one of these may catch raises them as Python does
FLOAT_ERRORS = {"Exception", "ArithmeticError", "ZeroDivisionError", "OverflowError", "ValueError"}


def supported_names() -> list[str]:
    """Every library name stage 1 understands (for the README and error hints)."""
    names = set(CONSTANTS) | set(UNARY_FLOAT) | set(BINARY_FLOAT) | set(PREDICATES) | NORM
    names |= {"math.log", "math.floor", "math.ceil", "math.trunc", "math.pow", "numpy.power",
              "numpy.abs", "numpy.absolute", "numpy.maximum", "numpy.minimum", "numpy.sign",
              "numpy.square"}
    return sorted(names)
