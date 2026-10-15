# A worked example, from Python to verified C++

One module taken through every step, with the real output of each command (written by `tools/worked_example.py`; run it to reproduce this page). The module is [`docs/worked_example/pricer.py`](worked_example/pricer.py): a Black-Scholes call, its vega, implied volatility by Newton's method, and a book of calls.

## 1. The Python

```python
"""A small option pricer: Black-Scholes, its vega, implied volatility, and a book of options."""
import math

import numpy as np
from scipy.special import ndtr

from qtrans.typing import Vec


def bs_call(S: float, K: float, T: float, vol: float, r: float) -> float:
    """Black-Scholes price of a European call."""
    if S <= 0.0 or K <= 0.0:
        raise ValueError("spot and strike must be positive")
    sd = vol * math.sqrt(T)
    d1 = (math.log(S / K) + (r + 0.5 * vol * vol) * T) / sd
    return S * ndtr(d1) - K * math.exp(-r * T) * ndtr(d1 - sd)


def vega(S: float, K: float, T: float, vol: float, r: float) -> float:
    """dPrice/dvol."""
    sd = vol * math.sqrt(T)
    d1 = (math.log(S / K) + (r + 0.5 * vol * vol) * T) / sd
    return S * math.sqrt(T) * math.exp(-0.5 * d1 * d1) / math.sqrt(2.0 * math.pi)


def implied_vol(price: float, S: float, K: float, T: float, r: float) -> float:
    """The volatility that reprices the call, by Newton's method from 20%."""
    vol = 0.2
    for _ in range(100):
        diff = bs_call(S, K, T, vol, r) - price
        if abs(diff) < 1e-12:
            return vol
        vol -= diff / vega(S, K, T, vol, r)
    raise ValueError("implied_vol did not converge")


def book(S: Vec, K: Vec, T: Vec, vol: Vec, r: float) -> Vec:
    """Prices for a book of calls."""
    out = np.zeros(len(S))
    for i in range(len(S)):
        out[i] = bs_call(float(S[i]), float(K[i]), float(T[i]), float(vol[i]), r)
    return out


_g = np.random.default_rng(3)
QTRANS_CASES = {
    "bs_call": [(100.0, 95.0, 1.0, 0.2, 0.03), (100.0, 95.0, 0.0, 0.2, 0.03), (-1.0, 95.0, 1.0, 0.2, 0.03)],
    "vega": [(100.0, 95.0, 1.0, 0.2, 0.03)],
    "implied_vol": [(10.0, 100.0, 95.0, 1.0, 0.03), (1e-9, 100.0, 95.0, 1.0, 0.03)],
    "book": [(100.0 + 10.0 * _g.standard_normal(1000), np.full(1000, 100.0), np.full(1000, 1.0),
              np.full(1000, 0.2), 0.03)],
}
```

`QTRANS_CASES` are the inputs you want checked by name: here an expired option (`T = 0`, so `sd = 0`: Python raises `ZeroDivisionError`), a negative spot (the function's own `ValueError`) and a price no volatility reaches (Newton gives up). `qtrans verify` adds 200 random inputs per function.

## 2. Translate

```
$ qtrans translate pricer.py -o cpp_out
qtrans: pricer.py: 4 of 4 functions translated
  [ok]       bs_call
  [ok]       vega
  [ok]       implied_vol
  [ok]       book
wrote cpp_out/pricer.hpp
```

The header, from its namespace on (above it: the includes and the fast-math guard):

```cpp
namespace pricer {

// Declarations (so functions can call each other in any order).
double bs_call(double S, double K, double T, double vol, double r);
double vega(double S, double K, double T, double vol, double r);
double implied_vol(double price, double S, double K, double T, double r);
qt::Vec book(const qt::Vec& S, const qt::Vec& K, const qt::Vec& T, const qt::Vec& vol, double r);

/// Black-Scholes price of a European call.
inline double bs_call(double S, double K, double T, double vol, double r) {
    if (S <= 0.0 || K <= 0.0) {
        throw qt::ValueError("spot and strike must be positive");
    }
    double sd = vol * qt::py::math_sqrt(T);
    double d1 = qt::py::truediv(qt::py::math_log(qt::py::truediv(S, K)) + (r + 0.5 * vol * vol) * T, sd);
    return S * qt::norm_cdf(d1) - K * qt::py::math_exp(-r * T) * qt::norm_cdf(d1 - sd);
}

/// dPrice/dvol.
inline double vega(double S, double K, double T, double vol, double r) {
    double sd = vol * qt::py::math_sqrt(T);
    double d1 = qt::py::truediv(qt::py::math_log(qt::py::truediv(S, K)) + (r + 0.5 * vol * vol) * T, sd);
    const double left = S * qt::py::math_sqrt(T);
    const double left2 = left * qt::py::math_exp(-0.5 * d1 * d1);
    return qt::py::truediv(left2, qt::py::math_sqrt(2.0 * qt::pi));
}

/// The volatility that reprices the call, by Newton's method from 20%.
inline double implied_vol(double price, double S, double K, double T, double r) {
    double vol = 0.2;
    for (long long _ = 0; _ < 100; ++_) {
        double diff = bs_call(S, K, T, vol, r) - price;
        if (std::fabs(diff) < 1e-12) {
            return vol;
        }
        vol -= diff / vega(S, K, T, vol, r);
    }
    throw qt::ValueError("implied_vol did not converge");
}

/// Prices for a book of calls.
inline qt::Vec book(const qt::Vec& S, const qt::Vec& K, const qt::Vec& T, const qt::Vec& vol, double r) {
    qt::Vec out = qt::zeros(S.size());
    for (long long i = 0, i_end = S.size(); i < i_end; ++i) {
        out(i) = bs_call(S(i), K(i), T(i), vol(i), r);
    }
    return out;
}

}  // namespace pricer
```

What to notice:

- `qt::py::truediv`, `qt::py::math_log`, `qt::py::math_sqrt`: Python's float errors, raised as Python raises them (`ZeroDivisionError: float division by zero` for the expired option), with the running Python's words.
- In `implied_vol`, `vol -= diff / vega(...)` keeps a plain `/`: `ndtr` returns a NumPy number, so `bs_call` does, and so `diff` is one, and NumPy's numbers don't raise (`np.float64(1.0) / 0.0` is inf, with a warning, in Python too). qtrans follows where each number comes from to decide.
- In `vega`, two parts of one expression can raise, so the left ones are worked out first (`const double left = ...`): Python's order, so Python's error.
- The docstrings became comments, the `raise ValueError(...)` a `throw qt::ValueError(...)`, the `for _ in range(100)` an ordinary loop; `book` takes its arrays as `const qt::Vec&`, since it only reads them.
- `ndtr` is `qt::norm_cdf`: SciPy's algorithm on the C library's `erf`, equal to SciPy's to rounding (the max rel err column of the verify table below says how close; `vega`, which doesn't use it, is identical to the bit).

## 3. Verify

```
$ qtrans verify pricer.py --cases 200
qtrans verify: pricer.py
  translation: 4 of 4 functions translated
  compiler:    g++ (0 warnings in generated code)
  tolerance:   rtol=1e-10, atol=1e-13

  function                   cases  match ill-cond  known MISMATCH  max rel err  max ULP
  bs_call                      203    203        0      0        0      2.8e-13     1792
  vega                         201    201        0      0        0      0.0e+00        0
  implied_vol                  202    202        0      0        0      3.3e-15       24
  book                         201    201        0      0        0      8.8e-13     4096

  max rel err / max ULP exclude results within atol of each other (values near zero)
  and ill-conditioned cases; for arrays, rel err is normwise and ULPs count elements
  >= 1% of the array's largest value.

  RESULT: PASS (no mismatches)
```

Every case is a match: the values within `rtol=1e-10` (most to the last bit), and every error the same class with the same message.

## 4. Time

The same functions on the inputs above, 7 rounds, Python and C++ taking turns, medians (µs per call; `tools/benchmark.py` has the protocol and many more):

| Function | Python (µs) | C++ (µs) | Speed-up |
|---|---:|---:|---:|
| `bs_call` | 0.62 | 0.034 | 18× |
| `vega` | 0.34 | 0.022 | 16× |
| `implied_vol` | 10.66 | 0.395 | 27× |
| `book` | 1,192.92 | 40.535 | 29× |

## 5. Hand it over

```
$ qtrans export pricer.py -o pricer_cpp
wrote pricer_cpp: target qtrans::pricer, 1 module(s)
golden tests: 207 cases of Python's results recorded in tests/golden.txt
built and tested with CMake here: 1 of 1 ctest tests passed (207 of 207 cases agree with Python)
```

`pricer_cpp/` is a CMake package: the headers, and golden tests holding Python's answers, so the C++ team can build and test it without Python. `qtrans bind pricer.py` would instead compile it into a Python module, for Python code to call the fast version.
