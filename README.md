# qtrans

[![tests](https://github.com/SunnyAlexV/qtrans/actions/workflows/tests.yml/badge.svg)](https://github.com/SunnyAlexV/qtrans/actions/workflows/tests.yml)
[![PyPI](https://img.shields.io/pypi/v/qtrans)](https://pypi.org/project/qtrans/)

**Translate typed quant-finance Python into readable, standalone C++, and validate that it
computes the same numbers through differential testing.**

A quant prototypes a pricer or a portfolio model in Python; a developer rewrites it in C++
for production, and bugs creep in during the rewrite. qtrans does the rewrite mechanically.
Its output is meant to be *handed over*: idiomatic C++ that keeps the original comments and
docstrings and is clean under `-Wall -Wextra -Wpedantic -Wshadow -Wconversion`. Scalar code
needs only the standard library; array code uses Eigen, the header-only library found in many
pricing libraries. Every translation comes with a verifier that compiles the C++, runs it and
Python on the same inputs, and reports every difference. That is testing, not a formal proof:
it shows agreement on every case run (hand-picked and thousands of random ones), which is
strong evidence, but no finite set of cases can rule out a difference elsewhere.

- **Stage 1, scalar code:** closed-form pricers, Greeks, root-finders, bond maths, lattice sums.
- **Stage 2, NumPy arrays:** portfolio construction (Markowitz, Black-Litterman), binomial
  trees with early exercise, finite-difference PDE solvers, risk statistics.
- **Stage 3, Monte Carlo:** NumPy's random generator (`np.random.default_rng`) reproduced in
  C++ bit for bit, so a translated simulation draws exactly the same numbers and gives the
  same price, not just a statistically similar one.
- **0.4, real code:** the numerical core of a real Black-Litterman portfolio app, written
  without qtrans in mind, translated with type hints as the only change. Making that work
  added `np.errstate`, `np.clip`, boolean masks, broadcasting, `keepdims`, `np.einsum`,
  `np.linalg.eigh`/`pinv`, `try`/`except LinAlgError`, and optional values (`x: float | None`).
  See [Tested on real code](#tested-on-real-code).
- **Stage 5, classes:** `@dataclass` (frozen or not), `NamedTuple` and plain classes with
  `__init__` become C++ structs with a constructor, `const` methods, properties, class methods
  and `==`; an object can own its random generator (a Monte Carlo engine). Objects are passed,
  modified and copied with Python's sharing semantics checked, not assumed. Strings can be
  compared, passed, returned and joined. See [Stage 5](#stage-5-classes-and-strings).
- **0.7, back to Python:** `python -m qtrans bind` compiles the translation into a Python
  module (pybind11) that keeps Python's semantics at the boundary: in-place changes reach your
  arrays, your generator advances exactly as far, your dataclasses go in and come back
  updated, and the exceptions are Python's. Checked like the C++ itself: 19,585 cases over the
  examples, 0 mismatches. See [Calling the C++ from Python](#calling-the-c-from-python).
- **0.8, projects:** files that import from each other (`from bl_core import nearest_psd`,
  `from bl_config import *`, `import rates as R`) are translated together, one header and C++
  namespace per module, with the analysis carried across the imports. The app's own four
  files, type hints added in place, translate as a project: 22 functions, 4,436 cases,
  0 mismatches, bindings included. See [Projects](#projects-several-modules).
- **0.8.1, stricter checks:** the verifier also tries non-square and empty matrices, vectors
  of different lengths and (`--profile stress`) inf, NaN, subnormals and the largest doubles;
  int arithmetic that could overflow is checked and raises `OverflowError` instead of wrapping;
  a Python call that hangs is stopped after 10 seconds. Length-1 arrays now broadcast as in
  NumPy (`a - b` with `len(a) == 1`), which the new shapes caught. See [Verification](#verification).

```python
def black_litterman(cov: Mat, w_mkt: Vec, P: Mat, Q: Vec, omega_diag: Vec,
                    delta: float = 2.5, tau: float = 0.05) -> tuple[Vec, Mat]:
    if np.any(omega_diag <= 0):
        raise ValueError("view variances must be positive")
    pi = delta * cov @ w_mkt  # equilibrium returns implied by market weights
    tau_cov = tau * cov
    M = P @ tau_cov @ P.T + np.diag(omega_diag)
    post_mu = pi + tau_cov @ P.T @ np.linalg.solve(M, Q - P @ pi)
    ...
```

becomes

```cpp
inline std::tuple<qt::Vec, qt::Mat> black_litterman(const qt::Mat& cov, const qt::Vec& w_mkt, const qt::Mat& P, const qt::Vec& Q, const qt::Vec& omega_diag, double delta, double tau) {
    if ((omega_diag <= 0.0).any()) {
        throw std::invalid_argument("view variances must be positive");
    }
    qt::Vec pi = qt::matvec(delta * cov, w_mkt);  // equilibrium returns implied by market weights
    qt::Mat tau_cov = tau * cov;
    qt::Mat M = qt::matmul(qt::matmul(P, tau_cov), P.transpose()) + qt::diag(omega_diag);
    qt::Vec post_mu = pi + qt::matvec(qt::matmul(tau_cov, P.transpose()), qt::solve(M, Q - qt::matvec(P, pi)));
    ...
```

The generated headers for every example are in [`examples/cpp_out/`](https://github.com/SunnyAlexV/qtrans/tree/main/examples/cpp_out).

## Results

Across eleven example modules: **87 functions, 17,921 test cases, 0 mismatches, 0 compiler
warnings** under the strict flags above. The full table is in
[`examples/RESULTS.md`](https://github.com/SunnyAlexV/qtrans/blob/main/examples/RESULTS.md), regenerated by `run_examples.py`.

- **31 functions are bit-identical to Python** (0 ULP apart in every case compared):
  - scalar code built from arithmetic and C-library maths: `d1`, `parity_gap`, the six bond
    functions, the CRR lattice;
  - all eight risk statistics on arrays, because the runtime reproduces NumPy's own
    algorithms (pairwise summation, `var`, `quantile`, `cumprod`) and so gets NumPy's exact
    roundings;
  - the finite-difference price;
  - 7 functions of the Black-Litterman app and 6 of `instruments` (the classes example).
- **Monte Carlo prices agree to a relative error of at most 1.5e-12**, because the draws are
  identical; the only differences come from NumPy's own `exp`. Antithetic European, Asian with
  a control variate, barrier, Heston, and portfolio VaR are all covered.
- **Functions using SciPy's `norm` or NumPy's `exp`/`pow` agree to a relative error of at most
  3.8e-11** (`book_value` in the classes example). They can't be bit-identical, because SciPy and NumPy use their own versions of
  `erf` and `exp`. Even inside Python, SciPy's `norm.cdf` and the same formula on `math.erf`
  agree to the bit only 63.9% of the time (max 45 ULP), and `numpy.exp` differs from
  `math.exp` on 4.4% of inputs (`tools/measure_library_ulps.py`).
- **Linear algebra goes through LAPACK in NumPy and Eigen in C++.** They agree to a relative
  error of at most 7.3e-11 in every case compared, random matrices included. 49 random cases
  (out of 12,544 for the array modules) were ill-conditioned, so the verifier reports them
  separately; see [Verification](#verification).
- **Speed, in the recorded run.** One call on one input each time; timings vary between runs.

  | Kind of code | Speed-up | Why |
  |---|---|---|
  | Scalar arithmetic | 7x to 44x | Python call overhead removed, code compiled |
  | Calls to `scipy.stats.norm` | 947x to 6,643x | Mostly SciPy's ~25-30 µs per-call overhead, against ~100-150 ns for `scipy.special.ndtr` |
  | Array code | 4x to 160x | Small where NumPy already runs in C (`cumprod`: 5x); large where Python loops over elements (drawdown: 140x, EWMA: 160x) |
  | Large vectorised Monte Carlo | about 1x to 6x | The time goes in `exp` over millions of elements, where NumPy uses run-time AVX code; portable `-O2` C++ calls the C library once per element (5.5x slower per `exp` in isolation). The C++ is still the same speed with no Python needed; `gbm_paths`, which loops over time steps, is 43x |

Three C++ programs in [`examples/use_from_cpp/`](https://github.com/SunnyAlexV/qtrans/tree/main/examples/use_from_cpp) use the generated
headers directly and print exactly what the Python prints:
- `price_book.cpp` computes implied vols and Greeks for an option book.
- `portfolio_demo.cpp` builds minimum-variance and Black-Litterman portfolios.
- `mc_demo.cpp` runs the Monte Carlo pricers. Its prices match Python's to all ten printed
  decimals, and so does the generator's final raw output.

## Quick start

**Install:** `pip install qtrans` (the translator and verifier), or `pip install "qtrans[bind]"`
to also build Python bindings. The examples and tests live in this repository, not in the
package.

**From an IDE:** open `run_examples.py` and press Run. It translates every example, compiles
and verifies each one, prints the reports and rewrites `examples/RESULTS.md`.

**From Python:**

```python
from qtrans import translate_file, verify_file

result = translate_file("my_model.py", "cpp_out")   # writes my_model.hpp + a report
report = verify_file("my_model.py", bench=True)      # compile, compare, time
print(report.to_text())
```

**From a terminal:** `python -m qtrans translate my_model.py -o cpp_out` (several files, or a
folder, for a project),
`python -m qtrans verify my_model.py --bench`, `python -m qtrans bind my_model.py --check`,
`python -m pytest` (414 tests). To see the Python next to its C++, ask what will happen
before translating, or check a new machine, see [Python and C++ side by side](#python-and-c-side-by-side).

**Requirements:**
- Python 3.10+, numpy and scipy.
- A C++17 compiler, for verification only. The verifier finds `g++`, `clang++` or `cl` on the
  PATH, or you pass `compiler=` a path. On Windows, a standalone MinGW-w64 GCC (for example
  from winlibs.com) can be unzipped anywhere.
- Eigen, for array and Monte Carlo code only. Install it with `conda install -c conda-forge eigen`. The
  verifier looks in the conda environment, then `/usr/include/eigen3`, or wherever
  `QTRANS_EIGEN` points.
- For `qtrans bind` only: `pip install pybind11 setuptools`, and the compiler Python's own build
  tools use. On Windows that is MSVC, also with Anaconda: install the free *Visual Studio Build
  Tools* with the "Desktop development with C++" workload (MinGW g++ is fine for `verify`, but
  can't build modules for an MSVC-built Python). On Linux and macOS, the system g++ or clang.
- `qtrans` must be importable for array examples (`from qtrans.typing import Vec, Mat`), so
  run from the repo root or `pip install -e .`.

Tested on every push by GitHub Actions on Linux (g++, Eigen 3.4 from apt) and Windows
(MinGW g++ for `verify` and MSVC for `bind`, Eigen from conda-forge, as on an Anaconda machine). In CI a skipped compiled test
counts as a failure, so a green run means the C++ really was compiled and compared.

## Python and C++ side by side

`python -m qtrans explain my_model.py --html my_model.html` writes a page with your Python on
the left and, on the right of each statement, the C++ it became. Under the C++, a line
explains each choice that isn't a plain transcription: a NumPy broadcasting helper where the
sizes can't be shown equal, a checked int operation, an index that may be negative, a copy
before a store that reads the array it writes, how each array is passed and why, a documented
Python/C++ difference that applies there. Functions qtrans refuses show the reason at the line
that caused it, with the hint. Without `--html` the same view prints in the terminal;
`FILE.py:FUNCTION` shows one function; `--json` gives it to other tools (the VS Code preview
planned for 0.10 will read it). `run_examples.py` writes one page per example to
[`examples/cpp_out/explain/`](https://github.com/SunnyAlexV/qtrans/tree/main/examples/cpp_out/explain)
(download one and open it in a browser).

```
  12     return X.T @ X / (T - ddof)        │     return qt::matmul(X.transpose(), X) / static_cast<double>(qt::isub(T, ddof));
                                            │ }
                                            │   ↳ int arithmetic that could leave 64 bits: the exact result, or OverflowError
                                            │     where Python's unbounded int would grow past what a long long holds (C++
                                            │     overflow is undefined)
```

How it works: the translator marks every C++ line with the Python lines it came from, as it
writes it, and `explain` groups them into rows (a statement, or an `if`/`for` header, and every
C++ line it produced; a block's closing brace goes with its last line). Tests check, for every
example, that each C++ line of a function body comes from a line of that function, that every
Python line is in exactly one row, in order, and that each reason sits next to the C++ it is
about ([`tests/test_explain.py`](tests/test_explain.py)). The marks are stripped before the
header is written, so headers are byte for byte what they were.

`python -m qtrans inspect my_model.py` lists, without compiling, what qtrans will make of each
function: translated or not (the reason, its line, the hint), the C++ signature and how each
parameter is passed, the `numpy.linalg` routines it uses, the functions it calls, and every
documented difference that applies, by line.

`python -m qtrans doctor` checks a machine: Python, NumPy and SciPy versions, the compiler
(it compiles and runs a C++17 program), Eigen and its version, pybind11; then it translates,
compiles and verifies a small module with arrays, `np.linalg.solve`, NumPy's random generator and
int `//`, which is what `qtrans verify` will do. `--bind` also builds and imports a compiled
module. Each problem comes with its fix (`conda install -c conda-forge eigen`, Visual Studio
Build Tools for bindings on Windows, ...).

## How it works

The translator reads the Python AST, plus the token stream so that comments survive, and
makes these passes over each function:

1. **Scan.** Accept only the supported statements, and collect every assignment.
2. **Check loops.** `for i in range(...)` becomes a C++ `for`. That is only equivalent if the
   body never assigns `i` and `i` isn't read after the loop, so both are checked.
3. **Infer types.** Every local gets one C++ type: the join of everything assigned to it on
   the ladder `bool < int < float`, or `Vec`/`Mat` for arrays. The pass iterates to a fixpoint.
4. **Check aliasing** (arrays). See below.
5. **Plan declarations.** Each variable is declared in the innermost block that uses it, but
   only if it is assigned before every read there. A Python variable read before it is
   assigned inside a loop carries its value over from the previous iteration, so it must live
   outside the loop.
6. **Emit.** Expressions carry their C++ precedence, so the output has exactly the parentheses
   needed: `a * s ** 2` becomes `a * (s * s)`, not `a * s * s`, which rounds differently.

### Stage 1: Python semantics that are preserved, not just syntax

| Python | Naive C++ would give | qtrans emits |
|---|---|---|
| `a / b == 3.5` for ints 7, 2 | `3` | `static_cast<double>(a) / static_cast<double>(b)` |
| `-7 // 2 == -4`, `-7 % 2 == 1` | `-3`, `-1` | `qt::floordiv`, `qt::mod` |
| `1.0 // 0.1 == 9.0` | `std::floor(1.0/0.1) == 10.0` | a port of CPython's `_float_div_mod` |
| `round(2.5) == 2` (half to even) | `std::round` gives 3 | `std::nearbyint` |
| `int(nan)` raises `ValueError` | undefined behaviour | `qt::to_int` raises the same |
| `100000 * 100000` | 32-bit overflow | `100000LL * 100000` |
| `a < b < c` | `(a < b) < c` | `a < b && b < c` |
| `norm.cdf(x, loc, scale)` gives NaN if `scale <= 0` | a number | `qt::norm_cdf(x, loc, scale)` checks |
| `a, b = b, a` | | `std::tie(a, b) = std::make_tuple(b, a)` |

### Stage 2: arrays

Arrays are annotated `Vec` (1-D) or `Mat` (2-D) from `qtrans.typing`. Plain `np.ndarray`
doesn't say the rank, and C++ needs it. They become Eigen *Arrays* rather than Matrices,
because Eigen Arrays are elementwise by default, as NumPy is. Matrices are row-major, so
memory order and flat reductions match NumPy's.

| Python | Difference a naive translation would hit | qtrans |
|---|---|---|
| `b = a`, `b = a[1:]`, `b = A.T` share memory | C++ copies | Analyses which names may share memory and refuses any pair where one side is later modified in place (suggests `.copy()`). Unshared code is translated as value copies, which then agree. |
| Passing an array to a function shares it | pass-by-value copies | Read only: `const qt::Vec&`. Rebound: by value. Modified in place: `qt::Vec&`, so the caller sees the change, as in Python. Both rebound and modified: refused. |
| `V[1:-1] = a*V[:-2] + b*V[1:-1] + c*V[2:]` computes the right side first | Eigen is lazy and reads values it has already overwritten | `.eval()` is added exactly when the right side reads the array being written |
| `a[-1]`, `a[i]` with `i` possibly negative | out of bounds | `a(a.size() - 1)`, `a(qt::idx(i, a.size()))`; plain `a(t)` when `t` is provably ≥ 0 (e.g. a `range(1, n)` loop variable, or `t - 1` inside it) |
| `a[2:100]` on a short array clips; `a[::-1]` reverses | assertion | `qt::slice`: CPython's slice-adjustment algorithm, line for line |
| `np.sum` uses pairwise summation | Eigen's `.sum()` adds in a different order | NumPy's algorithm ported (also `mean`, `var`/`std`, `quantile`, `linspace`, `arange`, `cumsum`): bit-identical |
| `A - v` broadcasts `v` across rows | size mismatch | `A.rowwise() - v.transpose()` when `v`'s length provably equals `A`'s columns (`mu = A.mean(axis=0)`); otherwise `qt::broadcast` |
| `a - b` with `a` of length 1 stretches it to `b`'s length (also `A + v`, `a < b`, `c += b`, `v[1:3] = w`, `np.maximum`, `np.where`) | size mismatch | Plain Eigen (`a - b`) where the translator can prove the sizes equal; otherwise `qt::broadcast`, `qt::broadcast_into`, `qt::store`, which follow NumPy's rule (equal sizes take Eigen's own operator: the same bits). The proof tracks a size token per dimension: parameters, `len(x)`, `np.zeros(n)`, `x[1:]` and `x[:-1]` (both `len(x) - 1`), results of elementwise operations, and variables that are rebound (`S = S * np.exp(z)`) when every assignment keeps the size (a fixpoint, like type inference) |
| `x ** 2`, `x ** 0.5`, `x ** -1` take fast paths in NumPy | `pow` rounds differently | `.square()`, `.sqrt()`, `.inverse()` |
| `np.maximum` propagates NaN; `np.max` of `[]` raises | Eigen doesn't | `qt::maximum`, `qt::max` follow NumPy |
| Shape and index errors raise `ValueError`/`IndexError` | Eigen asserts (abort) or undefined behaviour | Eigen's assertions become exceptions, caught as the matching Python error |
| `np.linalg.solve` of a singular matrix raises `LinAlgError` | Eigen returns inf/NaN | `qt::solve` checks for an exact zero pivot, as LAPACK does |
| `X - X.mean(axis=0, keepdims=True)`, `M - v[:, None]`: a `(1, m)` row or `(n, 1)` column broadcasts against `(n, m)` | Eigen refuses different shapes | `qt::broadcast`: NumPy's rule for any two 2-D arrays; equal shapes use Eigen's own vectorised operators |
| `y = np.asarray(x)` returns `x` itself when it is already float64, so `y[0] = 1` changes `x` | a copy | treated as the same array by the aliasing analysis; `x = np.asarray(x, dtype=float)` emits nothing |
| `np.maximum(-0.0, 0.0)` is `0.0` (a tie returns the second argument) but `np.clip(-0.0, 0.0, 1.0)` is `-0.0` (a tie keeps x); `1/x` tells them apart | `std::max` rules | `qt::np_maximum`, `qt::np_clip` follow each rule |
| `x[x >= c]` selects into a new array; `w[w < 0] = 0` changes only those elements | | `qt::select`, `qt::select_rows`, `qt::masked_update` |
| `with np.errstate(...)` | | translated as its body: it only changes which warnings NumPy prints |

Every rule above has a compiled test (`tests/test_array_semantics.py`). The three that are
easiest to get wrong were each mutation-tested by switching the rule off and confirming that
the tests fail: aliasing `.eval()`, negative-index wrapping, and pairwise summation.

### Stage 3: random numbers

`rng = np.random.default_rng(seed)` becomes `qt::Rng rng(seed);` (`qtrans_random.hpp`).
Same seed, same stream, bit for bit, because the whole chain is ported from NumPy's C and
Cython source:
- **`SeedSequence`:** hashing the seed into the initial state.
- **PCG64:** the 128-bit generator, with portable 128-bit arithmetic so MSVC works too, and a
  native fast path for GCC/Clang that gives identical output.
- **53-bit doubles:** how raw outputs become uniforms.
- **The ziggurat method behind `standard_normal`/`normal`:** NumPy's 3×256 tables are copied
  verbatim by `tools/gen_random_header.py`, under NumPy's BSD licence.

Checked against NumPy on 5 million draws across seeds up to 2⁶³−1, including the rare
ziggurat tail branch. A one-bit change in that branch makes the tests fail. Supported methods
are `random`, `standard_normal`, `normal` and `uniform`, with scalar, `n` or `(r, c)` sizes and
NumPy's argument checks (`scale < 0` raises `ValueError`, and so on).

| Python | Difference a naive translation would hit | qtrans |
|---|---|---|
| A generator passed to a function is shared: drawing advances the caller's stream | a copy would replay the same numbers | `qt::Rng&`; the verifier also checks that the caller's generator ends in the same state |
| `pair(rng.random(), rng.random())` evaluates left to right | C++ leaves argument order unspecified; GCC on x86 goes right to left, and with the rule switched off all test cases come out different (Python 0.097, C++ −1.004) | Two draws in one statement are refused, with a hint to draw on separate lines |
| `g2 = g` shares one stream | copies it | Refused |
| `np.random.normal(...)` uses hidden global state | | Refused, with a hint to use `default_rng(seed)` |
| `default_rng()` without a seed uses OS entropy | runs can't be compared | Refused: pass a seed |

The other way to make a simulation reproducible, passing the draws in as an array
(`european_from_draws(z, ...)`), needs nothing new: it is ordinary stage-2 code.

### 0.4: what real code needed

| Python | qtrans |
|---|---|
| `lag: int \| None = None`, `x: Optional[float]`, `-> float \| None` | `std::optional`. `x is None` is `!x.has_value()`; wherever Python uses the value itself (`x + 1`, `int(x)`, `x < 0`), `qt::value(x)` raises `TypeError` on None, as Python does. None is falsy. The verifier sends None as an input in a quarter of the random cases. |
| `try: solve(...) except np.linalg.LinAlgError: pinv(...)` | C++ `try`/`catch (const qt::linalg_error&)`. Variables assigned in the `try` are declared before it, and statements that ran before the error keep their effect, as in Python. |
| `except ValueError` | allowed unless the `try` body (or a function it calls) uses `math.sqrt`, `math.log`, ...: those raise `ValueError` in Python and return NaN in C++, so the handler would run in one and not the other. |
| `except ZeroDivisionError` / `OverflowError` / `Exception` / bare `except:` | refused, with the reason: Python raises them on float operations where C++ gives inf/NaN. |
| `np.linalg.eigh`, `eigvalsh` | Eigen's self-adjoint solver, reading the lower triangle as LAPACK does. Eigenvectors are unique only up to sign, so results like `V @ diag(f(w)) @ V.T` agree; code that reads individual entries may not (said in the report). |
| `np.linalg.pinv(A, rcond)` | NumPy's rank rule (`s > rcond * max(s)`) on Eigen's SVD; for full rank, a column-pivoted QR, which stays accurate on badly scaled matrices where rebuilding from the SVD loses 5-6 digits (see below). |
| `np.einsum("ij,ij->i", A, B)` | one or two operands of rank ≤ 2, implicit and explicit mode, size-1 broadcasting; common patterns run as direct loops or products. Agrees to rounding (NumPy's kernels add in a different order). |
| `np.var(R, axis=0, ddof=1)`, `np.cumprod(R, axis=0)`, `np.trace`, `np.column_stack`, `vstack`, `hstack`, `concatenate([[x0], v])`, `.ravel()`, `.astype(float)`, scalar `np.where`, scalar `np.clip` | NumPy's algorithms (`var` with an axis, `trace` and `cumprod` are bit-identical) |
| `if R.ndim == 1:` with `R: Mat` | the test is decided by the type, so the dead branch is left out (and said in the report) |

### Stage 5: classes and strings

```python
@dataclass(frozen=True)
class Option:
    S: float
    K: float
    T: float
    vol: float
    r: float = 0.0
    is_call: bool = True

    def __post_init__(self):
        if self.S <= 0 or self.K <= 0 or self.T <= 0 or self.vol <= 0:
            raise ValueError("S, K, T and vol must be positive")

    @property
    def forward(self) -> float:
        return self.S * math.exp(self.r * self.T)

    def price(self) -> float: ...
```

becomes

```cpp
struct Option {  // frozen dataclass
    double S{};
    ...
    Option() = default;  // only so a variable can be declared before it is assigned
    Option(double S_, double K_, double T_, double vol_, double r_ = 0.0, bool is_call_ = true);
    double forward() const;  // @property
    double price() const;
    static Option atm(double S_, double T_, double vol_, double r_ = 0.0);
    bool operator==(const Option& other) const;
    ...
};
```

and `Option(S=100, K=95, T=0.5, vol=0.2)` becomes `Option(100.0, 95.0, 0.5, 0.2)`, with keyword
arguments put in field order and defaults spelled out where a later field is given.
[`examples/instruments.py`](https://github.com/SunnyAlexV/qtrans/blob/main/examples/instruments.py) has options, quotes, a yield curve, bonds,
and a mutable portfolio inside an account.

**Sharing.** Python shares objects on assignment and in calls; a C++ struct is copied. They
agree as long as nothing changes a shared object, so the same analysis that guards arrays
applies to objects, and to the arrays inside them:

| Python | qtrans |
|---|---|
| a method that changes `self` (a field, an element of an array field, a method that does, a function that does) | non-`const`; every other method is `const` |
| a function that changes an object argument | `Portfolio&`, so the caller sees the change; the verifier compares the caller's object after the call, field by field |
| an object that can't change: frozen, with no array anywhere inside | always `const Option&`, never copied |
| `@dataclass(frozen=True)` with an array field | frozen only stops reassigning fields; `o.w[0] = 1` is legal Python, so it is treated like a mutable object |
| `p = Portfolio(w)`, then `p.scale(2)` | refused: `w` *is* `p.weights` in Python, so `w` changes too |
| `w = p.weights; w[0] = 5`, or `q = p; q.scale(2)` | refused, with the hint `copy.deepcopy(p)` |
| `copy.deepcopy(p)` | a plain C++ copy, which is a deep copy |
| `dataclasses.replace(p, cash=1.0)` | through the constructor (so `__post_init__` runs again); still shares the other fields' arrays, as in Python |
| `__post_init__` doing `self.w /= self.w.sum()` | refused: that changes the array the caller passed in; `self.w = self.w / self.w.sum()` is fine |

**Plain classes.** A class without `@dataclass` gets its fields from `__init__`: every
`self.x = ...` there (or `x: float` declared in the class body), typed by the same inference
as local variables, so `self.n = len(w)` is an int and `self.total = 0.0` a float. `__init__`
becomes the C++ constructor. A method that creates an attribute `__init__` doesn't set is
refused, and so is `==`, which compares identity for a plain class.

**A generator inside an object.** `self.rng = np.random.default_rng(seed)` in `__init__` (or a
`Generator` field) is a `qt::Rng` member, reproducing NumPy's stream exactly. Drawing from it
changes the object, so such methods aren't `const` and the object is passed by reference; the
verifier hands the generator to C++ mid-stream (its exact PCG64 state) and checks after the
call that both sides drew the same numbers. Passing an existing generator *into* an object is
refused: Python would share one stream with the caller, C++ would copy it.

**Evaluation order.** Python evaluates left to right; C++ leaves the order of operands and
arguments unspecified (GCC on x86 often goes right to left). So a statement may contain at
most one call that changes something (a random draw, a method that changes its object, a
function that changes an argument) and must not otherwise read what that call changes:
`p.step() + p.cash`, `push(a) + a[0]`, `p.cash += p.step()` and `rng.random() - rng.random()`
are refused with the hint to make the call on its own line. (Building this, the verifier's own
driver hit the same trap: reading a generator's four state words as the arguments of one call
got them in reverse order.)

**Also supported:** `field(default=..., default_factory=lambda: ..., init=False, compare=False)`;
`object.__setattr__(self, "x", value)` in a frozen class's `__post_init__`; `@classmethod`
alternate constructors with `cls(...)`; `@staticmethod`; nested objects (`acc.port.value(px)`);
`NamedTuple` indexing (`q[0]`) and unpacking (`bid, ask = q`); `==`/`!=` field by field.
Locals and parameters of a method that would hide a field get a trailing underscore, so
`-Wshadow` stays quiet.

**Refused, with the reason:** inheritance, class attributes with values (shared by every
object), property setters, defining `__eq__`, `__add__` and the like, `==` on an object holding an
array (Python raises), array defaults without `default_factory` (Python refuses them too),
`copy.copy` (shallow), assigning a field of a frozen object.

**Strings:** `str` parameters, fields, locals and returns (`std::string`), `==`, `!=`, ordering,
`+`, `s in ("a", "b")`, truthiness, string constants. Python orders strings by code point and
`std::string` by UTF-8 byte, which agree. The verifier's random strings come from the
module's own string literals, so branches like `if stance == "Long only":` run.

## Tested on real code

[`examples/black_litterman_app.py`](https://github.com/SunnyAlexV/qtrans/blob/main/examples/black_litterman_app.py) is the numerical core of
the Black-Litterman portfolio app (`github.com/SunnyAlexV/Black-Litterman-model`): 21
functions from `bl_core.py`, `bl_inference.py` and `bl_backtest.py`, including the posterior
mean and covariance, the He-Litterman Ω, Ledoit-Wolf shrinkage, variance ratios, the HAC
covariance and drifting buy-and-hold returns. The only change is type hints; every body is the
app's own code. The rest of the app is out of reach for now, for reasons that are on the
roadmap or out of scope: the optimiser and Idzorek's Ω call `scipy.optimize`; data handling
uses pandas; some helpers return strings or dicts; the bootstrap uses integer arrays.

- **Before 0.4:** with type hints added, 7 of the 20 functions first tried translated (and two
  translator bugs showed up, fixed: a missing `int`-to-`double` cast and complex-power
  classification).
- **Now: 21 of 21 translate**, and over 4,233 cases (random, plus realistic ones: 500 days of
  returns on 10 assets, three views) there are **0 mismatches**. 7 functions are bit-identical
  to NumPy; the others use LAPACK, `pow` or BLAS and agree to rounding. Per call, 1x to 40x
  faster at the realistic sizes: near 1x where NumPy already spends its time in optimised C
  (`_ewma_weights`; `ledoit_wolf_identity`'s matrix products), 15x to 40x where Python
  overhead dominates (`posterior_mu` 15x, `_cvar` 36x, `portfolio_stats` 37x).

What it turned up, beyond the missing features:

1. **Eigen's SVD-based pseudo-inverse loses accuracy on badly scaled matrices.** On random
   matrices mixing entries of 1e8 and 1e-2, rebuilding `pinv` from the SVD was off by 2e-10 to
   9e-9 against a 60-digit reference, where LAPACK stayed near 1e-15. Every Eigen SVD variant
   did the same; the column-pivoted QR route matches or beats NumPy: over the 93 hardest of
   600 random cases, median error 1.6e-15 against NumPy's 2.3e-11. A test keeps it that way.
2. **`np.maximum` and `np.minimum` broke signed-zero ties the wrong way**, including when the
   scalar comes first (`np.minimum(0.0, v)`).
3. **`np.asarray` was treated as a copy.** It returns its argument when that is already
   float64, so modifying the result modifies the caller's array.
4. **`S = S.copy(); S += ...` on a parameter was refused** as "modified in place and
   reassigned", though the copy comes first and the caller never sees the change.
5. **The verifier's ill-conditioning test was too strict for linear algebra.** It nudged each
   element by 4 ULPs of itself, but LAPACK and Eigen are accurate relative to a matrix's norm,
   so on badly scaled matrices two correct libraries differed by more than the nudge
   predicted. For functions using `np.linalg`, it now also nudges by 4 ULPs of the largest
   element.
6. **The benchmark could time an empty array** (the first random case) and report a
   meaningless speed-up. It now times a hand-picked case, or the largest random one.

## Projects: several modules

Real code is split across files. Give qtrans the files (or their folder) and it follows their
imports to the modules next to them:

```
python -m qtrans translate examples/bl_app -o cpp_out      # bl_config, bl_core, bl_inference, bl_backtest
python -m qtrans verify examples/bl_app --bench
python -m qtrans bind examples/bl_app --check
```

**One header and one namespace per module**, laid out as the Python is, so the C++ reads like
the code it came from. A module that imports from another includes its header and calls into
its namespace; types, constants and defaults are qualified on the way:

```cpp
// from tests/data/project/core.hpp (the // notes are added here): import rates as R, from rates import *
#include "rates.hpp"  // from rates import ...
namespace core {
double pv(double cf, double t, double rf = rates::DEFAULT_RF);   // a default from rates
double weights(qt::Vec& w);                                     // by reference: see below
inline double pv2(double cf, double t) {
    return cf * rates::discount(rates::DEFAULT_RF, t);          // R.discount(R.DEFAULT_RF, t)
}
inline double weights(qt::Vec& w) {
    rates::normalise(w);
    return qt::sum(w);
}
```

**The analysis is project-wide.** Modules are translated in import order, each after the ones it
imports, so a function knows exactly how every imported function treats its arguments:
`rates.normalise` divides its array in place, so `core.weights`, which passes its own array on,
takes it by reference, and so does a function two modules up that calls `core.weights`. The same
holds for methods of an imported class, a generator passed on to an imported function that
draws from it, an argument an imported function may hand back, and `try: ... except
ValueError` around an imported function that may raise a math domain error. Calling an imported
function that could not be translated is refused with its own reason ("calls bad(): bad (in
rates) could not be translated: module variable 'TABLE' was not translated: dicts are not
supported").

**Imports understood:** `from m import f`, `from m import f as g`, `from m import *` (public
names, or `__all__`, including what `m` itself imported, as Python does), `import m` and
`import m as alias` (then `alias.f`, `alias.Option`), and `from . import m` / `from .m import f`
in a package. A name defined in the module after it was imported wins, as in Python.
**Refused, with a message:** an import cycle (C++ headers can't include each other: "import
cycle: a -> b -> a"), two classes of one name in a project, and a class imported under another
name. Dotted imports of packages (`import pkg.sub`) aren't followed yet.

**Verify** builds one C++ driver for all the modules and checks the files given (`--all`: also
the modules they import). Hand-picked inputs can stay out of the code itself, in a
`qtrans_cases.py` next to the modules: `QTRANS_CASES = {"bl_core": {"nearest_psd": [...]}}`.
**Bind** builds one extension with a submodule per module (`bl_app_cpp.bl_core`, importable as
`import bl_app_cpp.bl_core`), so a class used by several modules is one Python type; names a
module imports are the same objects in its submodule; and when one file was given, its names are
also at the top level, so `import bl_core_cpp as bl_core` is a drop-in.

**Tested on the app's own files.** [`examples/bl_app`](https://github.com/SunnyAlexV/qtrans/tree/main/examples/bl_app) holds
`bl_config.py`, `bl_core.py`, `bl_inference.py` and `bl_backtest.py` from the Black-Litterman app,
with type hints added to the functions qtrans translates and one import line for them; a syntax-tree
comparison against the app's repository finds every one of the 53 function bodies unchanged.
`bl_core` and `bl_backtest` star-import `bl_config`, and `bl_backtest` star-imports `bl_core`.
**22 functions translate, 4,436 cases, 0 mismatches**, and the same through the bindings
([`examples/bl_app/RESULTS.md`](https://github.com/SunnyAlexV/qtrans/blob/main/examples/bl_app/RESULTS.md), regenerated by
`tools/project_example.py`). Its configuration constants come across with their comments
(`inline const double DEFAULT_VIEW_CONFIDENCE = 75.0;  // Idzorek confidence: ...`); its dict
constants (`FREQ_PER_YEAR`, `MARKETS`) are refused, and so is anything that uses them. The other
31 functions have no type hints: they work on pandas objects, call `scipy.optimize`, or format
text.

## Calling the C++ from Python

`python -m qtrans bind my_model.py --check --bench` compiles the translation into a Python
module, `my_model_cpp`, next to the file, then checks it against the Python and times both.
From Python: `build_extension(path)` and `check_bindings(path, bench=True)` in `qtrans.bind`.

```python
import black_litterman_app_cpp as bl
pi = bl.equilibrium_returns(2.5, Sigma, w_mkt)   # the same numbers, raising the same exceptions

import instruments_cpp as inst
p = Portfolio(weights, cash=1000.0)              # your own Python dataclass...
inst.run_rebalances(p, target, prices, 3)        # ...is read, and written back as Python would
```

**What crosses the boundary, and how.** The aim is that swapping the import changes nothing
but the speed, so where a conversion would silently change behaviour, the binding refuses.

| You pass | The C++ gets | Afterwards |
|---|---|---|
| numbers, strings, `None` | converted; an `int` parameter refuses a float | |
| an array the function only reads | a copy (any array-like; float64 C-ordered arrays are a plain `memcpy`) | |
| an array it modifies in place | your own float64 C-contiguous array, or `TypeError` (a converted copy would hide the change) | the change copied into your array, so views of it see it too |
| an `np.random.Generator` | its exact PCG64 state | the advanced state written back: your generator continues exactly where Python's would |
| one of your dataclass / class objects | its fields (no constructor runs) | if the function modifies it, the fields written back. Arrays and generators go into the objects it already holds when Python changes them in place, and are re-pointed when Python assigns a new one (decided per function from what it can call) |
| a compiled object (`inst.Portfolio(...)`) | the object itself | changed in place, as in Python |

Write-backs happen also when the call raises, since Python would leave the partial change.
Two arguments that share memory (the same array, a view of it, the same generator or object),
where one of them is modified, raise `ValueError`: in Python the change would show through
both, the compiled code works on separate copies. Exceptions come back as Python's:
`numpy.linalg.LinAlgError`, `ValueError`, `ZeroDivisionError`, `TypeError` (for `None` where a
number is needed), `IndexError`, `AssertionError`, `OverflowError`. Shape and bounds errors
raise the module's `ShapeError`, which is both a `ValueError` and an `IndexError`, because
NumPy raises one or the other and Eigen's checks can't tell which.

**Compiled classes** have the constructor (running `__post_init__` or `__init__`), the fields
as attributes (read-only on frozen classes and NamedTuples), methods, properties, static and
class methods, `==`, hashing for frozen classes (unhashable otherwise, as in Python), the
dataclass `repr`, `copy.copy`/`deepcopy`, tuple unpacking for a NamedTuple, and
`Cls._from_python(obj)` to convert one of your objects once instead of on every call.

**Checked like the C++ itself.** `--check` runs every function, method and constructor through
the compiled module and through Python on the same random and hand-picked inputs, with your
Python objects and again with compiled ones. It compares return values, in-place changes,
which arrays were changed in place and which re-pointed, generator states, and exceptions.
Over the eleven examples: **87 functions and 7 classes, 19,585 cases, 0 mismatches**
([`examples/BINDINGS.md`](https://github.com/SunnyAlexV/qtrans/blob/main/examples/BINDINGS.md), regenerated by `tools/bind_examples.py`).
Breaking the generator and array write-backs on purpose turns 92 cases of `instruments` and
30 of `monte_carlo` into mismatches, so the check sees what it is meant to see.

**Speed from Python**, per call, end to end (one run; timings vary between runs). A call costs
about 100 ns of overhead with number arguments and about 0.5 µs with two small arrays, so:

| Kind of code | Speed-up from Python | Examples |
|---|---|---|
| Calls to `scipy.stats.norm` | 200x to 1,700x | `bs_price` 496x, `implied_vol` 1,644x |
| Python loops over elements or objects | 20x to 115x | `max_drawdown` 114x, `leveraged_nav` 94x, `american_binomial` 29x |
| Array code at the examples' sizes | 2x to 16x | `black_litterman` 9x, `posterior_mu` 9x, `nearest_psd` 4x |
| Scalar arithmetic only | about 1x to 8x | `d1` 1.4x: the work is as small as the call overhead |
| Tiny functions, exp-bound Monte Carlo | 0.5x to 2x | `weight_bounds` 0.5x (a string compare); `european_from_draws` 0.6x, as for the standalone C++ |

**Not quite Python, by design:** a function returning its argument returns a copy, not the
same array; reading an array field of a compiled object gives a read-only copy (writing to it
would change nothing, so it can't be written); assigning an array to a field stores a copy;
reading a generator field gives a copy; a generator given to a compiled constructor is copied,
so the object and you no longer share it; `raise NotImplementedError` comes back as its parent
`RuntimeError`; `copy.copy` of a compiled object is a deep copy; and when a function re-points
an array field to an array *equal* to the old one, the compiled side keeps the old array (only
`is` can tell).

## Verification

`verify_file` / `python -m qtrans verify`:

1. **Builds a driver.** It translates the module, generates a C++ test driver, and compiles it
   with strict warnings and `-ffp-contract=off`. That flag stops the compiler fusing `a*b + c`
   into one FMA instruction, which rounds once where Python rounds twice.
2. **Builds inputs.** Your hand-picked cases from `QTRANS_CASES` in the module are used, plus
   200 seeded random ones per function. Random arrays share a size per call (Vec of n, Mat of
   n×n, with n from 0 to 6), and, since square inputs can't show a transposed dimension, the
   verifier also finds which non-square shapes the function accepts (each Vec of length n, m
   or 1, each Mat n×n, n×m, m×n, m×m, 1×m or n×1, kept when Python runs it) and uses them in
   40% of calls, with n ≠ m and either possibly 0 (an empty 0×m matrix) or 1. The 1s are where
   NumPy broadcasts: they found that a length-1 array met a longer one in plain Eigen code,
   which raised where NumPy stretches it (fixed: see the arrays table). Two deliberately planted shape
   bugs (a transposed `np.zeros_like`, `np.mean(axis=...)` dividing by the wrong dimension) are
   invisible to square inputs and give 69 and 72 mismatches with them
   (`tools/shape_mutants.py`). They also found a real bug: for a non-square matrix, NumPy's
   `np.linalg.solve` raises `LinAlgError` (which `except LinAlgError` catches), where the C++
   raised a shape error that escaped the handler. Fixed.

   `--profile stress` adds the numbers where floating point stops being ordinary: inf, NaN,
   ±1e308, subnormals, signed zeros, the doubles next to 1 and 0.5, rounding ties, in 30% of
   the inputs, and arrays up to 40 long. It found the int overflow below, and that NumPy
   itself can hang (its SVD, inside `np.linalg.pinv`, never returns on some matrices holding
   inf). So every Python call runs in a separate worker process with a 10-second limit; a call
   that doesn't finish is reported and left out, and the worker restarted.

   CI also compiles the drivers with AddressSanitizer and UndefinedBehaviorSanitizer
   (`QTRANS_CXXFLAGS`, see [Platforms](#platforms)). On the empty matrices above they found
   two cases of undefined behaviour that gave the right answers anyway: `A[:, -1]` of an
   n×0 matrix formed Eigen's column view (pointer arithmetic on a null pointer) before Eigen's
   bounds check raised the `IndexError`; and Eigen 3.4's triangular solve read element (0, 0)
   of an empty right-hand side. Rows and columns are now checked before the view is formed
   (`qt::row(A, i)`, `qt::col(A, j)`, with Python's negative indices), and `qt::solve` returns
   an empty result itself.
3. **Transfers values exactly.** Numbers go to C++ as hex floats and come back through
   `printf("%a")`, so nothing is rounded in transit.
4. **Compares everything a call produces.** That means the return value, the final state of
   every array the function modifies in place, and, for a generator argument, its next raw
   output after the call. Equal raw outputs show both sides consumed the same number of draws
   (two different states giving the same 64-bit output has probability about 2^-64).
5. **Classifies every case:**
   - **match**: within `rtol=1e-10` / `atol=1e-13`, or the same exception category. Arrays are
     compared normwise, relative to the array's largest value.
   - **ill-cond**: further apart than that, but within 100× of how far *Python's own* answer
     moves when the inputs are nudged by 4 ULPs (for functions using `np.linalg`, also by 4 ULPs
     of a matrix's largest element, the size of error LAPACK itself is allowed). There the
     inputs, known only to rounding, don't pin the answer down further: two backward-stable
     implementations (as LAPACK and Eigen are) can differ by that much. It does not mean no
     implementation could do better, e.g. one in higher precision. Also ill-cond: for functions
     using matrix products or `np.linalg`, one side raising `LinAlgError` ("Singular matrix")
     where the other returns, when the nudge itself flips Python between raising and
     returning, or when a matrix the Python code gave `np.linalg` to factor was singular to
     working precision (smallest singular value at most n·eps times the largest; for
     `cholesky`, smallest eigenvalue at most n·eps times the largest): an exactly-zero pivot
     is then a matter of rounding; and results that agree
     except at elements that are inf, -inf or NaN on both sides (a value that is zero up to
     rounding, `1e-16` from LAPACK and exactly `0` from Eigen, divided by zero: its sign, so
     rounding, decides between `-inf` and NaN). These cases are listed separately, never hidden.
   - **known**: Python raised a float error that IEEE-754 arithmetic doesn't raise
     (`ZeroDivisionError`, `OverflowError`, math domain error), or a Python float went complex
     (`(-8.0) ** (1/3)`), where C, like `np.float64`, gives NaN: in the result, in an array
     (or added into one in place, where NumPy raises a `TypeError`), or anywhere earlier in the
     call (`v > x ** 0.5` compares by the real part in NumPy and with NaN in C++; on a
     mismatch the verifier reruns Python with every `a ** b` watched); or an int left 64 bits, where
     Python's int grows and C++ raises `OverflowError`.
   - **edge**: linear algebra (matrix products, `np.linalg`) on inf, NaN, subnormal or
     near-overflow inputs, where NumPy's BLAS/LAPACK and Eigen add in different orders, so
     `inf - inf` gives NaN on one side only, or an intermediate overflows on one side only.
     Both are correct; listed, not counted as mismatches. Scalar code gets no such allowance.
   - **mismatch**: anything else, or a `numpy.linalg` operation in the call whose residual
     exceeds its backward-stability bound ([Linear algebra](#linear-algebra-checked-by-what-it-guarantees)).
     A correct translation has none.

Module-level settings the verifier reads (the translator ignores names starting `QTRANS_`):
`QTRANS_CASES = {"f": [(args...), ...]}` (trailing defaults may be omitted, arrays as NumPy
arrays) and `QTRANS_TOLERANCE = {"f": 1e-8}` for an algorithm that guarantees less than 1e-10.
No example needs `QTRANS_TOLERANCE`, and every hand-picked case is an outright match.

## Linear algebra, checked by what it guarantees

Comparing an answer with NumPy's only works when the problem is well conditioned. For a
matrix with condition number 1e12, two correct `solve`s can differ in the fourth digit, and
eigenvectors are only defined up to sign. So every `numpy.linalg` operation qtrans translates
is also judged by its defining property, the way numerical analysis judges LAPACK (N. Higham,
*Accuracy and Stability of Numerical Algorithms*, 2nd ed.; u = 2^-53 is the unit roundoff):

| operation | what is measured | bound |
|---|---|---|
| `solve` | the residual B − AX, componentwise via the LU factors (Theorem 9.4) | γ₃ₙ ‖\|L\|\|U\|\|X\|‖, plus the rounding of computing B − AX |
| `inv` | AX − I, the same way | the same |
| `det` (and the LU of `solve`, `inv`) | PA − LU (Theorem 9.3) | 2γₙ ‖\|L\|\|U\|‖ + u‖A‖ |
| `cholesky` | A − LLᵀ (Theorem 10.3) | (γₙ₊₁ + γₙ) ‖\|L\|\|Lᵀ\|‖ + u‖A‖ |
| `eigh` / `eigvalsh` | AV − V diag(w), VᵀV − I, ascending order | p(n)·u‖A‖ and p(n)·u, p(n) = 10n |
| `pinv` | the four Penrose conditions (AXA = A, XAX = X, AX and XA symmetric), each scaled; AXA = A up to the singular values `rcond` drops | p(n)·u |

γₖ = ku/(1 − ku). The first four are rigorous bounds, evaluated for the very matrices at hand,
so a ratio above 1 can only be a bug. For `eigh` and `pinv` the literature gives a "modest
constant p(n)" without a value; 10n is qtrans's choice, and the worst ratio measured below is
about half of it. A ratio of at most 1 means the result is the exact answer for a matrix within
rounding of the input. That is what LAPACK guarantees too, and it holds when the C++ and
NumPy disagree. (For LU it is relative to |L||U|: on Wilkinson's matrix, where partial pivoting
grows entries by 2^(n−1), the bound grows with it, for LAPACK as for Eigen.)

**In every verification.** `qtrans verify` compiles its driver with `-DQTRANS_LINALG_AUDIT`:
each of these operations, wherever it is called (an intermediate `solve` inside a longer
function too), measures its residual against its bound, and a call where any operation exceeds
it fails, as a mismatch would, even if the values agree with NumPy. The audit only reads: the
headers qtrans writes for you leave it out (the macro is off), so they carry no cost. The
report says how many operations were audited, the largest ratio, and how many of the
`ill-cond` and `edge` cases had every operation within its bound. Over the examples:
1,718 cases ran 4,339 operations, the largest ratio 0.528 (`eigh`, VᵀV − I, in `posterior_sigma`), none unstable (also in [`examples/RESULTS.md`](examples/RESULTS.md)). The bindings check
reuses the same C++ functions, so it isn't audited separately.

**Operation by operation.** `python -m tools.linalg_check --per-band 4` builds 2,549
matrices: sizes 0 to 50; condition numbers from 1 to 1e18 by construction (U diag(s) Vᵀ, the
singular values spread geometrically or one of them small); rows and columns scaled from 1e-8
to 1e8; Hilbert and Wilkinson matrices; rank-deficient and exactly singular ones; symmetric
with repeated, negative and zero eigenvalues; rectangular ones for `pinv`; empty, 1×1, zero,
1e-300 and 1e300. It runs each operation on all of them, in C++ with the audit and in NumPy,
and checks three things. The worst ratio of residual to bound, by the matrix's condition number:

| operation | matrices | audited | cond < 1e4 | cond 1e4 .. 1e10 | cond >= 1e10 | singular to working precision | worst overall |
|---|---:|---:|---:|---:|---:|---:|---:|
| `solve` | 479 | 463 | 0.266 | 0.096 | 0.085 | 0.081 | 0.266 |
| `inv` | 479 | 463 | 0.143 | 0.056 | 0.085 | 0.081 | 0.143 |
| `det` | 479 | 476 | 0.113 | 0.045 | 0.085 | 0.081 | 0.113 |
| `cholesky` | 314 | 254 | 0.497 | 0.301 | 0.215 | 0.214 | 0.497 |
| `eigh` | 314 | 312 | 0.386 | 0.492 | 0.490 | 0.445 | 0.492 |
| `pinv` | 484 | 483 | 0.170 | 0.101 | 0.313 | 0.279 | 0.313 |

Errors: the C++ raises `LinAlgError` (or a shape error) exactly where NumPy does, except where
the matrix is singular to working precision (smallest singular value at most n·2u times the
largest; for `cholesky`, smallest eigenvalue at most n·2u times the largest in size). There an
exactly-zero pivot, or a non-positive one, is a matter of rounding, and LAPACK and Eigen, adding
in different orders, can decide differently. And agreement as far as the conditioning allows:
`solve` and `inv` differ from NumPy by at most (‖R_C++‖ + ‖R_NumPy‖)/σ_min(A), R the two
residuals (X − X_exact = A⁻¹R exactly, so this is rigorous up to computing R); `det` is
compared with a 50-digit [mpmath](https://mpmath.org) determinant, within exp(‖A⁻¹‖‖E‖) − 1 for
its LU backward error E; eigenvalues differ by at most what the two residuals allow (first
order); for `cholesky` and `pinv` the difference is reported against the first-order bound
κ(A)(η_C++ + η_NumPy), not judged:

| operation | same error / both return | one side raises, matrix singular to working precision | unexplained | compared | worst difference / allowed |
|---|---:|---:|---:|---:|---:|
| `solve` | 472 | 7 | 0 | 362 | 0.520 |
| `inv` | 472 | 7 | 0 | 362 | 0.104 |
| `det` | 479 | 0 | 0 | 299 | 0.054 |
| `cholesky` | 306 | 8 | 0 | 249 | 0.026 (reported) |
| `eigh` | 314 | 0 | 0 | 311 | 0.343 |
| `pinv` | 484 | 0 | 0 | 343 | 0.808 (reported) |

CI runs it on every platform and compiler in the matrix (under the sanitizers and at -O0 and
-O3 too) and keeps the report with the run.

What it found: **`pinv` broke down on nearly rank-deficient matrices.** For full rank,
`qt::pinv` used a column-pivoted QR (complete orthogonal decomposition), more accurate than the
SVD formula on badly scaled matrices; but on a matrix with condition number near 1e15 its
result had AX off symmetric by 1e13 times the bound (NumPy's: 0.006 of it), the pseudo-inverse
12 times too large. The QR result is now used only when it meets all four Penrose conditions;
otherwise the SVD formula, as NumPy. Both earlier tests (the badly scaled matrices, against a
60-digit reference) and the new ones pass. A test also checks that the audit catches a wrong
answer: a 1e-9 relative change in one entry of a correct result lifts the ratio from at most
0.2 to between 7,900 (`pinv`) and 440,000 (LU) ([`tests/test_linalg_audit.py`](tests/test_linalg_audit.py)).

## Fuzzing

The examples, the app and the tests are code someone wrote on purpose, so they only reach the
combinations of features someone thought of. `python -m tools.fuzz` writes random functions in
the subset qtrans translates (scalars, ints, control flow, arrays, slicing, broadcasting, masks,
`np.where`, reductions, linear algebra, and calls between the functions, arrays changed in place
through them included), type by type, so each is valid Python; translates them, compiles them,
and checks them against Python on the verifier's inputs (non-square, empty and length-1 arrays,
and every other batch the stress profile's inf, NaN and extreme values). A crash, a compile
error, a warning or a mismatch is isolated to its function and shrunk automatically, by
deleting and simplifying the program while it still fails the same way, to a minimal
reproducer. CI runs it on every push on the same 500 programs (seed 1, `--batches 25`), and
by hand (Actions → fuzz → Run workflow) on new ones, a new seed for 30 minutes
([`fuzz.yml`](https://github.com/SunnyAlexV/qtrans/blob/main/.github/workflows/fuzz.yml)); a
failure uploads the reproducer with the run. A long run on new seeds goes before each release.

On this release, `python -m tools.fuzz --seed 107 --minutes 40` (on the reference platform)
generated 2,400 functions: 1,519 translated and checked on 60,760 inputs with no failure, and
881 refused with a reason. Most refusals are calls to a function that was itself refused, or
two calls in one statement that both change an argument (Python runs them left to right; C++
leaves the order unspecified). The CI check (`--seed 1 --batches 25`) is clean too. The
development runs before them (seeds 1 to 9 and 101 to 106, and CI's first run: 13,660
functions, 9,614 translated, 384,522 inputs) found 22 failures, which come down to the 12 items
below.

What it found, each fixed and kept as a test
([`tests/data/fuzz_regressions.py`](https://github.com/SunnyAlexV/qtrans/blob/main/tests/data/fuzz_regressions.py)):

- **Array sizes are Eigen's `Index`** (`long`), not `long long`: `min(n, len(v))` and
  `len(v) // 2` didn't compile.
- **Arrays used only as temporaries** (`float(np.array([...]) @ w)` in a function with scalar
  parameters) left the header without the array runtime: it didn't compile.
- **A slice in a product or a broadcast** (`v[:2] @ A`, `A / v[1:]`, `np.diag(v[::-1])`):
  Eigen's own slice view claims direct memory access but has none, which didn't compile.
  Slices are now strided maps over the array's memory (a negative step a negative stride).
- **Unused locals and parameters** (assigned and never read, or read only in a branch the
  translation drops, like the `else` of `if True:`) drew warnings; they are now
  `[[maybe_unused]]`.
- **A fresh value passed to a function that changes its argument in place** (`f(v + 1)`) was
  refused; it now goes through `qt::lvalue`, which gives C++ the temporary to change, as Python
  does. A *view* passed to it (`f(v[1:])`, `f(A.T)`) didn't compile: in Python the change goes
  through the view into the original, which the C++ doesn't do, so it is refused with that reason.
- **NumPy treats a boolean index with no elements as an empty index**, whatever the array's
  shape: `v[np.array([], dtype=bool)]` is `[]` and assigning through it changes nothing. The
  C++ raised `IndexError`.
- **`A[1] = A[:, j]`**, a column copied into a row of the same matrix: NumPy copies the two
  overlapping views element by element without a temporary and reads an element it has just
  overwritten (on `np.arange(25.).reshape(5, 5)`, row 1 becomes `[3, 18, 13, 18, 23]`, not
  column 3). The C++ gave column 3. Refused, with `.copy()` as the hint; every other
  overlapping assignment NumPy gets right (`v[1:] = v[:-1]`, `A[0:2] = A[1:3]`, `A[:] = A.T`)
  and the C++ agrees.
- **`v[::-1] = v`** gave `[0, 1, 2, 1, 0]` for `np.arange(5.)`, not `[4, 3, 2, 1, 0]`.
  NumPy computes the right side before storing it; the C++ asked Eigen for a temporary with
  `.eval()`, which makes one of an expression but returns a plain array itself, so the store
  read elements it had already overwritten. A plain array on the right is now copied.
- **A parameter set but never read in the C++**: `p = 0.0; p = p` (`p = p` becomes a
  comment, as Clang warns on a self-assignment), or `p` read only in an `if False:` branch
  and assigned after it. GCC 13 warns (`-Wunused-but-set-parameter`); such parameters are
  `[[maybe_unused]]` now.
- **The verifier compared a complex result as if it were real.** Python's `x ** 0.5` with
  `x < 0` is complex; put in an array, the verifier's float64 copy of Python's result dropped
  the imaginary part (with a `ComplexWarning`) and compared the real part with the C++'s NaN.
  Added into an array in place, NumPy raises `UFuncTypeError`, which the verifier didn't
  recognise as a `TypeError`. And in `v[v > x ** 0.5]` no complex value reaches the result
  at all (NumPy compares by the real part, C++ with NaN). All three are the documented
  complex-power difference now.
- **Two verifier rules were too strict** for linear algebra whose answer rounding decides:
  `np.linalg.solve(A, A) / 0` (Eigen solves exactly, NumPy leaves `1e-16` off the diagonal:
  NaN against `-inf`), and `np.linalg.inv(np.exp(P))` on a matrix singular to working
  precision (LAPACK returned, Eigen met an exactly-zero pivot). Both are `ill-cond` now, by
  the rules above.
- **Which of two errors is raised** when one expression breaks two checks: documented under
  [Differences kept by design](#differences-kept-by-design), and counted as `known` only when
  both sides raise. (GCC evaluates a call's arguments right to left and raises the other
  error; Clang goes left to right and raises Python's.) The other error can be one C++ meets
  only because of an IEEE-754 difference: `int(math.log(0.0))` is `int(-inf)`, an
  `OverflowError`, where Python already raised `ValueError` at the `log`.

## Platforms

CI ([`.github/workflows/tests.yml`](https://github.com/SunnyAlexV/qtrans/blob/main/.github/workflows/tests.yml))
runs the tests on every push, in these configurations:

| Job | What it shows |
|---|---|
| **Reference**: Ubuntu 24.04, x86-64, GCC 13, Python 3.13, NumPy and SciPy pinned ([`ci/requirements-pinned.txt`](https://github.com/SunnyAlexV/qtrans/blob/main/ci/requirements-pinned.txt)) | Where this README's numbers come from. It also runs `run_examples.py --check`: every header in `examples/cpp_out/` and every count in `examples/RESULTS.md` must reproduce, and so must the error and ULP columns of every function that doesn't use matrix products or `np.linalg` (OpenBLAS picks its kernels for the CPU, so NumPy's own rounding there changes from one CPU to another) |
| Ubuntu latest, latest NumPy and SciPy | An early warning: a new compiler or NumPy release that changes a result shows up here first |
| Clang | The generated C++ doesn't depend on GCC |
| `-O0` and `-O3` | The numbers don't depend on the optimisation level (`-ffp-contract=off` and no `-ffast-math`: no operation is reordered or fused) |
| AddressSanitizer + UndefinedBehaviorSanitizer | No out-of-bounds access, signed overflow or other undefined behaviour in the generated code or the runtime, on any input the verifier tries |
| Python 3.10 and 3.14 | The oldest and newest supported Python |
| Windows | MinGW g++ for the verifier, MSVC for the bindings |
| macOS ARM64 | Apple Clang on another CPU, C library and NumPy build |

Results built from `+ - * /` and `sqrt` are exactly rounded by IEEE-754, so they can be the
same bits on every platform. Results that use the C library's or NumPy's own `exp`, `log`,
`pow` or `erf` are checked against the verifier's tolerance everywhere; their ULP counts are
those of the reference platform. `QTRANS_CXXFLAGS` adds flags to the verifier's compile command
(after its own, so `-O3` replaces `-O2`).

What the first runs found, all fixed:
- **macOS**: with two compiled modules loaded, `LinAlgError` from one came out as a plain
  `ValueError`. pybind11's exception translators were process-wide, and each module's copy of
  `qt::linalg_error` is a different type to macOS's C++ library (which compares types by
  identity; Linux's compares names). Each module now translates only its own exceptions.
- **Python 3.14** words math domain errors by the argument ("expected a nonnegative input,
  got -1.0" for `math.sqrt(-1)`), which the verifier didn't recognise as the known
  IEEE-754 difference.
- **ARM**: NumPy's own `np.clip` returns a different zero than on x86-64 when `x` ties with a
  bound (`np.clip(0.0, -0.0, None)`); IEEE-754 leaves `max(-0, +0)` unspecified. The C++
  follows x86-64; on ARM only the sign of such a zero differs.
- **Clang** warned about `w = w;`, from `w = w.copy()` on a parameter passed by value (now a
  comment: the parameter is already the function's own copy).
- **Sanitizers**: the two undefined behaviours on empty matrices described under
  [Verification](#verification).

## Differences kept by design

Each is listed in the per-module report next to the functions it affects.

- **Float errors.** Python raises on `x / 0.0`, on overflow in `**` and `math.exp`, and on
  `math.log(-1)`. C++ follows IEEE-754 and returns ±inf/NaN. Integer `//` and `%` by zero
  *do* throw, as in Python.
- **64-bit integers.** Past ±9.2e18, C++ raises `OverflowError` instead of returning a big int.
  Until 0.8.1 that held only for float-to-int conversion: int `+ - *` and `**` wrapped around
  silently (and signed overflow is undefined behaviour in C++), which the stress profile
  found. Now int arithmetic is checked (`qt::iadd`, `qt::imul`, ...) unless the translator can
  show the result fits, from the range each operand is known to lie in: literals, array sizes
  (below 2^45), and `range()` loop indices (`i + 1` for `i in range(n)` stays plain, since
  `i < n`). Across the examples' headers that is 46 checked operations, mostly on function arguments
  (`n_steps + 1`, `2 * n`); loop and index
  arithmetic stays plain C++. `LLONG_MIN % -1` (undefined in C++) gives 0, as in Python.
- **Last-bit differences** where NumPy or SciPy use their own `exp`, `pow`, `erf` or BLAS/LAPACK.
- **Which of two errors is raised.** When one expression breaks two checks at once (shapes that
  don't broadcast, and an index out of range: `np.ones(3) / np.zeros(2) * A[64]`), Python
  raises the first, left to right. C++ leaves the order in which a call's arguments are
  evaluated unspecified (GCC and MSVC usually go right to left, Clang left to right), so it may
  raise the other. Both reject the input; matching Python's order would mean splitting every
  expression that can raise into its own statement. The verifier counts these as `known` only
  when both sides raise an error that rejects the input (`ValueError`, `IndexError`, a division
  by zero), never when one side returns a value.
- **`assert`** always runs in C++; `python -O` strips it.
- **Shape and bounds checks** are always on, as they are in NumPy. Defining your own
  `eigen_assert` before including the header turns them off.
- **Objects whose fields are the same NaN:** Python's dataclass `==` compares field tuples,
  which treat one float object as equal to itself even when it is NaN; C++ compares values.
- **LAPACK on `inf`:** for inputs containing `inf`, LAPACK's `eigh`/`pinv` results are erratic
  (NaN in one case, an identity matrix in another) and are not reproduced. NaN is.
- **Optional arrays** (`P: Mat | None`) are read, tested and rebound, but not modified in place:
  `std::optional<qt::Mat>` holds a copy, where Python would modify the caller's array.

## What is supported

- **Types:** `float`, `int`, `bool`, `str`, `Vec`, `Mat`, `np.random.Generator`, `tuple[...]`
  of those, `X | None` (also `Optional[X]`) for a number or array, `@dataclass` and `NamedTuple`
  classes, and `-> None`.
  Parameters and returns are annotated; locals are inferred.
- **Statements:** assignment (including tuple unpacking, swaps, `a[i] = x`, `a[i:j] = v`,
  `A[i] = row`, `A[:, j] = col`), augmented assignment, `if`/`elif`/`else`, `while`,
  `for` over `range(...)` or over a `Vec`, `break`, `continue`, `return`, `raise` with a
  string, `assert`, `try`/`except np.linalg.LinAlgError` or `ValueError`, `with np.errstate(...)`.
- **Scalar library:** `math.*`, scalar NumPy functions, `scipy.stats.norm.cdf/pdf/ppf/sf`
  with `loc`/`scale`, `scipy.special.erf/erfc/ndtr/ndtri`, and the builtins `abs`, `min`,
  `max`, `float`, `int`, `bool`, `round`, `pow`.
- **Array library:**
  - Creation: `zeros`, `ones`, `full`, `eye`, `linspace`, `arange`, `array([...])`,
    `*_like`, `copy`.
  - Elementwise: `exp`, `log`, `sqrt`, `abs`, `maximum`, `minimum`, `where`, `isnan`, ... and
    `norm.cdf/pdf` on arrays.
  - Reductions: `sum`/`mean`/`std`/`var` (with `axis`, `ddof`, `keepdims`), `max`, `min`,
    `argmax`, `argmin`, `cumsum`/`cumprod` (with `axis`), `sort`, `quantile`, `percentile`,
    `median`, `trace`, `any`/`all`/`count_nonzero` of comparisons.
  - Shapes: `v[:, None]`, `v[None, :]`, `.ravel()`, `.flatten()`, `.astype(float)`,
    `column_stack`, `vstack`, `hstack`, `concatenate`, `clip`.
  - Masks: `x[mask]`, `A[row_mask]`, `x[mask] = c`, `x[mask] += c`; stored comparisons
    (`tail = pnl <= -var`) become `qt::Mask`.
  - Linear algebra: `@`, `dot`, `outer`, `.T`, `diag`, `einsum`,
    `np.linalg.solve/inv/cholesky/det/eigh/eigvalsh/pinv`.
  - Attributes: `.shape`, `.size`, `.ndim`, `len()`.
- **Random numbers:** `np.random.default_rng(seed)`, and `Generator.random`,
  `standard_normal`, `normal` and `uniform`.

Everything else is **refused with a line number and a hint**, and anything that calls a
refused function is refused too. [`examples/not_yet_supported.py`](https://github.com/SunnyAlexV/qtrans/blob/main/examples/not_yet_supported.py)
shows the messages. A refusal is always preferred to a translation that might compute a
different number.

## What building this turned up

Most of these are bugs in the *Python* examples, found because the verifier insisted on an
explanation for every disagreement:

1. **Implied vol returned the wrong vol.** The first stopping rule was absolute
   (`|price error| < 1e-12`). For near-zero prices it passes immediately, and the solver
   returned 0.05 for options whose vol was 0.2. With a relative rule, all 84 well-posed test
   cases now recover the true vol to 1.3e-11.
2. **The binomial tree accepted arbitrage.** Random inputs with p = (e^{rΔt} − d)/(u − d)
   outside (0, 1) produced values growing like p^n (in one case p ≈ 10 over 97 steps), which
   is garbage in both languages. The pricers now refuse such trees.
3. **Textbook Black-Litterman is numerically fragile.** The information form inverts
   `inv(τΣ) + P'Ω⁻¹P`, which becomes singular as a view grows confident. Against a 50-digit
   reference it is accurate to only about 6 digits at Ω = 1e-14 (relative error 1.4e-6),
   while the algebraically identical Woodbury form keeps full precision (`tools/black_litterman_stability.py`). The example uses the Woodbury
   form.
4. **A parity check compared rounding noise.** C − P − (Se^{−qT} − Ke^{−rT}) is a difference
   of numbers around K whose true value is 0. It now returns the gap relative to S + K.
5. **A slow quantile.** The first `qt::quantile` sorted the whole array, 9x slower than NumPy.
   `std::nth_element` finds the same two order statistics in linear time, so the result is
   unchanged bit for bit and the speed now matches NumPy's.
6. **Translator bugs caught by the same process:**
   - `round(1e300)` was undefined behaviour in C++, and now raises.
   - `norm.cdf` with `scale <= 0` returned a number where SciPy gives NaN.
   - Self-referencing slice updates needed `.eval()`.

## Project layout

```
qtrans/
  translator.py      module level: imports, constants, signatures, parameter passing, header
  functions.py       per function: scan, loops, inference, aliasing, declarations, emission
  expressions.py     scalar expressions, Python semantics, library calls
  arrays.py          array expressions: arithmetic, broadcasting, indexing, NumPy functions
  classes.py         @dataclass / NamedTuple: fields, defaults, methods (stage 5)
  structs.py         object expressions: fields, properties, methods, constructors, ==
  core.py, types.py, library.py, report.py, typing.py (Vec, Mat)
  verify.py          compile, run, compare (incl. conditioning), benchmark
  explain.py         Python next to its C++, row by row, with reasons (text, HTML, JSON)
  inspect_.py, doctor.py   what qtrans will make of a file; whether a machine is ready
  bind.py            pybind11 bindings: generate, build, check against Python
  project.py         several modules: imports followed, import order, one header each
  runtime/qtrans_runtime.hpp   scalar helpers (standard library only)
  runtime/qtrans_arrays.hpp    array helpers on Eigen, NumPy algorithms ported
  runtime/qtrans_random.hpp    NumPy's Generator (SeedSequence + PCG64 + ziggurat)
  runtime/qtrans_pybind.hpp    the bindings' conversions, write-backs and exceptions
examples/            11 quant modules (one from a real app, one of classes), their C++
                     output, RESULTS.md, BINDINGS.md, three C++ consumers; bl_app/: the
                     app's own four modules, as a project
tests/               414 tests: generated text, refusals, compiled semantics, every example,
                     projects, the Python bindings
tools/               scripts that back the numbers in this README; fuzz/: the program fuzzer
                     (generator, checker, shrinker: `python -m tools.fuzz`); linalg_check/:
                     numpy.linalg checked operation by operation
ci/                  the reference job's pinned packages
run_examples.py      press-Run entry point
```

## Roadmap

1. ~~Scalars and control flow~~
2. ~~Arrays~~ (`float64`, 1-D and 2-D)
3. ~~Monte Carlo~~ (NumPy's generator reproduced exactly)
4. ~~Real code~~ (0.4: a real app's numerical core, unchanged but for type hints)
5. ~~Classes~~ (0.5: `@dataclass`, `NamedTuple`, strings; 0.6: plain classes, a generator held
   by an object)
6. ~~Tooling~~: a pybind11 wrapper so the C++ can be called back from Python (0.7), PyPI
   (0.7), multi-module projects (0.8).
7. **Trust**: 0.8.1 brought rectangular, empty and extreme-value inputs to the verifier,
   checked int arithmetic and a time limit on Python calls. 0.9 so far: Clang, sanitizers,
   optimisation levels, Python 3.10/3.14, Windows and macOS ARM64 in CI
   ([Platforms](#platforms)); a program fuzzer with automatic minimisation
   ([Fuzzing](#fuzzing)); every `numpy.linalg` operation audited for backward stability,
   in each verification and operation by operation ([Linear algebra](#linear-algebra-checked-by-what-it-guarantees)).
   `qtrans explain` (Python and C++ side by side, with reasons), `inspect` and `doctor`
   ([Python and C++ side by side](#python-and-c-side-by-side)). **Next (0.10)**: the side-by-side
   view inside VS Code, filling in as each line or block is written; traceable headers (source hash,
   Python line references, CMake); a reproducible benchmark protocol.
8. **Later (1.0)**: a plugin API mapping approved Python functions to existing C++ (QuantLib,
   in-house libraries), integer arrays (the bootstrap), a written specification of the
   supported subset. Not planned: pandas, all of Python, GPU backends.

## Licence

MIT (see `LICENSE`). `qtrans/runtime/qtrans_random.hpp` includes NumPy's ziggurat tables under
NumPy's BSD 3-Clause licence, reproduced in that file.
