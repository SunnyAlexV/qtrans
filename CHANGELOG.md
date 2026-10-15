# Changelog

## 1.1.0

**NumPy's own maths, to the bit.** `np.exp`, `np.log`, `np.power` and 17 more float64 functions
gave the C library's result in the C++, where NumPy runs its own code on many machines; they
agreed to rounding. They now run the code NumPy runs, so they agree to the bit.

- **What NumPy runs, measured** on 12 releases (1.21 to 2.5), with AVX-512, as an AVX2 machine and
  at baseline: on Linux x86-64 with AVX-512, Intel's SVML for 19 functions (its high-accuracy
  routines from NumPy 2.0, the low-accuracy ones in 1.22 to 1.26); NumPy's own `exp` / `log`
  (Tang's method) on other AVX-512 machines and in NumPy 1.21 to 1.22; NumPy's own `tanh` on any
  CPU with FMA SIMD (AVX2 PCs, ARM64, so Windows and Apple Silicon too); the C library elsewhere.
  `sin` / `cos`, `power` / `arctan2` and `tanh` moved between these in particular releases.
- **NumPy's code, not a rewrite.** The runtime carries NumPy's SVML assembly verbatim
  (`qtrans_svml_ha.hpp`, `qtrans_svml_la.hpp`) and NumPy's own C kernels (`qtrans_npy_own.hpp`),
  written by `tools/vendor_svml.py` from NumPy's source distribution (checked against PyPI's
  SHA-256), under NumPy's BSD-3 licence; `tanh` is a lane-by-lane port of NumPy's kernel.
  `qtrans_npmath.hpp` picks among them at the first call as NumPy does: platform, CPU, NumPy's
  `NPY_DISABLE_CPU_FEATURES` / `NPY_ENABLE_CPU_FEATURES` (read as each release reads them), and
  the NumPy release qtrans ran with (`qtrans_numpy.hpp`, new; `-DQTRANS_NUMPY_VERSION` overrides).
  On Linux it binds the glibc symbol versions NumPy's wheels call, so even NumPy's C-library
  path matches, NaN signs included; on Apple ARM it does as NumPy's `npy_sin` / `npy_cos` /
  `npy_tan` do there, `x - x` for an infinity or NaN (found by CI: Apple's `cos(-nan)` is `+nan`).
- **Powers as NumPy computes them.** `a ** s` keeps NumPy's shortcuts (-1, 0, 0.5, 1, 2), now also
  for an exponent known only at run time; `np.power(a, s)` follows its release (no shortcut before
  2.1, `x * x` from 2.1, all five from 2.3: the 1.x translation of `np.power(x, 2)` as `x * x`
  was right only from NumPy 2.1); an array exponent that is constant along NumPy's loop is a
  scalar to it, here too. `np.arctan2` on arrays, and `np.exp2`, `np.cbrt`, `np.arcsinh`,
  `np.arccosh`, `np.arctanh` on arrays, are translated; `scipy.stats.norm.pdf` uses NumPy's
  `exp`, as SciPy does.
- **Checked:** `qtrans.npmath_check` compares the runtime with the running NumPy on every function
  (arrays, numbers, all bit patterns, the edges, each power form); `tools/npmath_versions.py`
  runs it on each release and CPU setting ([docs/NUMPY_MATHS.md](docs/NUMPY_MATHS.md): 204,574,320 values on 12 releases in 3 CPU settings, 0 different);
  `tests/test_npmath.py` runs it on every CI platform; `qtrans doctor` on yours. Where a value
  differs, it reports the bits, and how NumPy and the C++ treat that value alone, in pairs, at each
  place of a short array and strided, so one report shows where the difference comes from.
- **`np.sum` faster, the same bits:** its eight partial sums are one AVX-512 register where the
  CPU has it (not on Windows: GCC bug 54412), added in NumPy's order; checked on every length.
- **`np.sum` of more than 8192 values follows the NumPy release.** Before NumPy 2.3 a reduction
  goes through NumPy's 8192-element buffer: each chunk is summed pairwise and the chunk sums are
  added in turn. With NumPy 2.2 or older, 1.0 used 2.3's single pairwise sum, so long sums
  (and `mean`, `var`, `std`, sums along a row) could differ in the last bits; they now match,
  checked on every release from 1.21 to 2.5.
- **Results:** in the examples, 45 of the 87 functions are bit-identical to Python (32 in 1.0),
  5 of the 8 Monte Carlo pricers among them; vectorised `np.exp` is 1.2× NumPy's speed (it was
  3.6× slower). `qtrans doctor` reports which code NumPy's maths runs on your machine.
- The spec ([docs/SPEC.md](docs/SPEC.md), new section 12.1) says so, and its versioning rule now
  says what this release does: where 1.x agreed with Python to rounding, a later 1.x may agree
  to the bit, never the other way.

## 1.0.2

The benchmark page made self-contained and its protocol complete, after review. What a
translation computes is unchanged.

- **The headline with its denominator**, at the top of [docs/BENCHMARKS.md](docs/BENCHMARKS.md):
  the median speed-up across the verified example functions on that machine, its range, and why
  it is not a property of qtrans alone (function, input size, libraries, hardware, baseline).
- **What is timed, stated:** per-call time of already-built code only; the arguments built
  before timing; one untimed warm-up call, then runs of 1, 4, 16, ... calls until one lasts
  0.1 s, and the calls timed per round reported; why the compiler may inline the call yet can't
  remove it.
- **One-off costs measured on their own:** translating, compiling, building the bindings and
  importing them, none of which is in any timing.
- **The machine in full:** CPU family, model and stepping, logical CPUs, SIMD (AVX2, FMA,
  AVX-512), whether it is a virtual machine, the CPU governor where the OS exposes it. On Linux
  the benchmark pins itself to one CPU (`--no-pin` to turn that off).
- **A robust spread:** the interquartile range over the median, in place of slowest / fastest.
- **m/i/k/x defined** above the tables, and rows with known differences (†) or ill-conditioned
  cases (‡) marked, so no row is taken for plain matches.
- `qtrans verify --bench` makes one untimed warm-up call on each side before timing, and records
  the calls timed per round (`FunctionCheck.py_reps`, `cpp_reps`); the driver's BENCH line
  carries them.

## 1.0.1

Evidence, not features: three documents, each generated by a script so every number reruns.
What a translation computes is unchanged.

- **A benchmark protocol** (`tools/benchmark.py` writes [docs/BENCHMARKS.md](docs/BENCHMARKS.md)).
  Every function is verified before it is timed and its counts are shown next to its timings;
  several rounds, Python and C++ taking turns; medians and the spread. Workloads with a fair
  baseline (`tools/bench_workloads.py`): the same book of 10,000 options priced by a Python
  loop and NumPy-vectorised, a 100,000-path Monte Carlo taken apart into its draws and its
  `np.exp`, a book of bonds solved by Newton; and the compiled module called from Python
  (`qtrans bind`), conversions included. It shows where qtrans loses: NumPy's SIMD `exp`
  against the C library's, one element at a time; porting it is next on the roadmap.
- **A worked example** (`tools/worked_example.py` writes
  [docs/WORKED_EXAMPLE.md](docs/WORKED_EXAMPLE.md)): a pricer with implied volatility taken
  through translate, verify, timing and export, with each command's real output.
- **An architecture document** ([docs/ARCHITECTURE.md](docs/ARCHITECTURE.md)): the passes of
  the translator, where each guarantee of the specification is kept, the runtime headers, how
  verification and the fuzzer check the result.
- `qtrans verify --bench --rounds N` (and `verify_file(..., bench=N)`): N rounds, Python and
  C++ taking turns, medians; `--bench` alone is one round, as before.
- The timed C++ call uses every element of a returned tuple and every field of a returned
  object, so no part of the result can be left uncomputed (it used the first only).

## 1.0.0

- **Python's float errors by default, everywhere.** A float division by zero, an overflowing
  `**`, `math.exp` or `math.cosh`, and a math domain error (`math.sqrt(-1)`, `math.log(0.0)`)
  raise `ZeroDivisionError`, `OverflowError` and `ValueError` in the C++ as in Python, with the
  running Python's message, in every function (until 0.14: only in code a `try` could catch them
  in). The examples' known differences fall from 1,623 of 17,921 cases (in 43 functions) to 40
  (in 2, all a float going complex), with 0 mismatches.
- **`# qtrans: ieee`** on a `def`, or at the top of a file for every function of the module,
  keeps IEEE-754's inf and NaN instead (fewer checks); the report says so at each function, and
  `qtrans verify` counts those inputs as known. Inside a `try` that may catch the error it has no
  effect.
- **The cost**, measured by `tools/float_error_cost.py` (the same modules with and without
  `# qtrans: ieee`, nine scalar pricing functions, same inputs, best of 7 alternating runs): 3%
  to 18% per call with GCC 13, 1% to 11% with Clang 18. Each check is one compare marked
  unlikely; the throw is a separate cold function. An int divisor known not to be zero
  (`1.0 / (k + 1)` for `k` in `range(n)`) has no check. NumPy arrays are unchanged.
- **A value that may be a NumPy number or a Python float** depending on the run
  (`max(np.exp(x), 0.0) / y`) gets NumPy's inf or NaN, and the report says so (Python raises on
  some runs and not on others); inside a `try` that may catch the error it is refused, as before.
- **Python's order of evaluation where it shows**: when both operands of an operator, or
  several arguments of a call of the file's own functions, methods or classes, can raise, the
  left one is worked out first, into a temporary (`const double left = ...;`), so the same error
  comes out as in Python (`math.log(S / K) / (sigma * math.sqrt(T))` with `S = K = 0`, `T < 0`:
  `ZeroDivisionError`, not `ValueError`). A call can raise only if the function's own C++ can.
  The object a method is called on comes before the arguments, an argument that changes
  something (modifies an argument or an object, prints, draws) is ordered against one that
  raises, and `xs[k] += v` finds `xs[k]` (IndexError, KeyError) before working out `v`. A library
  call whose arguments mix a change and an error (`max(bump(p), a / b)`) is refused, with a hint.
- **Checked by an independent review** before release (a separate agent, not the one writing the
  code): its 9 + 3 confirmed problems are fixed, each kept as a test
  (`tests/test_float_errors_default.py`, `tests/data/review10/`). Among them: a temporary named
  like a field of the class, a `try` around another module's `# qtrans: ieee` function, method
  or constructor (refused again), and calls found by their C++ names (overload versions, renamed
  keywords, imports under another name, `std::make_shared<Sub>`).
- `self.x` inside a method is that class's field when qtrans works out whether a value is
  NumPy's: a field of the same name in another class no longer turns it into "may be NumPy's".
  A comprehension's variable over a `list[float]` is a Python float.
- Storing into a list or dict held in a field (`self.xs[k] += 1.0`) is refused with the reason;
  it crashed the translator (also in 0.14).
- **Found by the program fuzzer before release**, each kept as a test:
  - `x /= y` on a Python float (a variable, a list element, a dict value, a field) raised
    nothing (C++'s `/=`); it raises ZeroDivisionError now, as `x / y` does. NumPy elements keep inf.
  - **A variable read where some path may not have set it** (`if c: n = 1.0` then `return n`):
    Python raises UnboundLocalError there, and the C++ read the value it started with (0),
    which the report only noted (since 0.1). Now such a variable carries a flag and the read
    raises Python's error, in its words (NameError at a program's top level); a variable always
    set first carries none. The paths are worked out as Python runs them (`qtrans/unbound.py`).
    Only a number, a bool or a str can carry a flag; another type read that way is refused.
- A `qtrans_python.hpp` cached by an earlier qtrans is accepted only if it holds every message
  the runtime uses; otherwise it is written again (a stale one failed to compile).
- **`qtrans verify` is stricter**: in a function that raises Python's float errors itself,
  Python raising where the C++ doesn't is a mismatch, no longer a known difference.
- An uncaught exception in a program prints Python's last traceback line
  (`ZeroDivisionError: float division by zero`) on stderr.
- A catch around a function of another module of the project that may raise a float error is
  translated (that module raises it itself now).
- `docs/SPEC.md` describes 1.0: the guarantee as a design rule checked by differential testing,
  not a proof; ints as Python's within 64 bits; the differences kept by design, updated. From
  1.0 on, a change to what it says is a change of major version, except for additions.

## 0.14.0

- **Integer arrays.** `np.arange` of ints, `np.zeros(n, dtype=int)`, `np.ones(n, dtype=int)`,
  `np.full(n, 3)`, `np.array([1, 2, 3])`, `x.astype(int)` and `rng.integers(..., size=n)` are
  int64 arrays, as in NumPy (`IVec`, `qt::IVec`), where they were float arrays: they print as
  NumPy prints int64 arrays, `//` and `%` keep NumPy's int rules (0 for a zero divisor), and
  `+`, `-`, `*`, `sum` wrap at 2**64. `/`, float operands and the float functions convert first,
  as NumPy does. `x[idx]` gathers (the bootstrap), `for i in idx:`, `idx[i]`, `idx[a:b]`,
  `i ** 2`, `2 ** i` (int64, wrapping; a negative exponent raises `ValueError`, as in NumPy),
  `len`, `.sum()`, `.max()`, `.min()`, `.cumsum()`, `.argmax()`, `.mean()`, `np.sort`, `np.abs`.
  `IVec` parameters and returns, in `qtrans verify`, `qtrans bind` (int64 NumPy arrays in and
  out) and `qtrans export`.
- **`Generator.integers`**, NumPy's draws exactly (Lemire's method on 32-bit draws when the range
  fits): the generator's state now includes the half word NumPy keeps for its next 32-bit draw,
  in `qtrans verify`'s comparisons, in objects handed to C++ and in the bindings.
- Refused, with the reason: 2-D int arrays, `x[idx] = ...`, masks into int arrays,
  `np.unique`, `np.argsort`, `np.bincount`.
- **The specification**, `docs/SPEC.md`: what qtrans translates, what the C++ computes, what it
  refuses, and the differences kept by design, section by section. Its 32 examples are checked by
  `tests/test_spec.py` (translated with the quoted C++, refused with the quoted reason, or run as
  programs that must print what Python prints), so the document can't drift from the code.
- An `async def` is listed in the report as refused, where it was only a note.

## 0.13.0

- **Class hierarchies.** A class may derive from one class of the same file, plain or
  `@dataclass`, abstract (`abc.ABC`, `@abstractmethod`) or not. Objects of a hierarchy live
  behind a `std::shared_ptr`, which is how Python holds every object: a `Payoff` variable,
  parameter, field or list element holds a `Call` or a `Put`, calls go to the object's own
  method (`virtual` / `override` / `= 0`), and assigning or passing an object shares it.
  `super().__init__(...)` becomes the base in the constructor's initializer list,
  `super().value(s)` becomes `Call::value(s)`, an inherited `__init__` becomes `using
  Base::Base;`, a dataclass subclass takes its base's fields first. Also `isinstance(p, Call)`
  (`std::dynamic_pointer_cast`), `type(p).__name__`, `p is q`, `==` between plain objects
  (identity, as in Python), `print(p)` with each class's own repr, `list[Payoff]`, and a
  variable given a `Call` here and a `Put` there typed as their common base.
- Refused, with the reason: two bases, a base from another module, mixing `@dataclass` and plain
  classes, an override with other parameter names, types or defaults, creating an object with
  abstract methods left, `super().__init__` not first, `self` passed on as a value,
  `@classmethod` in a hierarchy, `==` on dataclass objects of a hierarchy, and a statement that
  calls something that may change a shared object while reading another name for it.
- `qtrans verify` builds random objects of every concrete class for a `Payoff` parameter
  (`@Call` travels before the fields), and compares objects a function may change after the
  call. `qtrans bind` leaves class hierarchies out for now, saying so.
- **Python's random module**, reproduced exactly (`qtrans_pyrandom.hpp`): CPython's Mersenne
  Twister seeded as CPython seeds it, and `random`, `getrandbits`, `randrange`, `randint`,
  `uniform`, `triangular`, `gauss`, `normalvariate`, `lognormvariate`, `expovariate`, `choice`,
  `shuffle`, `sample` and `choices` as `Lib/random.py` computes them, with the running Python's
  messages and defaults. A program must seed it before drawing (Python seeds it from the OS);
  `qtrans verify` seeds both sides before each call of a function that draws and compares the
  generator afterwards. Checked on 4,453 calls and 73 seeds, on Python 3.10, 3.13 and 3.14.
- **`math.fsum`**: CPython's algorithm (exact partials, half-even rounding across them, its errors).
- **Fixed: `x ** 2` is Python's again.** It was written `x * x`, but Python calls the C library's
  `pow`, and glibc's `pow(x, 2.0)` differs from `x * x` in the last bit for about 1 value in
  1,000. Compilers make the same substitution (`std::pow(x, 2.0)` becomes `x * x`, and
  `std::pow(x, -1.0)` becomes `1.0 / x`, at `-O2`), so `**` and `math.pow` now go through
  `qt::pow`, which keeps the call. `np.power(x, 2)` is NumPy's square (`x * x`), and stays so.
- `qtrans_python.hpp` holds a facts version, so a cached copy written by an older qtrans is
  written again.

## 0.12.0

- **Exceptions, as Python has them.** Your own exception classes (`class PricingError(ValueError):
  pass`, any depth), `try` / `except` (several classes in one clause, `as e` with `str(e)`,
  `print(e)`, `f"{e}"`, `type(e).__name__`) / `else` / `finally`, `raise` with or without a
  message, `raise ... from ...`, a bare `raise` to re-raise. `finally` runs once on every way out:
  the end, `return` (its value computed first), `break`, `continue`, an exception; an error it
  raises replaces the one on its way.
- **A C++ exception family with Python's shape** (`qt::Exception`, `qt::ValueError`,
  `qt::LookupError`, `qt::KeyError`, ...), replacing the standard library's types, under which
  ValueError, IndexError, ZeroDivisionError and AssertionError were all `logic_error`s.
  `qtrans verify` now compares the exact class (`LinAlgError`, `PricingError`), where it compared
  a category; the bindings raise your classes as classes of the compiled module.
- **Python's float errors, where a handler may catch them.** In code an `except ZeroDivisionError`
  / `ArithmeticError` / `OverflowError` / `ValueError` / `Exception` guards (the function holding
  the `try` and everything its `try` block calls), `1.0 / 0.0`, `0.0 ** -1`, `10.0 ** 400`,
  `math.sqrt(-1)`, `math.exp(1000)`, `math.log(0.0)`, ... raise as CPython does (its `math_1` and
  `float_pow` rules). NumPy's numbers keep NumPy's behaviour (no error). These were refused.
- **Python's own words**: `qtrans_python.hpp` holds the error messages of the running Python,
  written from it by making each error happen (3.14 says "division by zero" where 3.13 says
  "float division by zero", and "expected a nonnegative input, got -1.0" for `math.sqrt(-1)`;
  `math.pow(0.0, -inf)` raises up to 3.10 only). Checked against Python 3.10 to 3.14
  (`tests/test_exceptions.py`): 22,957 values and errors, 21,261 on 3.10, all identical.
- **`math.gamma` and `math.lgamma` are CPython's own code** (its Lanczos approximation, ported
  operation for operation), not the C library's: the values are Python's to the last bit, and
  `lgamma(5e-324)` is 744.44 on Windows too (the Windows C library gives inf, which the first CI
  run of 0.12.0 caught).
- `assert` messages are Python's (`str(e)` is the message, or `""`), and `list.index`,
  `min([])`, `max([])` raise with the running Python's wording.
- A `print()` line is worked out left to right, as Python does (a braced list, where the `+` of
  C++ strings has no order): with two values that may raise, the first error is the one raised,
  and `print(f(a), f(b))` of functions that print themselves is no longer refused.

## 0.11.3

From setting qtrans up on a fresh Windows PC (a new user's steps, with the console check of
0.11.1 passing there: `straße οδος CAFÉ` from Python and from the C++, in code page 437):

- `qtrans translate` prints the command that builds a program as verify built it: the compiler
  qtrans found, by its full path when it isn't on the PATH (it printed a bare `g++`, which cmd
  can't run then), `-ffp-contract=off` (without it a compiler may fuse a * b + c into one
  rounding, and the results stop being Python's), `-static` with MinGW (the .exe then runs
  without MinGW's DLLs on the PATH), and the Eigen folder it found; for MSVC, the `cl` command.
- With no compiler, `qtrans doctor` says which winlibs.com download to take (newest GCC, UCRT,
  Win64, Zip archive) and that it can be unzipped anywhere: C:\, the Desktop or Downloads,
  inside the extra folder Windows' Extract All makes; qtrans finds it in all of these.
- Eigen 5 (what `conda install -c conda-forge eigen` now installs): `qtrans doctor` shows its
  version (it showed `?`), the whole test suite passes with Eigen 5.0.1 as with 3.4.0, and
  `run_examples.py --check` reproduces all 87 rows with it. CI's latest-packages job now uses it.

## 0.11.2

- A Python file saved with a UTF-8 byte-order mark (Windows Notepad, and Windows PowerShell 5's
  `Out-File` / `Set-Content -Encoding utf8`, write one) is read as Python reads it; qtrans
  stopped with `SyntaxError: invalid non-printable character U+FEFF`. `qtrans live` keeps the
  mark when it saves, as it keeps the line endings.
- `qtrans --version`.

## 0.11.1

- A program that prints text beyond ASCII shows it in a Windows console window as Python does.
  Python writes characters to the console (WriteConsoleW); the C++ wrote UTF-8 bytes, which the
  console read in its code page (`straße` came out as `straÃe`). Such a program now starts
  with `qt::py::windows_console console;`, which writes characters there too and changes
  nothing anywhere else (files, pipes, Linux, macOS: UTF-8, as Python writes in UTF-8 mode, the
  default from 3.15). `tools/windows_console_check.py` shows both side by side under Wine.
- The runtime headers are plain ASCII, so MSVC reads them the same in every code page.

## 0.11.0

- **Strings:** Python's str methods (`split`, `strip`, `replace`, `find`, `count`, `startswith`,
  `upper`, `title`, `casefold`, `isdigit` and the other tests, `zfill`, `center`, ...), `s[i]`,
  `s[a:b:c]`, `for ch in s`, `"x" in s`, `s * n`, and `float(s)` / `int(s)` with Python's syntax
  and error messages. Indexes count characters. Case, character classes, digits and `repr()`
  are exact for every Unicode character: `qtrans_unicode.hpp` is written from the running
  Python (shipped for Unicode 15.1; written once into the user's cache for another version).
  `repr()` of text now escapes what Python escapes (`'\xa0'`, `'\u200b'`), where it used to
  keep every non-ASCII character as it was.
- **Tuples in lists:** `list[tuple[str, float, int]]` parameters and returns, `for a, b in
  pairs`, `sorted(pairs)` / `min` / `max` ordering tuples as Python does (NaN in them stops the
  C++, since Python's answer then depends on its sort algorithm), `list(zip(...))`,
  `list(enumerate(...))`, `dict(zip(keys, values))`.
- **`key=`:** `sorted(xs, key=...)`, `xs.sort(key=..., reverse=...)`, `min` / `max` with `key`
  and `default`; a key is a lambda, a function or `d.get`, called once per element as Python
  does.
- **`for ... else` and `while ... else`:** the `else` runs when the loop wasn't left by `break`.
- **Lists and dicts as conditions:** `if not xs:`, `while queue:`.
- A module-level name that is also a C library name (`nan`, `exp`) no longer clashes.

## 0.10.1

Found by looking at what a user sees (`tools/screenshots.py`, new: screenshots of `qtrans live`,
`qtrans explain` and the terminal, from the real commands):

- A dict lookup that fails is `KeyError` on both sides: the checker reported the C++'s as
  `IndexError` (so verify counted mismatches that weren't), and the Python bindings raised
  `IndexError` too; they now raise Python's own `KeyError`, holding the key.
- A function's parameters and locals are its own: a parameter named like a script's variable no
  longer pulls that variable out of `main()` (which shadowed it, two compiler warnings).
- `sum(xs, start)`: with a float start the result is a float even for an empty list (the usual
  way to keep `sum()` a float), added from the start as CPython adds floats; also for
  generators, `sum((p * w for ...), 0.0)`.
- A program using a speed hint (`# qtrans: unroll 4`) compiles: its `.cpp` now defines the hint
  macros too.

## 0.10.0

**Programs.** A whole script translates: top-level code (or `if __name__ == "__main__":`)
becomes `int main()` in a self-contained `<module>.cpp`, and `qtrans verify` runs both and
compares what each printed, character for character, and the exit status.

- Python's printing, exactly: `print()`, f-strings, `str()`, `repr()`, `format()`,
  `"...".format()`, `%` formatting, `round(x, n)`; floats as Python's shortest repr, format specs
  by CPython's rules, arrays as NumPy prints them, NumPy scalars as `np.float64(...)`.
- Types from how a file calls its functions, one C++ version per kind of call where calls differ.
- A variable that starts as an int and later holds floats prints as Python prints it (a flag
  beside it follows Python's type).
- Lists of floats, ints and strings (`std::vector`), with Python's indexing, slicing, `sum`
  (CPython's compensated float sum), `sorted`, comprehensions and printing.
- Dicts with str or int keys (`qt::py::dict`, insertion order, `KeyError`), loops over keys,
  values and items, dict comprehensions.
- `enumerate` and `zip` loops (also over dict views) as index loops; generators read by `sum`,
  `min`, `max`, `sorted`, `list` and `str.join`; `any` / `all` stopping where Python stops.
- Objects print as Python prints them (dataclass repr, `__repr__`, `__str__`).
- Ints surely beyond 64 bits are refused at translation (`2 ** 100`), not left to overflow.

**Your own C++ for a function** (`<module>_overrides.hpp`, `qtrans override`): used in place of
the translation, kept across re-translations, checked against the Python and against qtrans's
own translation on the same inputs, and flagged when the Python it was written for changes
(`--accept` re-verifies; `verify --strict` fails while out of date). Also for functions qtrans
can't translate itself.

**Speed hints in the Python** (`# qtrans: unroll 4`, `inline`, `noinline`, `hot`, `cold`,
`likely`, `unlikely`), as portable macros; hints that could change a result are refused.

**`qtrans export`**: a CMake package (header-only target, install rules, package config) whose
golden tests check the C++ against Python's own recorded results with no Python needed, and a
program's exact output. The floating-point flags exactness depends on travel with the target;
every generated header refuses to compile under fast-math and names the Python (and its
SHA-256) it came from.

**`qtrans live`**: edit Python in the browser and see the C++ update as you type, with the
reason for each line and verification on demand.

**Fixed along the way**: list arguments from a program, a comprehension's own loop variable
taken for an outer loop's, `verify` reporting PASS when nothing was translated (it now says
what wasn't checked), dict and list sharing rules (a function changing a dict passed to it),
Clang's sign-conversion warning on index loops.

Earlier versions: see the README's roadmap.
