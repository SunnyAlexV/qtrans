# Changelog

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
