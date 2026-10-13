# Changelog

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
