# Changelog

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
