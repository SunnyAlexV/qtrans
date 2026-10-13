"""qtrans_python.hpp: the error messages and math-error rules of the Python that runs qtrans.

Python's wording changes between versions: `1.0 / 0.0` raises ZeroDivisionError("float division
by zero") up to 3.13 and ZeroDivisionError("division by zero") in 3.14; `math.sqrt(-1)` says
"math domain error" up to 3.13 and "expected a nonnegative input, got -1.0" in 3.14. A program
that prints the error it caught prints that text, so the C++ has to have the same. Rather than
keep a table per version, this asks the running Python: it makes each error happen and writes
down what Python said, and, where the message holds the value, where the value goes (checked
on two values, so a pattern that only looks right by accident is caught).

The header is written once per Python version into the user's cache (like qtrans_unicode.hpp),
and the C++ is checked against the same Python by `qtrans verify`.

    python -m qtrans.pyfacts > some/folder/qtrans_python.hpp
"""
from __future__ import annotations

import math
import os
import sys
import tempfile
from pathlib import Path

# --- fixed messages: (C++ name, how to make Python raise it) -----------------------------------
_FIXED = {
    "int_floordiv_zero": lambda: 1 // 0,
    "int_mod_zero": lambda: 1 % 0,
    "int_truediv_zero": lambda: 1 / 0,
    "float_truediv_zero": lambda: 1.0 / 0.0,
    "float_floordiv_zero": lambda: 1.0 // 0.0,
    "float_mod_zero": lambda: 1.0 % 0.0,
    "int_divmod_zero": lambda: divmod(1, 0),
    "float_divmod_zero": lambda: divmod(1.0, 0.0),
    "min_empty": lambda: min([]),
    "max_empty": lambda: max([]),
    "zero_negative_power": lambda: 0.0 ** -1.0,
    "int_zero_negative_power": lambda: 0 ** -1,
    "float_pow_overflow": lambda: 10.0 ** 400.0,
    "int_of_nan": lambda: int(float("nan")),
    "int_of_inf": lambda: int(float("inf")),
    "list_index": lambda: [][0],
    "str_index": lambda: ""[0],
    "pop_empty": lambda: [].pop(),
    "pop_index": lambda: [1].pop(5),
    "math_range": lambda: math.exp(1000.0),
    "math_domain_pow": lambda: math.pow(0.0, -1.0),
    "math_domain_fmod": lambda: math.fmod(1.0, 0.0),
}


def _message(f) -> tuple[str, str]:
    try:
        f()
    except Exception as e:  # noqa: BLE001 - the point is to see what Python raises
        return type(e).__name__, str(e)
    raise AssertionError("expected an error")


def _template(messages) -> tuple[str, str, bool, bool]:
    """messages: [(the texts the value may be written as, message)]. -> (prefix, suffix, holds
    the value, written as an int): one pattern for every probe, or AssertionError."""
    best = None
    for pick in range(2):  # the value written as repr(float(v)) or as repr(v)
        pats = set()
        for texts, msg in messages:
            text = texts[min(pick, len(texts) - 1)]
            i = msg.rfind(text)
            pats.add((msg[:i], msg[i + len(text):], True, pick == 1 and len(texts) > 1) if i >= 0 else
                     (msg, "", False, False))
        if len(pats) == 1:
            best = pats.pop()
            if best[2] or pick == 1:
                return best
    if best is not None:
        return best
    raise AssertionError(f"no single message pattern: {messages}")


def _list_index_template():
    msgs = []
    for v in (7, 12):
        try:
            [1, 2].index(v)
        except ValueError as e:
            msgs.append(((repr(v),), str(e)))
    return _template(msgs)


# --- math functions: when they fail and what they say ------------------------------------------
# The C library's answer decides, as in CPython's math_1: NaN from a non-NaN argument is a domain
# error (ValueError), an infinity from a finite one an overflow (OverflowError) for the functions
# that can overflow and a domain error for the others (log(0), atanh(1)). gamma and lgamma are
# CPython's own: nonpositive integers (and -inf for gamma) are domain errors.
UNARY = ["sqrt", "exp", "log", "log10", "log2", "log1p", "expm1", "sin", "cos", "tan", "asin", "acos", "atan",
         "sinh", "cosh", "tanh", "asinh", "acosh", "atanh", "erf", "erfc", "cbrt", "exp2", "fabs", "gamma", "lgamma"]
CAN_OVERFLOW = {"exp", "expm1", "sinh", "cosh", "exp2", "gamma", "lgamma"}
_C_NAME = {"gamma": "tgamma"}
# the argument's classes: a message may differ between them (3.14's log(0) for an int says no "got")
_CLASSES = ("finite", "zero", "neginf", "posinf")
_PROBES = (-1000.0, -3.0, -2.25, -1.5, -1.0, -0.5, 0.0, -0.0, 0.5, 1.0, 1.5, 2.0, 2.25, 3.0, 1000.0, -math.inf, math.inf)


def _class(x: float) -> str:
    return "zero" if x == 0 else "neginf" if x == -math.inf else "posinf" if x == math.inf else "finite"


def _math_messages(name):
    """{(class, 'float' | 'int'): (prefix, suffix, holds value)} for the domain errors of
    math.<name>, one pattern per class of argument (checked on every probe that fails)."""
    f = getattr(math, name)
    found = {}
    for p in _PROBES:
        for kind in ("float", "int"):
            if kind == "int" and not (math.isfinite(p) and p == int(p)):
                continue
            arg = int(p) if kind == "int" else p
            try:
                f(arg)
            except ValueError as e:
                found.setdefault((_class(p), kind), []).append(((repr(float(arg)), repr(arg)), str(e)))
            except OverflowError:
                pass
    return {k: _template(v) for k, v in found.items()}


def _cstr(s: str) -> str:
    return '"' + s.replace("\\", "\\\\").replace('"', '\\"') + '"'


def generate() -> str:
    version = ".".join(map(str, sys.version_info[:3]))
    lines = [f"// qtrans_python.hpp: the error messages of Python {version}, and when its math functions raise.",
             "// Written by qtrans.pyfacts from that Python, by making each error happen; do not edit.",
             "#pragma once", "", "#include <cmath>", "#include <string>", "",
             f'#define QTRANS_PYTHON_VERSION "{version}"', "", "namespace qt {", "namespace msg {", ""]
    for key, f in _FIXED.items():
        lines.append(f"inline constexpr const char* {key} = {_cstr(_message(f)[1])};")
    try:  # math.pow(0.0, -inf): ValueError up to 3.10, inf from 3.11 (IEEE 754)
        math.pow(0.0, -math.inf)
        zero_neg_inf = False
    except ValueError:
        zero_neg_inf = True
    lines.append(f"inline constexpr bool math_pow_zero_to_minus_inf_raises = {'true' if zero_neg_inf else 'false'};")
    pre, suf, has, _ = _list_index_template()
    lines += ["", "// list.index(x) without x: the message holds repr(x) or not",
              f"inline constexpr const char* list_index_prefix = {_cstr(pre)};",
              f"inline constexpr const char* list_index_suffix = {_cstr(suf)};",
              f"inline constexpr bool list_index_has_value = {'true' if has else 'false'};", ""]
    lines += ["// math.<f>(x) outside its domain: prefix, repr(x) (when has_value), suffix; by the class of x"]
    lines.append("struct DomainText { const char* prefix; const char* suffix; bool has_value; bool as_int; };")
    for name in UNARY:  # (one this Python lacks, cbrt and exp2 before 3.11, gets the old wording)
        msgs = _math_messages(name) if hasattr(math, name) else {}
        for kind in ("float", "int"):
            cells = []
            for cls in _CLASSES:
                t = msgs.get((cls, kind)) or msgs.get((cls, "float"))
                cells.append("{" + (f"{_cstr(t[0])}, {_cstr(t[1])}, {'true' if t[2] else 'false'}, "
                                    f"{'true' if t[3] else 'false'}" if t else '"math domain error", "", false, false') + "}")
            lines.append(f"inline constexpr DomainText {name}_{kind}[] = {{{', '.join(cells)}}};")
    lines += ["", "}  // namespace msg", "}  // namespace qt", ""]
    return "\n".join(lines)


SHIPPED = Path(__file__).parent / "runtime" / "qtrans_python.hpp"  # for the Python qtrans was built with


def _version_of(text: str):
    for line in text.splitlines()[:12]:
        if line.startswith("#define QTRANS_PYTHON_VERSION"):
            return line.split('"')[1]
    return None


def header_path() -> Path:
    """qtrans_python.hpp for the running Python: the shipped one when it was written by this very
    Python version, else one written once into the user's cache."""
    version = ".".join(map(str, sys.version_info[:3]))
    try:
        if _version_of(SHIPPED.read_text(encoding="utf-8")) == version:
            return SHIPPED
    except OSError:
        pass
    tag = f"{sys.implementation.name}-{version}"
    base = Path(os.environ.get("XDG_CACHE_HOME") or Path.home() / ".cache") / "qtrans"
    text = None
    for folder in (base, Path(tempfile.gettempdir()) / "qtrans"):
        path = folder / f"python-{tag}" / "qtrans_python.hpp"
        if path.exists():
            return path
        try:
            text = text or generate()
            path.parent.mkdir(parents=True, exist_ok=True)
            tmp = path.with_suffix(f".{os.getpid()}.tmp")
            tmp.write_text(text, encoding="utf-8")
            os.replace(tmp, path)
            return path
        except OSError:
            continue
    raise RuntimeError("qtrans: could not write qtrans_python.hpp for this Python")


if __name__ == "__main__":
    sys.stdout.write(generate())
