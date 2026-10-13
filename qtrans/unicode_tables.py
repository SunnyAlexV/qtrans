"""qtrans_unicode.hpp: the Unicode facts Python's str methods use, taken from Python itself.

str.upper(), isalpha(), float("١٢") and the rest follow the Unicode tables built into the
Python that runs them (3.11 has Unicode 14, 3.13 has 15.1, ...). The header is written from the
running Python, character by character, by asking that Python: so the C++ answers as that
Python does, whatever its Unicode version. qtrans ships one for the version it was built with,
and writes a fresh one next to the translation when the running Python has another.

    python -m qtrans.unicode_tables > qtrans/runtime/qtrans_unicode.hpp
"""
from __future__ import annotations

import sys
import unicodedata
from pathlib import Path

SHIPPED = Path(__file__).parent / "runtime" / "qtrans_unicode.hpp"
MAX = 0x110000


def _ranges(pred):
    out, start = [], None
    for c in range(MAX):
        ok = pred(chr(c))
        if ok and start is None:
            start = c
        elif not ok and start is not None:
            out.append((start, c - 1))
            start = None
    if start is not None:
        out.append((start, MAX - 1))
    return out


def _case_sets():
    """Two properties CPython's case methods use and Python doesn't expose: 'cased' (title()
    lowers a letter after a cased character) and 'case-ignorable' (lower() looks past these for
    the end of a word, to write a capital sigma there as the final form). Both are read off
    title() and lower() themselves."""
    ignorable, cased = [], []
    for c in range(MAX):
        ch = chr(c)
        if 0xD800 <= c <= 0xDFFF:
            continue
        is_cased = ("a" + ch + "b").title().endswith("b")
        if is_cased:
            cased.append(c)
            # a cased character ends a word for the sigma after it unless it's also ignorable
            if (ch + "\u03a3").lower()[-1] != "\u03c2":
                ignorable.append(c)
        elif ("A" + ch + "\u03a3").lower()[-1] == "\u03c2":
            ignorable.append(c)
    return _compress(ignorable), _compress(cased)


def _compress(points):
    out = []
    for c in points:
        if out and out[-1][1] == c - 1:
            out[-1] = (out[-1][0], c)
        else:
            out.append((c, c))
    return out


def _mapping(fn):
    rows = []
    for c in range(MAX):
        ch = chr(c)
        if 0xD800 <= c <= 0xDFFF:
            continue
        m = fn(ch)
        if m != ch:
            assert len(m) <= 3, (hex(c), m)
            rows.append((c, [ord(x) for x in m] + [0] * (3 - len(m))))
    return rows


def _title(ch):
    # capitalize() writes the first character in title case (CPython's _PyUnicode_ToTitleFull);
    # ch.title() of one character is that mapping
    return ch.title()


_HELPERS = r"""// --- reading the tables ---------------------------------------------------------------------
template <std::size_t N>
const Range* find_range(const Range (&t)[N], char32_t c) {
    const Range* r = std::upper_bound(t, t + N, c, [](char32_t x, const Range& q) { return x < q.lo; });
    return r != t && c <= (r - 1)->hi ? r - 1 : nullptr;
}
template <std::size_t N>
bool in(const Range (&t)[N], char32_t c) { return find_range(t, c) != nullptr; }

// UTF-8: the character starting at byte k (and how many bytes it takes)
inline char32_t decode(const std::string& s, std::size_t k, std::size_t* bytes = nullptr) {
    const unsigned char c = static_cast<unsigned char>(s[k]);
    const int n = c < 0x80 ? 0 : c >= 0xF0 ? 3 : c >= 0xE0 ? 2 : 1;
    char32_t v = n == 0 ? c : (c & (0x3Fu >> n));
    for (int i = 1; i <= n && k + static_cast<std::size_t>(i) < s.size(); ++i)
        v = (v << 6) | (static_cast<unsigned char>(s[k + static_cast<std::size_t>(i)]) & 0x3Fu);
    if (bytes) *bytes = static_cast<std::size_t>(n) + 1;
    return v;
}
inline void encode(std::string& out, char32_t c) {
    if (c < 0x80) {
        out += static_cast<char>(c);
    } else if (c < 0x800) {
        out += static_cast<char>(0xC0 | (c >> 6));
        out += static_cast<char>(0x80 | (c & 0x3F));
    } else if (c < 0x10000) {
        out += static_cast<char>(0xE0 | (c >> 12));
        out += static_cast<char>(0x80 | ((c >> 6) & 0x3F));
        out += static_cast<char>(0x80 | (c & 0x3F));
    } else {
        out += static_cast<char>(0xF0 | (c >> 18));
        out += static_cast<char>(0x80 | ((c >> 12) & 0x3F));
        out += static_cast<char>(0x80 | ((c >> 6) & 0x3F));
        out += static_cast<char>(0x80 | (c & 0x3F));
    }
}
"""


def generate() -> str:
    version = unicodedata.unidata_version
    props = {
        "alpha": _ranges(str.isalpha),
        "decimal": _ranges(str.isdecimal),
        "digit": _ranges(str.isdigit),
        "numeric": _ranges(str.isnumeric),
        "upper": _ranges(str.isupper),   # one character: CPython's Py_UNICODE_ISUPPER
        "lower": _ranges(str.islower),
        "title": _ranges(lambda ch: unicodedata.category(ch) == "Lt"),  # CPython's TITLE_MASK
        "space": _ranges(str.isspace),
        "printable": _ranges(str.isprintable),  # repr() keeps these as they are, escapes the rest
    }
    props["case_ignorable"], props["cased"] = _case_sets()
    maps = {"to_upper": _mapping(str.upper), "to_lower": _mapping(str.lower), "to_title": _mapping(_title),
            "to_fold": _mapping(str.casefold)}
    # the final-sigma context aside, lower() of a capital sigma is the plain small sigma
    out = [f"// qtrans_unicode.hpp: the Unicode {version} tables Python's str methods use.",
           "// Written by qtrans.unicode_tables from Python itself; do not edit.",
           "#pragma once", "", "#include <algorithm>", "#include <cstddef>", "#include <string>", "",
           f'#define QTRANS_UNICODE_VERSION "{version}"', "", "namespace qt {", "namespace uni {", "",
           "struct Range { char32_t lo, hi; };",
           "struct Delta { char32_t lo, hi; int delta; unsigned step; };  // lo, lo + step, ... hi map to cp + delta",
           "struct Map { char32_t cp; char32_t to[3]; };", ""]
    for name, rs in props.items():
        out.append(f"inline constexpr Range {name}[] = {{")
        row = []
        for lo, hi in rs:
            row.append(f"{{0x{lo:X}, 0x{hi:X}}}")
            if len(row) == 6:
                out.append("    " + ", ".join(row) + ",")
                row = []
        if row:
            out.append("    " + ", ".join(row) + ",")
        out += ["};", ""]
    for name, rows in maps.items():
        # one character to one: runs where to = cp + delta, every 1st or 2nd character; the rest
        # (ß -> SS, ...) as a list
        runs, multi = [], []
        for cp, to in rows:
            if to[1]:
                multi.append((cp, to))
                continue
            d = to[0] - cp
            last = runs[-1] if runs else None
            if last and last[2] == d and (last[3] in (0, cp - last[1])) and cp - last[1] in (1, 2):
                runs[-1] = (last[0], cp, d, cp - last[1])
            else:
                runs.append((cp, cp, d, 0))
        out.append(f"inline constexpr Delta {name}[] = {{")
        row = []
        for lo, hi, d, step in runs:
            row.append(f"{{0x{lo:X}, 0x{hi:X}, {d}, {step or 1}}}")
            if len(row) == 5:
                out.append("    " + ", ".join(row) + ",")
                row = []
        if row:
            out.append("    " + ", ".join(row) + ",")
        out += ["};", f"inline constexpr Map {name}_full[] = {{"]
        row = []
        for cp, to in multi:
            row.append(f"{{0x{cp:X}, {{0x{to[0]:X}, 0x{to[1]:X}, 0x{to[2]:X}}}}}")
            if len(row) == 4:
                out.append("    " + ", ".join(row) + ",")
                row = []
        if row:
            out.append("    " + ", ".join(row) + ",")
        out += ["};", ""]
    out.append(_HELPERS)
    out += ["}  // namespace uni", "}  // namespace qt", ""]
    return "\n".join(out)


def shipped_version() -> str | None:
    try:
        for line in SHIPPED.read_text(encoding="utf-8").splitlines()[:20]:
            if line.startswith("#define QTRANS_UNICODE_VERSION"):
                return line.split('"')[1]
    except OSError:
        pass
    return None


def header_for_this_python() -> str:
    """The header text matching the running Python (the shipped one when the versions agree)."""
    if shipped_version() == unicodedata.unidata_version:
        return SHIPPED.read_text(encoding="utf-8")
    return generate()


if __name__ == "__main__":
    sys.stdout.write(generate())


def header_path() -> Path:
    """qtrans_unicode.hpp for the running Python: the shipped file, or one written once into
    the user's cache when this Python has another Unicode version."""
    version = unicodedata.unidata_version
    if shipped_version() == version:
        return SHIPPED
    import os
    import tempfile
    base = Path(os.environ.get("XDG_CACHE_HOME") or Path.home() / ".cache") / "qtrans"
    for folder in (base, Path(tempfile.gettempdir()) / "qtrans"):
        path = folder / f"unicode-{version}" / "qtrans_unicode.hpp"
        if path.exists():
            return path
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            tmp = path.with_suffix(f".{os.getpid()}.tmp")
            tmp.write_text(generate(), encoding="utf-8")
            os.replace(tmp, path)
            return path
        except OSError:
            continue
    raise RuntimeError("qtrans: could not write qtrans_unicode.hpp for this Python's Unicode version")
