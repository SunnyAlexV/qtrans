"""Python's str methods, as qt::py functions on std::string (qtrans_list.hpp).

A Python str is a sequence of characters (code points); a std::string holds UTF-8 bytes. So
everything that counts or indexes (len, s[i], s[a:b], find, center, ...) counts characters, as
Python does, and `for ch in s` goes character by character.

Case, character classes and digits (upper(), isalpha(), float("١٢")) use the Unicode tables of
the Python the translation was made with (qtrans_unicode.hpp, written from that Python), so
they answer as it does for every character; whitespace is its exact list too.

A method whose arguments are all literals is worked out here, by Python itself.
"""
from __future__ import annotations

import ast

from .core import P_ATOM, P_UNARY, Expr
from .errors import Unsupported
from .types import BOOL, INT, STR, ListType

# name -> (result type, the arguments it takes: "s" a str, "i" an int; "?" optional from there on)
_METHODS = {
    "upper": (STR, ""), "lower": (STR, ""), "capitalize": (STR, ""), "title": (STR, ""), "swapcase": (STR, ""),
    "casefold": (STR, ""), "istitle": (BOOL, ""),
    "strip": (STR, "?s"), "lstrip": (STR, "?s"), "rstrip": (STR, "?s"),
    "split": (ListType(STR), "?si"),
    "replace": (STR, "ss?i"),
    "find": (INT, "s?ii"), "index": (INT, "s?ii"), "count": (INT, "s?ii"),
    "startswith": (BOOL, "s"), "endswith": (BOOL, "s"),
    "isdigit": (BOOL, ""), "isdecimal": (BOOL, ""), "isnumeric": (BOOL, ""), "isalpha": (BOOL, ""),
    "isalnum": (BOOL, ""), "isspace": (BOOL, ""), "isupper": (BOOL, ""), "islower": (BOOL, ""),
    "isprintable": (BOOL, ""),
    "zfill": (STR, "i"), "ljust": (STR, "i?s"), "rjust": (STR, "i?s"), "center": (STR, "i?s"),
}
_KEYWORDS = {"split": ("sep", "maxsplit"), "rsplit": ("sep", "maxsplit")}
_HINTS = {
    "rsplit": "rsplit() is not supported; split() is",
    "partition": "not supported; use s.split(sep, 1)",
    "rpartition": "not supported",
    "rfind": "not supported; find() is",
    "rindex": "not supported; index() is",
    "splitlines": "not supported; use s.split('\\n')",
    "format": "use an f-string",
    "encode": "bytes are not supported",
    "expandtabs": "not supported",
    "translate": "not supported", "maketrans": "not supported",
    "removeprefix": "not supported; s[len(p):] if s.startswith(p) else s",
    "removesuffix": "not supported; s[:-len(p)] if p and s.endswith(p) else s",
}


class StrOps:
    """Mixed into ExprTranslator: needs expr, as_type, note."""

    def str_method(self, node, base, name):
        if name == "join":
            return self.str_join(node, base)
        if name not in _METHODS:
            raise Unsupported(node, f"str.{name}() is not supported", _HINTS.get(name))
        ret, spec = _METHODS[name]
        args = list(node.args)
        allowed = _KEYWORDS.get(name, ())
        for k in node.keywords:
            if k.arg not in allowed:
                raise Unsupported(node, f"str.{name}() takes no keyword '{k.arg}'"
                                        + (" (Python raises TypeError)" if not allowed else ""))
        if node.keywords:  # split(sep=..., maxsplit=...): into their places
            given = {k.arg: k.value for k in node.keywords}
            if any(allowed[i] in given for i in range(len(args))):
                raise Unsupported(node, f"str.{name}() got an argument twice (Python raises TypeError)")
            for i in range(len(args), len(allowed)):
                args.append(given.pop(allowed[i], None))
            while args and args[-1] is None:
                args.pop()
            args = [a if a is not None else ast.Constant(None) for a in args]
        need = spec.split("?")[0]
        kinds = spec.replace("?", "")
        if not len(need) <= len(args) <= len(kinds):
            raise Unsupported(node, f"str.{name}() takes {len(need)}"
                              + (f" to {len(kinds)}" if len(kinds) > len(need) else "")
                              + f" argument(s), got {len(args)} (Python raises TypeError)")
        if name in ("startswith", "endswith") and isinstance(args[0], ast.Tuple):
            return self._str_affix(node, base, name, args[0])
        codes, values = [], []
        for a, kind in zip(args, kinds):
            if isinstance(a, ast.Constant) and a.value is None:  # split(None, 1), find(x, None, 3)
                if not ((kind == "s" and name in ("split", "strip", "lstrip", "rstrip"))
                        or (kind == "i" and name in ("find", "index", "count"))):
                    raise Unsupported(a, f"None is not an argument str.{name}() takes here (Python raises TypeError)")
                codes.append("std::nullopt")
                values.append(None)
                continue
            e = self.expr(a)
            want = STR if kind == "s" else INT
            if e.type != want and not (want == INT and e.type == BOOL):
                raise Unsupported(a, f"str.{name}() needs a {'str' if want == STR else 'int'} here, got a {e.type} "
                                     "(Python raises TypeError)")
            codes.append(e.code if want == STR else self.int_arg(e).code)  # a literal binds to const std::string&
            values.append(e.value)
        if isinstance(base.value, str) and all(isinstance(v, (str, int)) or v is None for v in values) \
                and not isinstance(ret, ListType):
            try:  # all literals: Python's own answer
                v = getattr(base.value, name)(*values)
            except Exception as ex:
                raise Unsupported(node, f"Python raises {type(ex).__name__} here: {ex}")
            return self.expr(ast.Constant(v))
        call = f"qt::py::{name}({', '.join([base.code] + codes)})"
        return Expr(call, P_ATOM, ret)

    def _str_affix(self, node, base, name, tup):
        parts = []
        for a in tup.elts:
            e = self.expr(a)
            if e.type != STR:
                raise Unsupported(a, f"str.{name}() of a tuple needs strings (Python raises TypeError)")
            parts.append(e.code)
        return Expr(f"qt::py::{name}({base.code}, {{{', '.join(parts)}}})", P_ATOM, BOOL)

    def str_subscript(self, node, base):
        sl = node.slice
        if isinstance(sl, ast.Slice):
            parts = []
            for b in (sl.lower, sl.upper, sl.step):
                if b is None or (isinstance(b, ast.Constant) and b.value is None):
                    parts.append("qt::py::none")
                else:
                    e = self.expr(b)
                    if e.type not in (INT, BOOL):
                        raise Unsupported(b, f"slice indices must be ints (Python raises TypeError for a {e.type})")
                    if isinstance(e.value, int) and b is sl.step and e.value == 0:
                        raise Unsupported(b, "slice step 0 raises ValueError in Python")
                    parts.append(self.int_arg(e).code)
            if parts[2] == "qt::py::none":
                parts = parts[:2]
            return Expr(f"qt::py::slice({base.code}, {', '.join(parts)})", P_ATOM, STR)
        i = self.expr(sl)
        if i.type not in (INT, BOOL):
            raise Unsupported(sl, f"string indices must be ints (Python raises TypeError for a {i.type})")
        return Expr(f"qt::py::char_at({base.code}, {self.int_arg(i).code})", P_ATOM, STR)

    def str_contains(self, node, x, s, negate):
        if x.type != STR:
            raise Unsupported(node, f"`in <str>` needs a str on the left, got a {x.type} (Python raises TypeError)")
        code = f"qt::py::contains({s.code}, {x.code})"
        return Expr(f"!{code}" if negate else code, P_UNARY if negate else P_ATOM, BOOL)

    def str_repeat(self, node, s, n):
        if isinstance(s.value, str) and isinstance(n.value, int):
            return self.expr(ast.Constant(s.value * n.value))
        return Expr(f"qt::py::repeat({s.code}, {self.as_type(n, INT).code})", P_ATOM, STR)

    @staticmethod
    def str_chars(e: Expr) -> Expr:
        """`for ch in s` / `[c for c in s]`: the characters, one string each."""
        return Expr(f"qt::py::chars({e.code})", P_ATOM, ListType(STR))

