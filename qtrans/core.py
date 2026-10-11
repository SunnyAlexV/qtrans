"""Shared building blocks for expression translation: the Expr value, C++ precedence,
identifier and literal spelling."""
from __future__ import annotations

import math
from dataclasses import dataclass

# C++ operator precedence levels (lower binds tighter)
P_ATOM, P_UNARY, P_MUL, P_ADD, P_REL, P_EQ, P_AND, P_OR, P_COND = 0, 3, 5, 6, 9, 10, 14, 15, 16

CPP_RESERVED = frozenset("""
alignas alignof and and_eq asm auto bitand bitor bool break case catch char char8_t char16_t
char32_t class compl concept const consteval constexpr constinit const_cast continue co_await
co_return co_yield decltype default delete do double dynamic_cast else enum explicit export
extern false float for friend goto if inline int long mutable namespace new noexcept not not_eq
nullptr operator or or_eq private protected public register reinterpret_cast requires return
short signed sizeof static static_assert static_cast struct switch template this thread_local
throw true try typedef typeid typename union unsigned using virtual void volatile wchar_t while
xor xor_eq std qt NULL main
""".split())


def cpp_ident(name: str) -> str:
    """Python names that are C++ keywords get a trailing underscore (`new` -> `new_`)."""
    return name + "_" if name in CPP_RESERVED else name


@dataclass
class Expr:
    code: str
    prec: int
    type: object
    int_lit: bool = False  # a bare C++ int literal such as `5` (type int, not long long)
    value: object = None   # the Python value, when the expression is a numeric literal
    simple: bool = False   # a name or literal: cheap and safe to write twice (x * x)
    iv: tuple | None = None  # for an int: (lo, hi) with lo <= value <= hi known (None: any int64)
    # Sizes, for NumPy broadcasting. For an array: one token per dimension (None: unknown);
    # two arrays whose tokens are equal provably have that dimension equal, so `a - b` needs no
    # broadcasting. For an int: the token of the size it equals (len(x) -> x's length).
    dims: tuple | None = None
    dimtok: object = None


I64 = (-2 ** 63, 2 ** 63 - 1)
SIZE_IV = (0, 2 ** 45)  # an array's length, shape or count: user address space is at most 2**48 bytes


def iv_of(e: "Expr") -> tuple:
    """The interval an int expression is known to lie in (the whole int64 range if unknown)."""
    if e.iv is not None:
        return e.iv
    if isinstance(e.value, bool) or isinstance(e.value, int):
        return (int(e.value), int(e.value))
    return I64


def iv_fits(iv) -> bool:
    return iv[0] >= I64[0] and iv[1] <= I64[1]


def iv_clip(iv) -> tuple:
    return (max(iv[0], I64[0]), min(iv[1], I64[1]))


def iv_op(sym, a, b) -> tuple:
    if sym == "+":
        return (a[0] + b[0], a[1] + b[1])
    if sym == "-":
        return (a[0] - b[1], a[1] - b[0])
    c = [x * y for x in a for y in b]
    return (min(c), max(c))


def dims_of(e: "Expr") -> tuple:
    """The size tokens of an array expression, one per dimension (None where unknown)."""
    rank = getattr(e.type, "rank", 0)
    if e.dims is not None and len(e.dims) == rank:
        return e.dims
    return (None,) * rank


def same_dim(a, b) -> bool:
    """Two dimensions provably equal: the same known token."""
    return a is not None and a == b


def int_token(e: "Expr"):
    """The size token an int expression stands for: a literal is its value, len(x) x's length."""
    if e.dimtok is not None:
        return e.dimtok
    if isinstance(e.value, int) and not isinstance(e.value, bool):
        return ("lit", e.value)
    return None


def wrap(e: Expr, max_prec: int) -> str:
    return f"({e.code})" if e.prec > max_prec else e.code


def float_literal(v: float) -> tuple[str, int]:
    if math.isnan(v):
        return "qt::nan", P_ATOM
    if math.isinf(v):
        return ("qt::inf", P_ATOM) if v > 0 else ("-qt::inf", P_UNARY)
    r = repr(v)  # shortest string that round-trips: C++ parses it to the same double
    if "e" not in r and "." not in r:
        r += ".0"
    return r, (P_UNARY if r.startswith("-") else P_ATOM)

def cpp_string(s: str) -> str:
    out = []
    for b in s.encode("utf-8"):
        c = chr(b)
        if c == "\\":
            out.append("\\\\")
        elif c == '"':
            out.append('\\"')
        elif c == "\n":
            out.append("\\n")
        elif 32 <= b < 127:
            out.append(c)
        else:
            out.append(f"\\{b:03o}")  # octal escapes: always exactly 3 digits, no ambiguity
    return '"' + "".join(out) + '"'
