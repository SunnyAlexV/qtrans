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


