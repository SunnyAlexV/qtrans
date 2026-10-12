"""Speed hints written in the Python, as comments, that the C++ follows:

    for i in range(n):  # qtrans: unroll 4        ->  QT_UNROLL(4) before the loop
    def payoff(...):    # qtrans: inline          ->  QT_FORCE_INLINE double payoff(...)
                        # qtrans: noinline / hot / cold
    if x > barrier:     # qtrans: unlikely        ->  if (QT_UNLIKELY(x > barrier))

A hint goes on the line it is about, or on a comment line of its own just above it. None of these
can change a number: unrolling repeats the loop body without reordering its floating-point
operations (no -ffast-math), and the others only steer inlining and code layout. Hints that
could change a result -- fast-math, simd / vectorize, parallel, ivdep, restrict -- are refused
with the reason, as is anything unknown (a typo shouldn't be silently ignored). The macros are
defined in the generated header itself, for GCC, Clang and MSVC (MSVC has no unroll pragma: there
QT_UNROLL does nothing).
"""
from __future__ import annotations

import ast
import re

HINT = re.compile(r"^qtrans:\s*(.*)$")

ON_LOOP = {"unroll"}
ON_DEF = {"inline", "noinline", "hot", "cold"}
ON_IF = {"likely", "unlikely"}

REFUSED = {
    "fast-math": "fast-math lets the compiler reorder and approximate floating-point arithmetic, so results change",
    "fastmath": "fast-math lets the compiler reorder and approximate floating-point arithmetic, so results change",
    "simd": "SIMD reductions add in a different order than Python, so sums change in their last digits",
    "vectorize": "a forced vectorised reduction adds in a different order than Python, so sums change",
    "parallel": "a parallel loop adds in a different order than Python (and nondeterministically)",
    "ivdep": "ivdep promises the compiler the iterations don't depend on each other; if that's wrong, results change",
    "restrict": "restrict promises arrays don't overlap; qtrans would have to prove that, and doesn't yet",
    "align": "aligned arrays need qtrans to allocate them differently, which it doesn't yet",
}

MACROS = r"""// speed hints written in the Python (# qtrans: ...): none of them changes a number
#ifndef QT_UNROLL
#if defined(__clang__)
#define QT_PRAGMA_(x) _Pragma(#x)
#define QT_UNROLL(n) QT_PRAGMA_(clang loop unroll_count(n))
#elif defined(__GNUC__)
#define QT_PRAGMA_(x) _Pragma(#x)
#define QT_UNROLL(n) QT_PRAGMA_(GCC unroll n)
#else
#define QT_UNROLL(n)
#endif
#if defined(__GNUC__) || defined(__clang__)
#define QT_FORCE_INLINE inline __attribute__((always_inline))
#define QT_NOINLINE __attribute__((noinline))
#define QT_HOT __attribute__((hot))
#define QT_COLD __attribute__((cold))
#define QT_LIKELY(x) __builtin_expect(!!(x), 1)
#define QT_UNLIKELY(x) __builtin_expect(!!(x), 0)
#elif defined(_MSC_VER)
#define QT_FORCE_INLINE __forceinline
#define QT_NOINLINE __declspec(noinline)
#define QT_HOT
#define QT_COLD
#define QT_LIKELY(x) (x)
#define QT_UNLIKELY(x) (x)
#else
#define QT_FORCE_INLINE inline
#define QT_NOINLINE
#define QT_HOT
#define QT_COLD
#define QT_LIKELY(x) (x)
#define QT_UNLIKELY(x) (x)
#endif
#endif"""


def _parse(text):
    """'unroll 4' -> ("unroll", 4); 'hot' -> ("hot", None); else (word, error)."""
    words = text.strip().split()
    if not words:
        return None, "an empty qtrans hint"
    word = words[0].lower()
    if word in REFUSED:
        return word, f"refused: {REFUSED[word]}"
    if word == "unroll":
        if len(words) != 2 or not words[1].isdigit() or not 2 <= int(words[1]) <= 64:
            return word, "write it as `unroll N`, N from 2 to 64"
        return word, int(words[1])
    if word in ON_DEF | ON_IF:
        if len(words) != 1:
            return word, f"`{word}` takes nothing after it"
        return word, None
    known = "unroll N, inline, noinline, hot, cold, likely, unlikely"
    return word, f"unknown qtrans hint (known: {known})"


def collect(tree, full_comments, inline_comments):
    """(hints, problems): hints maps the line of a loop / if / def to {word: value}; problems is
    a list of 'line N: ...' texts for hints that are refused, unknown, or sit next to nothing."""
    owners = {}
    for n in ast.walk(tree):
        if isinstance(n, (ast.For, ast.While)):
            owners[n.lineno] = ("loop", n)
        elif isinstance(n, ast.If):
            owners.setdefault(n.lineno, ("if", n))
        elif isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)):
            owners[n.lineno] = ("def", n)
    hints, problems = {}, []
    comments = [(ln, t, True) for ln, t in inline_comments.items()] + [(ln, t, False) for ln, t in full_comments.items()]
    for ln, text, inline in sorted(comments):
        m = HINT.match(text.strip())
        if m is None:
            continue
        target = ln
        if not inline:  # on its own line: about the next line that isn't a comment
            target = ln + 1
            while target in full_comments:
                target += 1
        word, value = _parse(m.group(1))
        if isinstance(value, str):
            problems.append(f"line {ln}: `# qtrans: {m.group(1).strip()}`: {value}")
            continue
        owner = owners.get(target)
        if owner is None:
            problems.append(f"line {ln}: `# qtrans: {m.group(1).strip()}` isn't on a loop, an if or a def: not used")
            continue
        kind = owner[0]
        fits = (kind == "loop" and word in ON_LOOP) or (kind == "def" and word in ON_DEF) or (kind == "if" and word in ON_IF)
        if not fits:
            where = {"loop": "a loop", "def": "a def", "if": "an if"}[kind]
            problems.append(f"line {ln}: `# qtrans: {word}` doesn't apply to {where} "
                            f"(unroll: loops; inline / noinline / hot / cold: defs; likely / unlikely: ifs)")
            continue
        h = hints.setdefault(target, {})
        if word in ("inline", "noinline") and ({"inline", "noinline"} - {word}) & set(h):
            problems.append(f"line {ln}: both inline and noinline: not used")
            continue
        h[word] = value
    return hints, problems


def def_head(head: str, h: dict) -> str:
    """`inline double f(...)` with a def's hints."""
    if not h:
        return head
    if "inline" in h:
        head = re.sub(r"^inline\b", "QT_FORCE_INLINE", head)
    pre = [m for w, m in (("noinline", "QT_NOINLINE"), ("hot", "QT_HOT"), ("cold", "QT_COLD")) if w in h]
    return " ".join(pre + [head])
