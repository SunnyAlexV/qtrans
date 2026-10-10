"""Errors raised when Python code falls outside the supported subset."""
from __future__ import annotations


class Unsupported(Exception):
    """A construct qtrans refuses to translate.

    Refusing is deliberate: for financial code a clear refusal is far better than
    a translation that silently computes a different number.
    """

    def __init__(self, node, message: str, hint: str | None = None):
        self.lineno = getattr(node, "lineno", None)
        self.message = message
        self.hint = hint
        where = f"line {self.lineno}: " if self.lineno else ""
        super().__init__(where + message + (f" (hint: {hint})" if hint else ""))
