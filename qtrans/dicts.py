"""Python dicts as qt::py::dict (qtrans_list.hpp): keys in the order they were added, d[k]
raising KeyError, printing as Python prints a dict.

Keys are str or int, values float, int or str, one type each (Python keeps each value's own
type: an int prints as 1, a float as 1.0). d.keys() and d.values() are lists here (std::vector
copies, in order), so sum(d.values()), sorted(d), max(d.values()) are the list operations. A
loop over a dict reads its keys; `for k, v in d.items()` is rewritten (loops.py) as a loop over
the keys with `v = d[k]`, which reads the same value at the same moment.

Sharing works as for lists (functions.check_lists): `b = a` with either changed afterwards, a
function changing the caller's dict, and a loop changing the size of the dict it goes through
(Python raises RuntimeError) are refused.
"""
from __future__ import annotations

import ast

from .core import P_ATOM, P_UNARY, Expr
from .errors import Unsupported
from .types import BOOL, FLOAT, INT, STR, DictType, ListType

KEYS = (STR, INT)
VALUES = (FLOAT, INT, STR)


def dict_of(t) -> bool:
    return isinstance(t, DictType)


class DictOps:
    # --- {k: v, ...} ------------------------------------------------------------------------------
    def e_Dict(self, node):
        if any(k is None for k in node.keys):
            raise Unsupported(node, "{**other} in a dict literal is not supported", "use d.update(other)")
        if not node.keys:
            return Expr("{}", P_ATOM, DictType(None, None), value={}, simple=True)
        for v in node.values:
            if isinstance(v, ast.Constant) and v.value is None:
                raise Unsupported(v, "a dict holding None among its values is not supported (yet)",
                                  "use a value that means 'none', such as \"\" or -1, or leave that key out")
        keys = [self.expr(k) for k in node.keys]
        vals = [self.expr(v) for v in node.values]
        kt = self._one_type(node, keys, "keys", KEYS)
        vt = self._one_type(node, vals, "values", VALUES)
        t = DictType(kt, vt)
        pairs = [f"{{{self._key(k, kt, n).code}, {self._val(v, vt).code}}}"
                 for k, v, n in zip(keys, vals, node.keys)]
        return Expr(f"{t.cpp()}{{{', '.join(pairs)}}}", P_ATOM, t)

    def _one_type(self, node, items, what, allowed):
        types = {e.type for e in items}
        if len(types) != 1:
            names = ", ".join(sorted(str(t) for t in types))
            raise Unsupported(node, f"a dict whose {what} have several types ({names}): Python keeps each one's "
                                    "type (an int prints as 1, a float as 1.0), the C++ dict has one",
                              "make them one type: 1.0 rather than 1")
        t = types.pop()
        if t not in allowed:
            kinds = "str or int" if what == "keys" else "float, int or str"
            raise Unsupported(node, f"dict {what} of type {t} are not supported", f"dicts with {kinds} {what} are")
        return t

    def _key(self, e, kt, node):
        if e.type == kt:
            return self.as_type(e, INT) if kt == INT else e
        if kt == INT and e.type == BOOL:
            raise Unsupported(node, "a bool used as a dict key: Python treats True as the key 1",
                              "use int(...) to make that explicit")
        raise Unsupported(node, f"a {e.type} key in a dict with {kt} keys"
                          + (": Python finds the int key 1 by the float 1.0" if e.type == FLOAT and kt == INT else ""))

    def _val(self, e, vt):
        return self.as_type(e, INT) if vt == INT else self.as_double(e) if vt == FLOAT else e

    def _need_dict(self, e, node, what):
        if e.type.key is None:
            raise Unsupported(node, f"the key and value types of `{ast.unparse(node)}` are not known (an empty "
                                    "dict that nothing is stored in)", "annotate it: d: dict[str, float] = {}")

    def e_DictComp(self, node):
        return self.e_ListComp(node, mode="dict")

    # --- d[k] ---------------------------------------------------------------------------------------
    def dict_subscript(self, node, base):
        if base.type.key is None and getattr(self, "inferring", False):
            from .expressions import _Unknown
            raise _Unknown(getattr(node.value, "id", "dict"))
        self._need_dict(base, node.value, "indexing")
        if isinstance(node.slice, ast.Slice):
            raise Unsupported(node, "a dict can't be sliced (Python raises TypeError)")
        k = self._key(self.expr(node.slice), base.type.key, node.slice)
        return Expr(f"{base.code}.at({k.code})", P_ATOM, base.type.val)

    # --- methods ----------------------------------------------------------------------------------
    def dict_method(self, node, base, name):
        t = base.type
        if node.keywords:
            raise Unsupported(node, f"keyword arguments to dict.{name}() are not supported")
        args = node.args
        if t.key is None and getattr(self, "inferring", False):
            # `counts = {}`, then counts[w] = counts.get(w, 0) + 1: while the dict's types are
            # being worked out, get / setdefault / pop with a default have the default's type
            if name in ("get", "setdefault", "pop") and len(args) == 2:
                return Expr("<inferring>", P_ATOM, self.expr(args[1]).type)
            from .expressions import _Unknown
            raise _Unknown(getattr(node.func.value, "id", name))
        if name not in ("clear", "copy", "update") or t.key is not None:
            self._need_dict(base, node.func.value, name)

        def key(a):
            return self._key(self.expr(a), t.key, a)

        def value(a):
            e = self.expr(a)
            if e.type != t.val and not (t.val == FLOAT and e.type in (INT, BOOL)):
                raise Unsupported(a, f"a {e.type} as the value of a dict of {t.val}: Python keeps each value's "
                                     "type (an int prints as 1, a float as 1.0)")
            if t.val == FLOAT and e.type in (INT, BOOL):
                raise Unsupported(a, "this value is a Python int, in a dict of floats (it prints as 1, not 1.0)",
                                  "write it as a float: 0.0 rather than 0")
            return self._val(e, t.val)

        def arity(lo, hi):
            if not lo <= len(args) <= hi:
                raise Unsupported(node, f"dict.{name}() takes {lo if lo == hi else f'{lo} to {hi}'} arguments")
        if name == "get":
            arity(1, 2)
            if len(args) == 1:
                raise Unsupported(node, "d.get(k) returns None when k is missing", "give the default: d.get(k, 0.0)")
            return Expr(f"{base.code}.get({key(args[0]).code}, {value(args[1]).code})", P_ATOM, t.val)
        if name == "pop":
            arity(1, 2)
            extra = f", {value(args[1]).code}" if len(args) == 2 else ""
            return Expr(f"{base.code}.pop({key(args[0]).code}{extra})", P_ATOM, t.val)
        if name == "setdefault":
            arity(2, 2)
            return Expr(f"{base.code}.setdefault({key(args[0]).code}, {value(args[1]).code})", P_ATOM, t.val)
        if name in ("keys", "values"):
            arity(0, 0)
            return Expr(f"{base.code}.{name}()", P_ATOM, ListType(t.key if name == "keys" else t.val))
        if name == "items":
            raise Unsupported(node, "d.items() is supported in loops and comprehensions: for k, v in d.items()")
        if name == "clear":
            arity(0, 0)
            return Expr(f"{base.code}.clear()", P_ATOM, None)
        if name == "copy":
            arity(0, 0)
            return Expr(f"{t.cpp()}({base.code})", P_ATOM, t)
        if name == "update":
            arity(1, 1)
            other = self.expr(args[0])
            if not dict_of(other.type) or (other.type.key is not None and other.type != t and t.key is not None):
                raise Unsupported(node, f"d.update() with a {other.type}: a dict of the same types is supported")
            return Expr(f"{base.code}.update({other.code})", P_ATOM, None)
        raise Unsupported(node, f"dict.{name}() is not supported")

    # --- len(d), list(d), sorted(d), dict(d) ------------------------------------------------------
    def dict_builtin(self, node, name):
        """None if the argument isn't a dict. Python goes through a dict's keys."""
        if len(node.args) != 1 or node.keywords:
            return None
        arg = node.args[0]
        if isinstance(arg, (ast.ListComp, ast.List, ast.GeneratorExp)) or (
                isinstance(arg, ast.Call) and isinstance(arg.func, ast.Name) and arg.func.id == "range"):
            return None  # list(range(n)), sum([...]): not a dict
        a = self.expr(arg)
        if not dict_of(a.type):
            return None
        self._need_dict(a, node.args[0], f"{name}()")
        if name == "len":
            return Expr(f"{a.code}.size()", P_ATOM, INT)
        if name == "dict":
            return Expr(f"{a.type.cpp()}({a.code})", P_ATOM, a.type)
        # sum / min / max / sorted / list of a dict: of its keys
        keys = ast.copy_location(ast.Call(func=ast.Attribute(value=node.args[0], attr="keys", ctx=ast.Load()),
                                          args=[], keywords=[]), node.args[0])
        call = ast.copy_location(ast.Call(func=node.func, args=[keys], keywords=[]), node)
        return self.expr(call)

    # --- k in d -----------------------------------------------------------------------------------
    def dict_contains(self, node, k_node, d, negate):
        self._need_dict(d, node.comparators[0], "`in`")
        k = self._key(self.expr(k_node), d.type.key, k_node)
        code = f"{d.code}.contains({k.code})"
        return Expr(f"!{code}" if negate else code, P_UNARY if negate else P_ATOM, BOOL)
