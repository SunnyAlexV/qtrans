"""Text: print(), str(), repr(), format(), f-strings and "...".format(...).

What the C++ prints is what Python prints, character for character: floats as Python's
shortest repr (0.1, 1e+16, 2.5), format specs by CPython's rules, arrays as NumPy lays them
out, and NumPy scalars as NumPy 2 shows their repr (np.float64(2.5)). The runtime is
qtrans_print.hpp (namespace qt::py); tests/test_print.py checks it against Python and NumPy.

The C++ is written the way someone would write it by hand:

    print("total:", n, x)            std::cout << "total: " << n << ' ' << qt::py::str(x) << '\\n';
    print(f"{x:.3f} ({k} of {n})")   std::cout << qt::py::format(x, ".3f") << " (" << k << " of " << n << ")\\n";
    s = f"run {k}"                   std::string s = "run " + std::to_string(k);

A text is a list of pieces, each a literal or a value; print() streams them, a string value
joins them with +.
"""
from __future__ import annotations

import ast
import re
import string
from dataclasses import dataclass

from .core import P_ADD, P_ATOM, Expr, cpp_string, wrap
from .structs import THIS
from .errors import Unsupported
from .origins import NP, PY
from .types import (ArrayType, DictType, ExcType, BOOL, FLOAT, INT, NONE, RNG, STR, VOID, ListType, MaskType, OptionalType, Scalar, StructType,
                    TupleType, is_array)

_SAMPLE = {"float": 1.5, "int": 7, "bool": True}  # a value of each type, to try a format spec on
_PERCENT = re.compile(r"%(?:\((\w*)\))?([-+ 0#]*)(\*|\d+)?(?:\.(\*|\d*))?[hlL]?([diouxXeEfFgGcrsa%])")
_CONV = {-1: None, 115: "s", 114: "r", 97: "a"}


@dataclass
class Piece:
    lit: str | None = None  # literal text, or
    code: str = ""          # C++ that makes the text: a std::string (std=True) or a const char*
    stream: str = ""        # C++ to write with << (an int writes itself; no std::to_string)
    std: bool = True
    raises: bool | None = None  # can working it out raise? (None: judge from its C++)


def _lit(text):
    return Piece(lit=text)


def _merge(pieces):
    out = []
    for p in pieces:
        if p.lit is not None and out and out[-1].lit is not None:
            out[-1] = _lit(out[-1].lit + p.lit)
        elif p.lit != "":
            out.append(p)
    return out


_SAFE_CALL = re.compile(r"qt::py::(str|format|repr|np|round_numpy)\(|static_cast<[^>]*>\(|std::\w+\(|\.\w+\(")


def _may_throw(code: str) -> bool:
    """Could working out this piece raise? A qt:: function or a translated function might; the
    printing helpers (with a spec checked at translation), a conversion, the C library's maths
    (std::exp, ...) and member functions (v.size()) don't."""
    code = re.sub(r'"(?:[^"\\]|\\.)*"', '""', code)
    return bool(re.search(r"[\w:]\(", _SAFE_CALL.sub(" (", code)))


def _nonempty(node) -> bool:
    """A list expression that surely has an element: [a, ...], or one joined (+) with it."""
    if isinstance(node, ast.List):
        return bool(node.elts)
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Add):
        return _nonempty(node.left) or _nonempty(node.right)
    if isinstance(node, ast.IfExp):
        return _nonempty(node.body) and _nonempty(node.orelse)
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Mult):  # [x] * 3
        for a, b in ((node.left, node.right), (node.right, node.left)):
            if _nonempty(a) and isinstance(b, ast.Constant) and isinstance(b.value, int) and b.value > 0:
                return True
    return False


def _stream_lit(text):
    return {"\n": r"'\n'", " ": "' '"}.get(text, cpp_string(text))


class TextOps:
    # --- where a value came from (see origins.py) ------------------------------------------
    def origin(self, node):
        fdef = getattr(self, "fdef", None)
        if fdef is None:
            return PY if isinstance(node, ast.Constant) else None
        return self.mod.origins.function(fdef, getattr(self, "sig", None)).of(node)

    # --- a value as text ----------------------------------------------------------------
    def struct_text(self, node, e, kind) -> Piece:
        """str(obj) / repr(obj): the class's own __str__ / __repr__ (translated as methods), or the
        repr a dataclass or NamedTuple gets (Point(x=1.0, y=2.0)), generated where it is used."""
        from .classes import auto_repr
        info = self.mod.classes.get(e.type.name)
        if info is None:
            raise Unsupported(node, f"showing a {e.type} is not supported")
        dot = "" if e.code == THIS else "->" if e.type.ref else "."
        obj = "" if e.code == THIS else wrap(e, P_ATOM)
        if info.ref:  # the object may be of any subclass: each must show the way Python shows it
            return self._hierarchy_text(node, info, kind, obj + dot)
        for method in (["__str__", "__repr__"] if kind == "str" else ["__repr__"]):
            if method in info.methods:
                key = info.methods[method].key
                if key in self.mod.rejected:
                    raise Unsupported(node, f"{info.name}.{method} could not be translated: "
                                            f"{self.mod.rejected[key].message}")
                c = f"{obj}{dot}{'str' if method == '__str__' else 'repr'}()"
                return Piece(code=c, stream=c, raises=True)
        if not auto_repr(info):
            raise Unsupported(node, f"Python shows a {info.name} (a class with no __repr__) by its memory address, "
                                    "which differs on every run", f"give {info.name} a __repr__, or print its fields")
        self._need_repr(node, info)
        c = f"{obj}{dot}repr()"
        return Piece(code=c, stream=c)

    def _hierarchy_text(self, node, info, kind, prefix) -> Piece:
        """str(p) / repr(p) for p of a class hierarchy: a virtual repr() (and str()), so the object's
        own class decides, as in Python. Every class p could be must show the same way."""
        from .classes import auto_repr
        classes = [info] + info.descendants()
        classes = [c for c in classes if not (c.abc and c.abstract_left())] or [info]
        uses_str = kind == "str" and any(c.find_method("__str__") is not None for c in classes)
        if uses_str and not all(c.find_method("__str__") is not None for c in classes):
            miss = next(c for c in classes if c.find_method("__str__") is None)
            raise Unsupported(node, f"str() of a {info.name}: {miss.name} has no __str__ (Python then uses its "
                                    "__repr__) while others do", f"give {info.name} a __str__")
        method = "__str__" if uses_str else "__repr__"
        raises = False
        for c in classes:
            m = c.find_method(method)
            if m is not None:
                if m.key in self.mod.rejected:
                    raise Unsupported(node, f"{c.method_owner(method).name}.{method} could not be translated: "
                                            f"{self.mod.rejected[m.key].message}")
                own = c.method_owner(method)
                if method == "__repr__" and own is not c and auto_repr(c):
                    self._need_repr(node, c)  # a dataclass writes its own repr, over its base's __repr__
                else:
                    raises = True
                continue
            if not auto_repr(c):
                raise Unsupported(node, f"Python shows a {c.name} (a class with no __repr__) by its memory address, "
                                        "which differs on every run", f"give {c.name} a __repr__")
            self._need_repr(node, c)
        if info.find_method(method) is None and info.name not in getattr(self.mod, "repr_needed", set()):
            # (an abstract class: Python never shows one, but the call needs a repr() to go through)
            self.mod.__dict__.setdefault("text_pure", set()).add((info.name, method))
        c = f"{prefix}{'str' if method == '__str__' else 'repr'}()"
        return Piece(code=c, stream=c, raises=raises)

    def _need_repr(self, node, info, seen=None):
        """The dataclass repr of info (and of the dataclasses inside it) is generated, if it can be exact."""
        from .classes import auto_repr
        seen = seen if seen is not None else set()
        if info.name in seen:
            return
        seen.add(info.name)
        if "__repr__" not in info.methods:
            problem = self.mod.struct_repr_problem(info)
            if problem:
                raise Unsupported(node, problem[1], "convert it with float(...) where it goes in")
            self.mod.__dict__.setdefault("repr_needed", set()).add(info.name)
        for f in info.fields:
            if f.repr and isinstance(f.type, StructType) and f.type.ref:
                self._hierarchy_text(node, self.mod.classes[f.type.name], "repr", "")  # any subclass may be there
                continue
            if f.repr and isinstance(f.type, StructType):
                inner = self.mod.classes[f.type.name]
                if "__repr__" not in inner.methods and not auto_repr(inner):
                    raise Unsupported(node, f"field '{f.name}' holds a {inner.name}, which Python shows by its memory "
                                            "address", f"give {inner.name} a __repr__")
                self._need_repr(node, inner, seen)

    def _refuse_text(self, node, e, what):
        src = ast.unparse(node)
        if isinstance(e.type, StructType):
            raise Unsupported(node, f"{what} of a class instance (`{src}`) is not translated yet",
                              "print its fields instead")
        if e.type == RNG:
            raise Unsupported(node, f"{what} of a random generator shows its memory address in Python")
        if isinstance(e.type, MaskType):
            raise Unsupported(node, f"{what} of a boolean array is not translated yet",
                              "print the array itself, or .sum() / .any()")
        raise Unsupported(node, f"{what} of a {e.type} value is not supported")

    def _flag_cond(self, node, e):
        """The flag condition for a float value that may be a Python int, if it has one."""
        if not isinstance(node, ast.AST):
            return None
        cond = self.int_cond(node)
        if not isinstance(cond, str) or cond == "true":
            return None
        return cond

    def _int_or_float(self, node, e, what):
        """A float in C++ that may be a Python int: a local that starts as `total = 0` and later
        gets floats, or a function that returns 0 on one path and a float on another. Python
        shows the int as 0, the C++ double shows 0.0: shown as text, that is refused."""
        if not isinstance(node, ast.AST):
            return
        nodes = []
        if e.type == FLOAT:
            nodes = [node]
        elif isinstance(e.type, TupleType) and isinstance(node, ast.Tuple):
            nodes = [n for n, t in zip(node.elts, e.type.elts) if t == FLOAT]
        elif isinstance(e.type, TupleType) and isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            maybe = getattr(self.mod, "maybe_int", set())
            hit = next((k for k, t in enumerate(e.type.elts) if t == FLOAT and (node.func.id, k) in maybe), None)
            if hit is not None:
                self._refuse_int(node, what, f"{node.func.id}() returns an int as element {hit} on some path")
        for n in nodes:
            why = self.maybe_int(n)
            if why and not isinstance(self.int_cond(n), str):
                self._refuse_int(node, what, why)

    def _refuse_int(self, node, what, why):
        start = f"start it as a float ({node.id} = 0.0 rather than 0)" if isinstance(node, ast.Name) else \
            "make it a float where it starts (0.0 rather than 0, return 0.0)"
        raise Unsupported(node, f"{what} of `{ast.unparse(node)}`: it can be a Python int here ({why}), which "
                                "Python shows as 0 where the C++ double shows 0.0",
                          f"{start}, or format it with a float spec (:.2f, :g)")

    def int_cond(self, node):
        """Whether Python holds an int in this value, as C++: "true", None (a float, surely), a
        condition on the flags of int-or-float locals ("total_is_int"), or False when it depends
        on the run in a way that can't be written down (a function that returns 0 on one path)."""
        flagged = getattr(self, "flagged", set())
        t = self._quiet_type(node)
        if t in (INT, BOOL):
            return "true"
        if t != FLOAT:
            return None
        if isinstance(node, ast.Name):
            if node.id in flagged:
                return self.flag_name(node.id)
            return False if self.maybe_int(node) else None
        if isinstance(node, ast.BinOp):
            if isinstance(node.op, ast.Div):
                return None
            a, b = self.int_cond(node.left), self.int_cond(node.right)
            if a is False or b is False:
                return False
            if a is None or b is None:
                return None
            if a == "true":
                return b
            if b == "true":
                return a
            return f"{a} && {b}"
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.USub, ast.UAdd)):
            return self.int_cond(node.operand)
        if self._empty_sum(node):  # sum([]) is the int 0
            return f"{self.ident(node.args[0].id)}.empty()"
        empty = self._empty_comprehension_sum(node)
        if empty:
            return empty
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "sum" \
                and len(node.args) == 1 and isinstance(node.args[0], ast.Call) and not node.args[0].args \
                and isinstance(node.args[0].func, ast.Attribute) and node.args[0].func.attr == "values" \
                and isinstance(node.args[0].func.value, ast.Name) and not self.is_variable("sum") \
                and isinstance(self._quiet_type(node.args[0].func.value), DictType) and t == FLOAT:
            return f"{self.ident(node.args[0].func.value.id)}.empty()"  # sum(d.values()) of no values: the int 0
        return False if self.maybe_int(node) else None

    def _empty_comprehension_sum(self, node):
        """sum(x * w for x, w in zip(xs, ws)), sum(x * x for x in xs): the int 0 exactly when
        the sequences it goes through are empty (no `if` in it): that condition as C++, or None."""
        if not (isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "sum"
                and len(node.args) == 1 and isinstance(node.args[0], ast.ListComp) and not self.is_variable("sum")):
            return None
        comp = node.args[0]
        if len(comp.generators) != 1 or comp.generators[0].ifs or self._quiet_type(node) != FLOAT:
            return None
        it = comp.generators[0].iter
        if isinstance(it, ast.Call) and isinstance(it.func, ast.Attribute) and it.func.attr in ("keys", "values") \
                and not it.args and isinstance(it.func.value, ast.Name):
            it = it.func.value  # sum(v for v in d.values()): empty exactly when d is
        if isinstance(it, ast.Name):
            names = [it]
        elif isinstance(it, ast.Call) and isinstance(it.func, ast.Name) and it.func.id == "range" \
                and len(it.args) == 1 and not it.keywords:
            stop = it.args[0]
            lens = stop.args if isinstance(stop, ast.Call) and isinstance(stop.func, ast.Name) \
                and stop.func.id == "min" else [stop]
            names = []
            for n in lens:  # len(xs), or len(xs) - 1 (zip(xs, xs[1:]))
                cut = 0
                if isinstance(n, ast.BinOp) and isinstance(n.op, ast.Sub) and isinstance(n.right, ast.Constant) \
                        and type(n.right.value) is int and n.right.value > 0:
                    n, cut = n.left, n.right.value
                if not (isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id == "len"
                        and len(n.args) == 1 and isinstance(n.args[0], ast.Name)):
                    return None
                names.append((n.args[0], cut))
        else:
            return None
        if isinstance(it, ast.Name):
            names = [(it, 0)]
        conds = []
        for n, cut in names:
            t = self._quiet_type(n)
            if not (isinstance(t, (ListType, DictType)) or t == ArrayType(1)):
                return None
            x = self.ident(n.id)
            conds.append(f"{x}.size() <= {cut}" if cut else f"{x}.empty()" if isinstance(t, (ListType, DictType))
                         else f"{x}.size() == 0")
        return " || ".join(dict.fromkeys(conds))

    def _empty_sum(self, node) -> bool:
        """sum(xs) of a list of floats, xs a variable: the int 0 when xs is empty, a float otherwise."""
        if not (isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "sum"
                and len(node.args) == 1 and isinstance(node.args[0], ast.Name) and not self.is_variable("sum")
                and not _nonempty(node.args[0])):
            return False
        t = self._quiet_type(node.args[0])
        return isinstance(t, ListType) and t.elem == FLOAT

    def maybe_int(self, node):
        """Why a value the C++ holds as a double may be a Python int, or None if it can't be."""
        widened = getattr(self, "widened", set())
        maybe = getattr(self.mod, "maybe_int", set())
        if isinstance(node, ast.Name):
            if node.id in widened and node.id not in getattr(self, "flagged", ()) and not self._surely_float(node.id):
                return f"'{node.id}' is assigned an int, or a value that can be one"
            return None
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and not self.is_variable(node.func.id):
            f = node.func.id
            if (f, None) in maybe:
                return f"{f}() returns an int on some path"
            if f == "sum" and len(node.args) == 2:  # from a start: an int only if the start is one
                return self.maybe_int(node.args[1])
            if f == "sum" and node.args and not _nonempty(node.args[0]):
                t = self._quiet_type(node.args[0])
                if isinstance(t, ListType) and t.elem == FLOAT:
                    return "sum() of an empty list is the int 0"
            if f == "abs" and node.args:
                return self.maybe_int(node.args[0])
            if f in ("min", "max"):
                return next((w for w in (self.maybe_int(a) or self._int_typed(a) for a in node.args) if w), None)
            return None
        if isinstance(node, ast.BinOp) and not isinstance(node.op, ast.Div):
            left, right = self.maybe_int(node.left), self.maybe_int(node.right)
            if (left or right) and (left or self._int_typed(node.left)) and (right or self._int_typed(node.right)):
                return left or right  # int op int stays an int in Python
            return None
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.USub, ast.UAdd)):
            return self.maybe_int(node.operand)
        if isinstance(node, ast.IfExp):
            return self.maybe_int(node.body) or self.maybe_int(node.orelse) or \
                self._int_typed(node.body) or self._int_typed(node.orelse)
        return None

    def _int_typed(self, node):
        """'an int' if the value is an int (or bool) in C++ too, else None."""
        return "an int" if self._quiet_type(node) in (INT, BOOL) else None

    def _quiet_type(self, value):
        if not hasattr(self, "expr_type_quiet"):
            return None
        saved = (list(self.notes), set(self._noted), self.inferring, self.current)
        self.inferring = True  # only its type: no notes for the statement being translated
        try:
            return self.expr_type_quiet(value)
        except Exception:  # noqa: BLE001 - a type not known
            return None
        finally:
            self.notes, self._noted, self.inferring, self.current = saved

    # --- does a variable surely hold a float here? -----------------------------------------
    def _py_float_value(self, value, var) -> bool:
        """Is this value a Python float whatever the run (2.5, x * 1.5 with x a float)?"""
        return self._quiet_type(value) == FLOAT and not self.maybe_int(value)

    def _sets_float(self, st, var) -> bool:
        """After st has run, var surely holds a float."""
        if isinstance(st, (ast.Assign, ast.AnnAssign)) and st.value is not None:
            targets = st.targets if isinstance(st, ast.Assign) else [st.target]
            if any(isinstance(t, ast.Name) and t.id == var for t in targets):
                return self._py_float_value(st.value, var)
        if isinstance(st, ast.AugAssign) and isinstance(st.target, ast.Name) and st.target.id == var:
            if isinstance(st.op, ast.Div):
                return True  # x /= y: a float, always
            return self._py_float_value(st.value, var)
        if isinstance(st, ast.For) and self._runs_once(st):
            return self._block_sets_float(st.body, var)
        if isinstance(st, ast.If):
            return self._block_sets_float(st.body, var) and self._block_sets_float(st.orelse, var)
        return False

    def _block_sets_float(self, body, var) -> bool:
        for k in range(len(body) - 1, -1, -1):  # the last statement that decides it
            if self._sets_float(body[k], var):
                return True
            if self._may_set_int(body[k], var):
                return False
        return False

    def _may_set_int(self, st, var) -> bool:
        for n in ast.walk(st):
            if isinstance(n, (ast.Assign, ast.AnnAssign)) and n.value is not None:
                targets = n.targets if isinstance(n, ast.Assign) else [n.target]
                if any(isinstance(t, ast.Name) and t.id == var for t in targets) and not self._py_float_value(n.value, var):
                    return True
            if isinstance(n, ast.AugAssign) and isinstance(n.target, ast.Name) and n.target.id == var \
                    and not self._py_float_value(n.value, var):
                return True
        return False

    @staticmethod
    def _runs_once(loop) -> bool:
        """A for-loop whose body surely runs: over a literal list, or range() of constants."""
        it = loop.iter
        if isinstance(it, (ast.List, ast.Tuple)):
            return bool(it.elts)
        if isinstance(it, ast.Call) and isinstance(it.func, ast.Name) and it.func.id == "range" \
                and all(isinstance(a, ast.Constant) and isinstance(a.value, int) for a in it.args) and it.args:
            try:
                return len(range(*[a.value for a in it.args])) > 0
            except (TypeError, ValueError):
                return False
        return False

    def _surely_float(self, var) -> bool:
        """At the statement being translated, var surely holds a float (it was assigned a float
        on the way here, with no int assigned since)."""
        fdef, here = getattr(self, "fdef", None), getattr(self, "current", None)
        if fdef is None or here is None:
            return False

        def path(body, target):  # [(block, index)] from the function's body down to target
            for k, st in enumerate(body):
                if st is target:
                    return [(body, k)]
                for field in ("body", "orelse", "finalbody"):
                    sub = getattr(st, field, None)
                    if isinstance(sub, list):
                        p = path(sub, target)
                        if p is not None:
                            return [(body, k)] + p
                for h in getattr(st, "handlers", []):
                    p = path(h.body, target)
                    if p is not None:
                        return [(body, k)] + p
            return None
        levels = path(fdef.body, here)
        if levels is None:
            return False
        for depth in range(len(levels) - 1, -1, -1):
            block, k = levels[depth]
            if depth < len(levels) - 1 and isinstance(block[k], (ast.For, ast.While)) and self._may_set_int(block[k], var):
                return False  # an int assigned later in the loop reaches this statement next time round
            for j in range(k - 1, -1, -1):
                if self._sets_float(block[j], var):
                    return True
                if self._may_set_int(block[j], var):
                    return False
        return False

    def str_piece(self, node, e=None) -> Piece:
        """str(x), as print() shows x."""
        e = self.expr_raw(node) if e is None else e
        self._int_or_float(node, e, "showing the value")
        t = e.type
        if t == STR:
            if isinstance(e.value, str):
                return _lit(e.value)
            return Piece(code=e.code, stream=wrap(e, P_ADD))
        if t == NONE:
            return _lit("None")
        if isinstance(t, ExcType):  # str(e) of a caught exception: its message
            return Piece(code=f"std::string({e.code}.what())", stream=f"{e.code}.what()")
        if t == VOID:  # print(f()) of a function returning None: f runs, then None is shown
            return Piece(code=f"({e.code}, \"None\")", stream=f"({e.code}, \"None\")", std=False)
        if isinstance(e.value, bool) or t == BOOL and e.value is not None:
            return _lit("True" if e.value else "False")
        if t == INT:
            if isinstance(e.value, int):
                return _lit(str(e.value))
            return Piece(code=f"std::to_string({e.code})", stream=wrap(e, P_ADD))
        if t == FLOAT:
            if isinstance(e.value, float):
                return _lit(repr(e.value))
            cond = self._flag_cond(node, e)
            if cond:  # an int or a float in Python, as the run went: written as the one it is
                c = f"qt::py::str({e.code}, {cond})"
                return Piece(code=c, stream=c)
            c = f"qt::py::str({e.code})"
            return Piece(code=c, stream=c)
        if t == BOOL:
            c = f"({e.code} ? \"True\" : \"False\")"
            return Piece(code=c, stream=c, std=False)
        if isinstance(t, OptionalType) and isinstance(t.inner, Scalar) or is_array(t):
            c = f"qt::py::str({e.code})"
            return Piece(code=c, stream=c)
        if isinstance(t, (TupleType, ListType, DictType)):
            return self.repr_piece(node, e)
        if isinstance(t, StructType):
            return self.struct_text(node, e, "str")
        self._refuse_text(node, e, "printing")

    def repr_piece(self, node, e=None) -> Piece:
        """repr(x): a string in quotes, a NumPy scalar as np.float64(...)."""
        e = self.expr_raw(node) if e is None else e
        self._int_or_float(node, e, "repr")
        t = e.type
        if t == STR:
            if isinstance(e.value, str):
                return _lit(repr(e.value))
            c = f"qt::py::repr({e.code})"
            return Piece(code=c, stream=c)
        if t == NONE:
            return _lit("None")
        if isinstance(t, Scalar):
            if e.value is not None and not isinstance(e.value, str):
                return _lit(repr(e.value))
            o = self._scalar_origin(node, t, "repr")
            if o == PY:
                return self.str_piece(node, e)
            c = f"qt::py::repr(qt::py::np({self.as_type(e, t).code}))"
            return Piece(code=c, stream=c)
        if isinstance(t, OptionalType) and isinstance(t.inner, Scalar):
            self._scalar_origin(node, t.inner, "repr", allow_np=False)
            c = f"qt::py::repr({e.code})"
            return Piece(code=c, stream=c)
        if is_array(t):
            c = f"qt::py::repr({e.code})"
            return Piece(code=c, stream=c)
        if isinstance(t, StructType):
            return self.struct_text(node, e, "repr")
        if isinstance(t, ListType):
            if e.value == []:
                return _lit("[]")
            numpy = False
            if t.elem in (FLOAT, INT):
                o = self.origin(node)
                eo = o[1] if isinstance(o, tuple) and o[0] == "list" else None
                if eo not in (PY, NP):
                    raise Unsupported(node, self._origin_message(node, "the list's elements"),
                                      "convert them with float(...) / int(...) as they go in")
                numpy = eo == NP
                if t.elem == FLOAT and self.maybe_int(node):
                    raise Unsupported(node, "an element of the list may be a Python int", "0.0 rather than 0")
            c = f"qt::py::repr({e.code}{', true' if numpy else ''})"
            return Piece(code=c, stream=c)
        if isinstance(t, DictType):
            if e.value == {}:
                return _lit("{}")
            self._need_dict(e, node, "printing")
            numpy = False
            if t.val in (FLOAT, INT):
                o = self.origin(node)
                vo = o[1] if isinstance(o, tuple) and o[0] == "dict" else None
                if vo not in (PY, NP):
                    raise Unsupported(node, self._origin_message(node, "the dict's values"),
                                      "convert them with float(...) / int(...) as they go in")
                numpy = vo == NP
            c = f"qt::py::repr({e.code}{', true' if numpy else ''})"
            return Piece(code=c, stream=c)
        if isinstance(t, TupleType):
            o = self.origin(node)
            parts = o[1] if isinstance(o, tuple) and len(o[1]) == len(t.elts) else [None] * len(t.elts)
            marks = ""
            for k, (et, eo) in enumerate(zip(t.elts, parts)):
                if isinstance(et, TupleType) or isinstance(et, (StructType, MaskType, OptionalType)) or et == RNG:
                    raise Unsupported(node, f"showing a tuple holding a {et} is not translated yet")
                if isinstance(et, Scalar) and eo not in (PY, NP):
                    raise Unsupported(node, self._origin_message(node, f"element {k} of the tuple"),
                                      "convert it with float(...) / int(...) where the tuple is made")
                marks += "n" if eo == NP and isinstance(et, Scalar) else "p"
            c = f"qt::py::repr({e.code}, \"{marks}\")" if "n" in marks else f"qt::py::repr({e.code})"
            return Piece(code=c, stream=c)
        self._refuse_text(node, e, "repr")

    def _origin_message(self, node, what):
        return (f"qtrans can't tell whether {what} (`{ast.unparse(node)}`) is a Python number or a NumPy "
                "scalar here (it depends on the run), and repr() shows them differently: 2.5 or np.float64(2.5)")

    def _scalar_origin(self, node, t, what, allow_np=True):
        o = self.origin(node)
        if o == PY or (o == NP and allow_np):
            return o
        raise Unsupported(node, self._origin_message(node, "the value"),
                          f"write {'float' if t == FLOAT else 'int' if t == INT else 'bool'}(...) around it to make "
                          "it a Python number, or show it with str() / print(), which don't differ")

    def format_piece(self, node, e, spec, spec_node=None) -> Piece:
        """format(x, spec), f"{x:spec}": spec a literal str, or a Piece list (a spec made at run time)."""
        if spec == "" or (spec in ("d", "n") and e.type == INT and spec == "d") or (spec == "s" and e.type == STR):
            return self.str_piece(node, e)  # format(n, "d") is str(n)
        if not (isinstance(spec, str) and spec[-1:] in tuple("eEfFgG%")):
            self._int_or_float(node, e, "formatting")  # a float type writes an int as it writes a float
        t = e.type
        inner = t.inner if isinstance(t, OptionalType) and isinstance(t.inner, Scalar) else t
        if not (isinstance(inner, Scalar) or inner == STR):
            raise Unsupported(spec_node or node, f"a format spec for a {t} value: Python raises TypeError "
                                                 "(unsupported format string)", "format its elements one by one")
        if isinstance(spec, str):
            sample = "a" if inner == STR else _SAMPLE[inner.name]
            try:
                py = format(sample, spec)
            except (ValueError, TypeError) as err:
                raise Unsupported(spec_node or node, f"format spec '{spec}' for a {inner}: Python raises "
                                                     f"{type(err).__name__}: {err}")
            if e.value is not None and not isinstance(t, OptionalType):
                return _lit(format(e.value, spec))  # a literal: formatted now
            del py
            spec_code = cpp_string(spec)
        else:
            spec_code = self.join_code(spec)
        arg = self.as_string(e) if t == STR else self.as_type(e, t) if isinstance(t, Scalar) else e
        c = f"qt::py::format({arg.code}, {spec_code})"
        cond = self._flag_cond(node, e) if t == FLOAT and not (isinstance(spec, str) and spec[-1:] in tuple("eEfFgG%")) else None
        if cond:
            c = f"qt::py::format({arg.code}, {spec_code}, {cond})"
        # None with a spec raises TypeError, and a spec made at run time may be one Python rejects
        raises = isinstance(t, OptionalType) or not isinstance(spec, str) or _may_throw(arg.code)
        if raises:
            self.text_may_raise = True
        return Piece(code=c, stream=c, raises=raises)

    # --- f-strings, str(), repr(), format(), str.format -------------------------------------
    def _spec(self, spec_node):
        """An f-string field's format spec: a literal string, or pieces (it has {fields} itself)."""
        if spec_node is None:
            return ""
        pieces = _merge(self.text_pieces(spec_node))
        if all(p.lit is not None for p in pieces):
            return "".join(p.lit for p in pieces)
        return pieces

    def field_piece(self, value, conv, spec, spec_node=None) -> Piece:
        if conv == "a":
            raise Unsupported(value, "the !a conversion (ascii()) is not supported", "use !r")
        if conv is None:
            return self.format_piece(value, self.expr_raw(value), spec, spec_node)
        p = self.repr_piece(value) if conv == "r" else self.str_piece(value)
        if spec == "":
            return p
        if p.lit is not None and isinstance(spec, str):
            try:
                return _lit(format(p.lit, spec))
            except (ValueError, TypeError) as err:
                raise Unsupported(spec_node or value, f"format spec '{spec}' for a str: Python raises "
                                                      f"{type(err).__name__}: {err}")
        text = Expr(p.code if p.std else f"std::string({p.code})", P_ATOM, STR)
        return self.format_piece(value, text, spec, spec_node)

    def text_pieces(self, node) -> list:
        """The pieces of a text-valued expression; any other value as str() shows it (print)."""
        if isinstance(node, ast.JoinedStr):
            out = []
            for v in node.values:
                if isinstance(v, ast.Constant):
                    out.append(_lit(v.value))
                else:
                    out.append(self.field_piece(v.value, _CONV[v.conversion], self._spec(v.format_spec), v.format_spec))
            return _merge(out)
        if isinstance(node, ast.Call) and not node.keywords and isinstance(node.func, ast.Name) \
                and not self.is_variable(node.func.id) and node.func.id not in self.mod.sigs:
            name, args = node.func.id, node.args
            if name in ("str", "repr") and len(args) == 1:
                return [self.str_piece(args[0]) if name == "str" else self.repr_piece(args[0])]
            if name == "str" and not args:
                return [_lit("")]
            if name == "format" and len(args) in (1, 2):
                spec = ""
                if len(args) == 2:
                    se = self.expr(args[1])
                    if se.type != STR:
                        raise Unsupported(args[1], f"format()'s spec must be a str, got {se.type}")
                    spec = se.value if isinstance(se.value, str) else [self.str_piece(args[1], se)]
                return [self.format_piece(args[0], self.expr_raw(args[0]), spec, args[1] if len(args) == 2 else None)]
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == "format" \
                and isinstance(node.func.value, ast.Constant) and isinstance(node.func.value.value, str):
            return self.str_format(node)
        if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Mod) and isinstance(node.left, ast.Constant) \
                and isinstance(node.left.value, str):
            return self.percent_format(node)
        return [self.str_piece(node)]

    def percent_format(self, node) -> list:
        """"%-8s %6.2f %d" % (name, x, n): each conversion as the format spec that writes the same
        (checked against Python on sample values before it is used)."""
        fmt = node.left.value
        values = list(node.right.elts) if isinstance(node.right, ast.Tuple) else [node.right]
        if not isinstance(node.right, ast.Tuple):
            t = self.expr_raw(node.right).type
            if isinstance(t, TupleType):
                raise Unsupported(node.right, "% formatting with a tuple variable is not supported",
                                  "write the values as a tuple: \"...\" % (a, b)")
        out, k, pos = [], 0, 0
        for m in _PERCENT.finditer(fmt):
            out.append(_lit(fmt[pos:m.start()]))
            pos = m.end()
            key, flags, width, prec, conv = m.groups()
            if conv == "%":
                out.append(_lit("%"))
                continue
            if key is not None or width == "*" or prec == "*":
                raise Unsupported(node, f"the conversion {m.group(0)!r} (a mapping key or a * width) is not supported",
                                  "use an f-string")
            if k >= len(values):
                raise Unsupported(node, "not enough arguments for the format string: Python raises TypeError")
            value = values[k]
            k += 1
            out.append(self._percent_piece(node, m.group(0), value, flags, width, prec, conv))
        if "%" in fmt[pos:]:
            raise Unsupported(node, f"the format string {fmt!r} has an incomplete conversion: Python raises ValueError")
        out.append(_lit(fmt[pos:]))
        if k != len(values):
            raise Unsupported(node, "not all arguments converted during string formatting: Python raises TypeError")
        return _merge(out)

    def _percent_piece(self, node, text, value, flags, width, prec, conv) -> Piece:
        e = self.expr_raw(value)
        t = e.type.inner if isinstance(e.type, OptionalType) else e.type
        align = "<" if "-" in flags else ""
        sign = "+" if "+" in flags else " " if " " in flags else ""
        zero = "0" if "0" in flags and not align else ""
        alt = "#" if "#" in flags else ""
        w, p = width or "", f".{prec}" if prec is not None else ""
        if conv in "sr":
            spec = f"{align or '>'}{w}{p}"  # % right-aligns text too
            piece = self.field_piece(value, conv, spec if (w or p) else "")
            sample = "abc"
        elif conv == "a":
            raise Unsupported(node, "%a (ascii()) is not supported", "use %r")
        elif t == STR:
            raise Unsupported(value, f"{text} needs a number, but this is a str: Python raises TypeError")
        elif conv in "diu":
            if prec is not None:
                raise Unsupported(node, f"{text}: a precision with %d (minimum digits) is not supported")
            spec = f"{align}{sign}{zero}{w}d"
            if t == FLOAT:  # Python's %d of a float is int(x): truncated toward zero
                e = Expr(f"qt::to_int({e.code})", P_ATOM, INT)
            piece = self.format_piece(value, e, spec)
            sample = 7
        elif conv in "xXoc":
            if t == FLOAT:
                raise Unsupported(value, f"{text} needs an int, but this is a float: Python raises TypeError")
            spec = f"{align}{sign}{alt}{zero}{w}{conv}"
            piece = self.format_piece(value, e, spec)
            sample = 65
        else:  # e E f F g G
            spec = f"{align}{sign}{alt}{zero}{w}{p or '.6'}{conv}"
            piece = self.format_piece(value, self.as_double(e) if t in (INT, BOOL) else e, spec)
            sample = 1234.5678
        # the spec must write what % writes: tried on a sample value (and a negative one)
        def outcome(f, v):
            try:
                return f(v)
            except (ValueError, TypeError, OverflowError):
                return "error"
        if conv not in "sr":
            for v in (sample, -sample):
                a = outcome(lambda v: text % (v,), v)
                b = outcome(lambda v: format(int(v) if conv in "diu" else v, spec), v)
                if a != b:
                    raise Unsupported(node, f"the conversion {text!r} has no exact format-spec equivalent here",
                                      "use an f-string")
        return piece

    def str_format(self, node) -> list:
        fmt = node.func.value.value
        try:
            fields = list(string.Formatter().parse(fmt))
        except ValueError as err:
            raise Unsupported(node, f"the format string: Python raises ValueError: {err}")
        args, kwargs = node.args, {k.arg: k.value for k in node.keywords}
        if any(isinstance(a, ast.Starred) for a in args) or None in kwargs:
            raise Unsupported(node, "*args / **kwargs in .format() are not supported")
        out, auto, manual, used = [], 0, False, []
        for literal, field, spec, conv in fields:
            if literal:
                out.append(_lit(literal))
            if field is None:
                continue
            if "{" in (spec or ""):
                raise Unsupported(node, "a {field} inside a .format() spec is not supported", "use an f-string")
            if field == "":
                if manual:
                    raise Unsupported(node, "the format string mixes {} and {0}: Python raises ValueError")
                key, auto = auto, auto + 1
            elif field.isdigit():
                if auto:
                    raise Unsupported(node, "the format string mixes {} and {0}: Python raises ValueError")
                key, manual = int(field), True
            elif field.isidentifier():
                key = field
            else:
                raise Unsupported(node, f"the field {{{field}}}: attribute and index fields are not supported",
                                  "pass the value itself")
            if isinstance(key, int):
                if key >= len(args):
                    raise Unsupported(node, f"the format string uses argument {key} but .format() has "
                                            f"{len(args)}: Python raises IndexError")
                value = args[key]
            else:
                if key not in kwargs:
                    raise Unsupported(node, f"the format string uses {{{key}}} but .format() has no {key}=: "
                                            "Python raises KeyError")
                value = kwargs[key]
            used.append(id(value))
            out.append(self.field_piece(value, conv, spec or ""))
        for a in list(args) + list(kwargs.values()):
            n = used.count(id(a))
            simple = isinstance(a, (ast.Name, ast.Constant)) or (isinstance(a, ast.Attribute) and isinstance(a.value, ast.Name))
            if n != 1 and not simple:
                raise Unsupported(a, f"`{ast.unparse(a)}` is used {n} times by the format string: Python works it "
                                     "out once", "assign it to a variable first")
        return _merge(out)

    def join_code(self, pieces) -> str:
        """The pieces as one std::string expression."""
        pieces = _merge(pieces)
        if not pieces:
            return 'std::string()'
        codes = []
        for k, p in enumerate(pieces):
            if p.lit is not None:
                codes.append(f"std::string({cpp_string(p.lit)})" if k == 0 and len(pieces) > 1 and not (
                    pieces[1].lit is None and pieces[1].std) else cpp_string(p.lit))
            elif k == 0 and not p.std:
                codes.append(f"std::string({p.code})")
            else:
                codes.append(p.code)
        if len(pieces) == 1 and pieces[0].lit is not None:
            return f"std::string({cpp_string(pieces[0].lit)})"
        throwing = [p for p in pieces if p.lit is None and (p.raises if p.raises is not None else _may_throw(p.code))]
        if len(throwing) > 1:
            # Python works the pieces out left to right, so the first that raises is the error; the
            # operands of + have no order in C++, the elements of a braced list do
            return "qt::py::concat_text({" + ", ".join(
                cpp_string(p.lit) if p.lit is not None else p.code for p in pieces) + "})"
        return " + ".join(codes)

    def text_value(self, node) -> Expr:
        """An f-string, str(x), repr(x), format(x, spec) or "...".format(...) as a str value."""
        pieces = _merge(self.text_pieces(node))
        if len(pieces) == 1 and pieces[0].lit is not None:
            return Expr(cpp_string(pieces[0].lit), P_ATOM, STR, value=pieces[0].lit, simple=True)
        if not pieces:
            return Expr('""', P_ATOM, STR, value="", simple=True)
        if len(pieces) == 1 and pieces[0].std:
            return Expr(pieces[0].code, P_ATOM if pieces[0].code.endswith(")") else P_ADD, STR)
        return Expr(self.join_code(pieces), P_ADD, STR)

    # --- print() ------------------------------------------------------------------------------
    def print_statement(self, call) -> str:
        """print(...) as one C++ statement: std::cout << ... ;"""
        if any(isinstance(a, ast.Starred) for a in call.args):
            raise Unsupported(call, "print(*values) is not supported", "print the values one by one")
        sep, end, stream, flush = [_lit(" ")], [_lit("\n")], "std::cout", False
        for k in call.keywords:
            if k.arg in ("sep", "end"):
                v = k.value
                if isinstance(v, ast.Constant) and v.value is None:
                    continue
                e = self.expr(v)
                if e.type != STR:
                    raise Unsupported(v, f"print()'s {k.arg} must be a str or None: Python raises TypeError")
                pieces = self.text_pieces(v)
                if k.arg == "sep":
                    sep = pieces
                else:
                    end = pieces
            elif k.arg == "file":
                q = self.qualify(k.value)
                if q == "sys.stdout":
                    continue
                if q != "sys.stderr":
                    raise Unsupported(k.value, "print(file=...) is supported for sys.stdout and sys.stderr only")
                stream = "std::cerr"
            elif k.arg == "flush":
                if not (isinstance(k.value, ast.Constant) and isinstance(k.value.value, bool)):
                    raise Unsupported(k.value, "print(flush=...) must be True or False")
                flush = k.value.value
            else:
                raise Unsupported(call, f"print() has no argument '{k.arg}'")
        pieces = []
        for i, a in enumerate(call.args):
            if i:
                pieces += sep
            pieces += self.text_pieces(a)
        pieces = _merge(pieces + end)
        tail = " << std::flush" if flush else ""
        if not pieces:
            return f"{stream}{tail};" if flush else ""
        if self._prints_inside(call) or any(p.lit is None and (p.raises if p.raises is not None else _may_throw(p.code))
                                            for p in pieces):
            # Python works out the whole line before writing any of it: a function called here
            # that prints too prints first, and an exception leaves nothing of the line written.
            # So the line is made first, then written.
            return f"{stream} << ({self.join_code(pieces)}){tail};"
        parts = [_stream_lit(p.lit) if p.lit is not None else p.stream for p in pieces]
        return f"{stream} << {' << '.join(parts)}{tail};"

    def _prints_inside(self, node) -> bool:
        printing = getattr(self.mod, "printing", set())
        return any(isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id in printing
                   for n in ast.walk(node))
