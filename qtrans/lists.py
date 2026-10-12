"""Lists: a Python list of floats, ints or strings as a std::vector, written as a C++ programmer
would write it (xs.push_back(x), xs.size(), for (double x : xs)), with qtrans_list.hpp where
Python's rules differ from std::vector's: an index may be negative and is checked (IndexError),
slices clip, sum() adds floats as CPython does, min()/max()/sorted() pick as Python picks, and
printing shows the list as Python does ([1.0, 2.5]).

What stays out, refused with the reason: lists holding several types ([1, 2.5] keeps each
element's type in Python, and prints them differently), lists of lists or arrays, and any
sharing a C++ copy would break: in Python `b = a` makes one list with two names, so where either
is changed afterwards (append, xs[i] = ..., +=, sort), or a function changes a list passed to it,
the translation is refused (see FunctionTranslator.check_lists).
"""
from __future__ import annotations

import ast
import sys

from .core import P_ATOM, P_UNARY, Expr, wrap
from .errors import Unsupported
from .types import BOOL, FLOAT, INT, STR, VOID, ArrayType, ListType, Scalar, is_array

ELEMS = (FLOAT, INT, STR)
MUTATORS = {"append", "extend", "insert", "pop", "sort", "reverse", "clear", "remove"}


def list_of(t) -> bool:
    return isinstance(t, ListType)


class ListOps:
    # --- [a, b, c] ---------------------------------------------------------------------------
    def e_List(self, node):
        if not node.elts:
            return Expr("{}", P_ATOM, ListType(None), value=[], simple=True)
        items = [self.expr(e) for e in node.elts]
        types = {e.type for e in items}
        if len(types) != 1:
            names = ", ".join(sorted(str(t) for t in types))
            raise Unsupported(node, f"a list holding several types ({names}): Python keeps each element's type "
                                    "(an int prints as 1, a float as 1.0), a std::vector has one",
                              "make the elements one type: 1.0 rather than 1")
        t = types.pop()
        if t not in ELEMS:
            raise Unsupported(node, f"a list of {t} values is not supported", "lists of float, int or str are")
        codes = [self.as_type(e, t).code if t == INT else e.code for e in items]
        return Expr(f"std::vector<{t.cpp()}>{{{', '.join(codes)}}}", P_ATOM, ListType(t))

    def _need_list(self, e, node, what):
        if not list_of(e.type):
            raise Unsupported(node, f"{what} needs a list, got a {e.type}")
        if e.type.elem is None:
            raise Unsupported(node, f"the elements' type of `{ast.unparse(node)}` is not known (an empty list "
                                    "that nothing is added to)", "annotate it: xs: list[float] = []")

    # --- xs[i], xs[a:b] -------------------------------------------------------------------------
    def list_subscript(self, node, base):
        self._need_list(base, node.value, "indexing")
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
                    parts.append(self.as_type(e, INT).code)
            if parts[2] == "qt::py::none":
                parts = parts[:2]
            return Expr(f"qt::py::slice({base.code}, {', '.join(parts)})", P_ATOM, base.type)
        i = self.expr(sl)
        if i.type not in (INT, BOOL):
            raise Unsupported(sl, f"list indices must be ints (Python raises TypeError for a {i.type})")
        if self.index_in_range(node.value, sl):  # for i in range(len(xs)): xs[i] -- no check needed
            return Expr(f"{wrap(base, P_ATOM)}[{i.code}]", P_ATOM, base.type.elem)
        return Expr(f"qt::py::item({base.code}, {self.as_type(i, INT).code})", P_ATOM, base.type.elem)

    # --- for k in range(len(xs)): xs[k] is always in range ------------------------------------
    @staticmethod
    def range_over_len(it, body) -> set:
        """The lists a range() loop's index stays inside: `range(len(xs))`, `range(1, len(xs))`,
        `range(min(len(xs), len(ys)))`, when the loop (body: its statements or expressions)
        neither changes the list's length nor rebinds it."""
        from .loops import _changes
        if not (isinstance(it, ast.Call) and isinstance(it.func, ast.Name) and it.func.id == "range") \
                or it.keywords or not 1 <= len(it.args) <= 3:
            return set()
        def const(n):
            return n.value if isinstance(n, ast.Constant) and type(n.value) is int else None
        if len(it.args) >= 2 and (const(it.args[0]) is None or const(it.args[0]) < 0):
            return set()
        if len(it.args) == 3 and (const(it.args[2]) is None or const(it.args[2]) <= 0):
            return set()
        stop = it.args[0] if len(it.args) == 1 else it.args[1]
        lens = stop.args if isinstance(stop, ast.Call) and isinstance(stop.func, ast.Name) and stop.func.id == "min" \
            and not stop.keywords else [stop]
        names = set()
        for n in lens:
            if not (isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id == "len"
                    and len(n.args) == 1 and isinstance(n.args[0], ast.Name)):
                return set()
            names.add(n.args[0].id)
        stmts = [b if isinstance(b, ast.stmt) else ast.Expr(value=b) for b in body]
        return {x for x in names if _changes(stmts, x) is None}

    def index_in_range(self, seq, index) -> bool:
        if not (isinstance(seq, ast.Name) and isinstance(index, ast.Name)):
            return False
        return seq.id in self.__dict__.get("safe_index", {}).get(index.id, ())

    def _safe_index_push(self, var, names):
        safe = self.__dict__.setdefault("safe_index", {})
        saved = safe.get(var)
        if names:
            safe[var] = names
        else:
            safe.pop(var, None)
        return saved

    def _safe_index_pop(self, var, saved):
        safe = self.__dict__.setdefault("safe_index", {})
        if saved is None:
            safe.pop(var, None)
        else:
            safe[var] = saved

    # --- methods ---------------------------------------------------------------------------------
    def list_method(self, node, base, name):
        t = base.type
        args = node.args
        if node.keywords and not (name == "sort" and all(k.arg == "reverse" for k in node.keywords)):
            raise Unsupported(node, f"keyword arguments to list.{name}() are not supported")

        def elem_arg(a):
            e = self.expr(a)
            if t.elem is not None and e.type != t.elem:
                raise Unsupported(a, f"adding a {e.type} to a list of {t.elem}: Python keeps its type (an int prints "
                                     "as 1, a float as 1.0), a std::vector has one",
                                  "convert it: float(x)" if t.elem == FLOAT else None)
            if self._list_maybe_int(a, t.elem):
                raise Unsupported(a, "this value may be a Python int here, and the list's other elements floats",
                                  "make it a float on every path (0.0 rather than 0)")
            return self.as_type(e, INT) if t.elem == INT else self.as_string(e) if t.elem == STR else e
        if name == "append":
            self._arity(node, args, 1)
            return Expr(f"{wrap(base, P_ATOM)}.push_back({elem_arg(args[0]).code})", P_ATOM, VOID)
        if name == "extend":
            self._arity(node, args, 1)
            other = self.expr(args[0])
            self._need_list(other, args[0], "extend()")
            if other.type.elem != t.elem:
                raise Unsupported(args[0], f"extending a list of {t.elem} with a list of {other.type.elem}")
            if isinstance(args[0], ast.Name) and isinstance(node.func.value, ast.Name) and args[0].id == node.func.value.id:
                return Expr(f"{base.code} = qt::py::concat({base.code}, {base.code})", P_ATOM, VOID)
            src = other.code if other.simple else f"std::vector<{t.elem.cpp()}>({other.code})"
            if other.simple:
                return Expr(f"{base.code}.insert({base.code}.end(), {src}.begin(), {src}.end())", P_ATOM, VOID)
            return Expr(f"{base.code} = qt::py::concat({base.code}, {other.code})", P_ATOM, VOID)
        if name == "pop":
            self._arity(node, args, 0, 1)
            i = f", {self.as_type(self.expr(args[0]), INT).code}" if args else ""
            return Expr(f"qt::py::pop({base.code}{i})", P_ATOM, t.elem)
        if name == "insert":
            self._arity(node, args, 2)
            i = self.expr(args[0])
            if i.type not in (INT, BOOL):
                raise Unsupported(args[0], "list.insert()'s index must be an int")
            return Expr(f"qt::py::insert({base.code}, {self.as_type(i, INT).code}, {elem_arg(args[1]).code})",
                        P_ATOM, VOID)
        if name in ("index", "count"):
            self._arity(node, args, 1)
            x = elem_arg(args[0])
            return Expr(f"qt::py::{name}({base.code}, {x.code})", P_ATOM, INT)
        if name == "copy":
            self._arity(node, args, 0)
            return Expr(f"{t.cpp()}({base.code})", P_ATOM, t)
        if name == "sort":
            self._arity(node, args, 0)
            rev = self._reverse_kw(node)
            return Expr(f"qt::py::sort({base.code}{', true' if rev else ''})", P_ATOM, VOID)
        if name == "reverse":
            self._arity(node, args, 0)
            return Expr(f"std::reverse({base.code}.begin(), {base.code}.end())", P_ATOM, VOID)
        if name == "clear":
            self._arity(node, args, 0)
            return Expr(f"{base.code}.clear()", P_ATOM, VOID)
        raise Unsupported(node, f"list.{name}() is not supported",
                          "supported: append, extend, insert, pop, index, count, copy, sort, reverse, clear")

    def _reverse_kw(self, node) -> bool:
        for k in node.keywords:
            if k.arg == "reverse":
                if not (isinstance(k.value, ast.Constant) and isinstance(k.value.value, bool)):
                    raise Unsupported(k.value, "reverse= must be True or False")
                return k.value.value
            raise Unsupported(node, f"sorted() / sort() argument '{k.arg}' is not supported")
        return False

    def _list_maybe_int(self, node, elem) -> bool:
        return elem == FLOAT and hasattr(self, "maybe_int") and bool(self.maybe_int(node))

    # --- builtins of a list -------------------------------------------------------------------
    def list_builtin(self, node, name):
        """len / sum / min / max / sorted / list of a list (None if the argument isn't one)."""
        if not node.args:
            return None
        if name == "list" and len(node.args) == 1 and isinstance(node.args[0], ast.Call) \
                and isinstance(node.args[0].func, ast.Name) and node.args[0].func.id == "range" \
                and not self.is_variable("range"):
            r = node.args[0]
            vals = [self.as_type(self.expr(a), INT) for a in r.args]
            if not 1 <= len(vals) <= 3 or any(v.type != INT for v in vals):
                raise Unsupported(r, "range() takes 1 to 3 int arguments")
            codes = ["0LL"] + [v.code for v in vals] if len(vals) == 1 else [v.code for v in vals]
            return Expr(f"qt::py::range_list({', '.join(codes)})", P_ATOM, ListType(INT))
        a = self.expr(node.args[0])
        if not list_of(a.type):
            return None
        self._need_list(a, node.args[0], f"{name}()")
        t = a.type
        if name == "len":
            self._arity(node, node.args, 1)
            return Expr(f"static_cast<long long>({wrap(a, P_ATOM)}.size())", P_ATOM, INT)
        if name == "sum":
            self._arity(node, node.args, 1)
            if t.elem == STR:
                raise Unsupported(node, "sum() of strings raises TypeError in Python", "use ''.join(...)")
            if t.elem == FLOAT:
                # CPython 3.12 adds floats with a compensated sum; 3.10 and 3.11 left to right
                fn = "qt::py::sum" if sys.version_info >= (3, 12) else "qt::py::sum_left"
                self.note("listsum", "sum() of a list of floats is CPython's: compensated (Neumaier) from 3.12, "
                                     "left to right before; this translation follows the Python running qtrans "
                                     f"({sys.version_info.major}.{sys.version_info.minor})")
                return Expr(f"{fn}({a.code})", P_ATOM, FLOAT)
            return Expr(f"qt::py::sum({a.code})", P_ATOM, INT)
        if name in ("min", "max"):
            if len(node.args) != 1 or node.keywords:
                raise Unsupported(node, f"{name}() of a list with more arguments (key=, default=) is not supported")
            return Expr(f"qt::py::{name}({a.code})", P_ATOM, t.elem)
        if name == "sorted":
            self._arity(node, node.args, 1)
            rev = self._reverse_kw(node)
            return Expr(f"qt::py::sorted({a.code}{', true' if rev else ''})", P_ATOM, t)
        if name == "list":
            self._arity(node, node.args, 1)
            return Expr(f"{t.cpp()}({a.code})", P_ATOM, t)
        return None

    # --- x in xs, xs + ys ---------------------------------------------------------------------
    def list_contains(self, node, x_node, xs_node, negate):
        xs = self.expr(xs_node)
        self._need_list(xs, xs_node, "`in`")
        x = self.expr(x_node)
        if x.type != xs.type.elem and not (xs.type.elem == FLOAT and x.type in (INT, BOOL)):
            raise Unsupported(node, f"`in` with a {x.type} and a list of {xs.type.elem}")
        x = self.as_type(x, xs.type.elem) if isinstance(xs.type.elem, Scalar) else self.as_string(x)
        code = f"qt::py::contains({xs.code}, {x.code})"
        if xs.type.elem == FLOAT:
            self.note("listnan", "`x in xs` of floats compares with ==, so NaN is never found; Python also "
                                 "finds the very same NaN object (rarely the same)")
        return Expr(f"!{code}" if negate else code, P_UNARY if negate else P_ATOM, BOOL)

    def list_concat(self, node, L, R):
        for e, n in ((L, node.left), (R, node.right)):
            self._need_list(e, n, "+")
        if L.type.elem != R.type.elem:
            raise Unsupported(node, f"joining a list of {L.type.elem} and a list of {R.type.elem}")
        return Expr(f"qt::py::concat({L.code}, {R.code})", P_ATOM, L.type)

    # --- ", ".join(names) ------------------------------------------------------------------------
    def str_join(self, node, sep):
        if node.keywords or len(node.args) != 1:
            raise Unsupported(node, "str.join() takes one list")
        xs = self.expr(node.args[0])
        if not list_of(xs.type):
            raise Unsupported(node.args[0], f"str.join() of a {xs.type} is not supported", "join a list of str")
        if xs.type.elem not in (STR, None):
            raise Unsupported(node.args[0], f"str.join() of a list of {xs.type.elem}: Python raises TypeError "
                                            "(it joins only strings)", "join [str(x) for x in xs]")
        code = "std::vector<std::string>{}" if xs.type.elem is None else xs.code
        return Expr(f"qt::py::join({sep.code}, {code})", P_ATOM, STR)

    # --- [expr for x in it if cond] -------------------------------------------------------------
    def any_all(self, node, name):
        """any(cond for x in xs) / all(...): Python stops at the first element that decides, so
        the C++ is a loop that returns there (elements after it are never computed)."""
        if node.keywords or len(node.args) != 1 or not isinstance(node.args[0], ast.GeneratorExp):
            raise Unsupported(node, f"{name}() is supported over a generator: {name}(x > 0 for x in xs)")
        return self.e_ListComp(node.args[0], mode=name)

    def e_ListComp(self, node, mode="list"):
        if len(node.generators) != 1 or node.generators[0].is_async:
            raise Unsupported(node, "a list comprehension with several `for`s is not supported",
                              "write the loops out")
        g = node.generators[0]
        if not isinstance(g.target, ast.Name):
            why = getattr(g, "qt_why", None)
            raise Unsupported(g.target, f"this enumerate() / zip() comprehension is not supported: {why}" if why
                              else "a comprehension variable must be one name",
                              None if why else "unpacking works for enumerate() and zip()")
        var = g.target.id
        it = g.iter
        if getattr(g, "qt_rewritten", False):
            self.note("indexloop", "enumerate() / zip() become an index loop: Python's iterator reads element k by position, so "
                  "xs[k] is the same value; zip stops at the shorter sequence (the min of the lengths)")
        if mode != "list":
            self.note("anyall", f"{mode}() of a generator stops at the first element that decides, in C++ "
                                "too: a loop that returns there, so later elements are never computed")
        is_range = isinstance(it, ast.Call) and isinstance(it.func, ast.Name) and it.func.id == "range" \
            and not self.is_variable("range")
        if is_range:
            if it.keywords or not 1 <= len(it.args) <= 3:
                raise Unsupported(it, "range() takes 1 to 3 positional arguments")
            vals = [self.expr(a) for a in it.args]
            if any(v.type not in (INT, BOOL) for v in vals):
                raise Unsupported(it, "range() arguments must be ints")
            step = 1
            if len(vals) == 3:
                step = vals[2].value
                if not isinstance(step, int) or isinstance(step, bool) or step == 0:
                    raise Unsupported(it, "range() step must be a nonzero integer constant")
            start = self.as_type(vals[0], INT).code if len(vals) >= 2 else "0"
            stop = self.as_type(vals[1] if len(vals) >= 2 else vals[0], INT).code
            vt = INT
        else:
            src = self.expr(it)
            if list_of(src.type):
                self._need_list(src, it, "a comprehension")
                vt = src.type.elem
            elif src.type == ArrayType(1):
                vt = FLOAT
            else:
                raise Unsupported(it, f"a comprehension over a {src.type} is not supported",
                                  "over range(...), a list or a Vec")
        ident = var if not self._clashes(var) else var + "_"
        outer = self.__dict__.setdefault("comp_vars", {})
        saved = outer.get(var)
        outer[var] = (vt, ident)
        safe_saved = self._safe_index_push(var, self.range_over_len(it, [node.elt] + list(g.ifs)) if is_range else set())
        try:
            e = self.expr(node.elt)
            conds = [self.truth(self.expr(c), c) for c in g.ifs]
        finally:
            self._safe_index_pop(var, safe_saved)
            if saved is None:
                outer.pop(var, None)
            else:
                outer[var] = saved
        if mode != "list":
            test = self.truth(e, node.elt)
        elif e.type not in ELEMS:
            raise Unsupported(node.elt, f"a list of {e.type} values is not supported", "lists of float, int or str are")
        if mode == "list" and self._list_maybe_int(node.elt, e.type):
            raise Unsupported(node.elt, "this value may be a Python int for some elements and a float for others",
                              "make it a float on every path (0.0 rather than 0)")
        out = "out"
        while out in getattr(self, "all_names", set()) or out == ident:
            out += "_"
        if is_range:
            cmp, inc = ("<", f"++{ident}") if step == 1 else ((">", f"--{ident}") if step == -1 else
                                                               ("<" if step > 0 else ">", f"{ident} += {step}"))
            loop = f"for (long long {ident} = {start}; {ident} {cmp} {stop}; {inc})"
        else:
            decl = "const std::string&" if vt == STR else vt.cpp()
            loop = f"for ({decl} {ident} : {src.code})"
        if mode != "list":
            hit = wrap(test, 13) if mode == "any" else "!" + wrap(test, P_UNARY)
            if not conds and mode == "any":
                hit = test.code
            body = f"if ({' && '.join([wrap(c, 13) for c in conds] + [hit])}) return {'true' if mode == 'any' else 'false'};"
            return Expr(f"[&] {{ {loop} {body} return {'false' if mode == 'any' else 'true'}; }}()", P_ATOM, BOOL)
        body = f"{out}.push_back({e.code});"
        if conds:
            body = f"if ({' && '.join(wrap(c, 13) for c in conds)}) {body}"
        code = f"[&] {{ {ListType(e.type).cpp()} {out}; {loop} {body} return {out}; }}()"
        return Expr(code, P_ATOM, ListType(e.type))

    def _clashes(self, name) -> bool:
        """Would a C++ variable of this name hide another one (-Wshadow)?"""
        return self.is_variable(name) or name in getattr(self.mod, "constants", {}) \
            or name in getattr(self.mod, "sigs", {})
