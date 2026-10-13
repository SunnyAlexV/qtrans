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
from .types import BOOL, FLOAT, INT, STR, VOID, ArrayType, DictType, ListType, Scalar, TupleType, is_array

ELEMS = (FLOAT, INT, STR)


def list_elem_ok(t) -> bool:
    """What a list may hold: floats, ints or strings, or tuples of those (and bools)."""
    return t in ELEMS or (isinstance(t, TupleType) and bool(t.elts) and all(e in ELEMS + (BOOL,) for e in t.elts))
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
        if not list_elem_ok(t):
            raise Unsupported(node, f"a list of {t} values is not supported",
                              "lists of float, int or str (or tuples of them) are")
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
                    parts.append(self.int_arg(e).code)
            if parts[2] == "qt::py::none":
                parts = parts[:2]
            return Expr(f"qt::py::slice({base.code}, {', '.join(parts)})", P_ATOM, base.type)
        i = self.expr(sl)
        if i.type not in (INT, BOOL):
            raise Unsupported(sl, f"list indices must be ints (Python raises TypeError for a {i.type})")
        if self.index_in_range(node.value, sl):  # for i in range(len(xs)): xs[i] -- no check needed
            return Expr(f"{wrap(base, P_ATOM)}[static_cast<std::size_t>({i.code})]", P_ATOM, base.type.elem)
        return Expr(f"qt::py::item({base.code}, {self.int_arg(i).code})", P_ATOM, base.type.elem)

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
        if node.keywords and not (name == "sort" and all(k.arg in ("reverse", "key") for k in node.keywords)):
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
            key = next((k.value for k in node.keywords if k.arg == "key"), None)
            rev = self._reverse_kw(node)
            if key is not None and not (isinstance(key, ast.Constant) and key.value is None):
                keys = self._keys(node, base, key)
                return Expr(f"{base.code} = qt::py::sorted_by({base.code}, {keys}{', true' if rev else ''})",
                            P_ATOM, VOID)
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
            if k.arg == "key":
                continue
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
        a = self.pairs_iterable(node.args[0]) if name in ("list", "sorted", "min", "max") else None
        if a is not None and name == "list" and len(node.args) == 1:
            return a  # list(zip(...)): the list it makes is already a new one
        a = a or self.expr(node.args[0])
        if not list_of(a.type):
            return None
        self._need_list(a, node.args[0], f"{name}()")
        t = a.type
        if name == "len":
            self._arity(node, node.args, 1)
            return Expr(f"static_cast<long long>({wrap(a, P_ATOM)}.size())", P_ATOM, INT)
        if name == "sum" and len(node.args) == 2:
            return self._sum_from(node, a, t)
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
            if len(node.args) != 1 or any(k.arg not in ("key", "default") for k in node.keywords):
                raise Unsupported(node, f"{name}() of a list takes key= and default=")
            key = next((k.value for k in node.keywords if k.arg == "key"), None)
            dflt = next((k.value for k in node.keywords if k.arg == "default"), None)
            keyed = key is not None and not (isinstance(key, ast.Constant) and key.value is None)

            def pick(xs):
                return f"qt::py::{name}_by({xs.code}, {self._keys(node, xs, key)})" if keyed else f"qt::py::{name}({xs.code})"
            if dflt is None:
                return Expr(self._items_once(a, pick) if keyed else pick(a), P_ATOM, t.elem)
            d = self.expr(dflt)  # an empty list gives the default instead of ValueError
            if d.type != t.elem:
                raise Unsupported(dflt, f"default= is a {d.type} for a list of {t.elem}: Python returns it as it "
                                        "is, a value of another type", "give a default of the list's type")
            if t.elem == FLOAT and self.maybe_int(dflt):
                raise Unsupported(dflt, "the default may be a Python int here", "write it as a float: 0.0")
            return Expr(self._items_once(Expr(a.code, P_ATOM, a.type, simple=False),
                                   lambda xs: f"{xs.code}.empty() ? {t.elem.cpp()}({d.code}) : {pick(xs)}"),
                        P_ATOM, t.elem)
        if name == "sorted":
            self._arity(node, node.args, 1)
            rev = self._reverse_kw(node)
            key = next((k.value for k in node.keywords if k.arg == "key"), None)
            if key is not None and not (isinstance(key, ast.Constant) and key.value is None):
                return Expr(self._items_once(a, lambda xs: f"qt::py::sorted_by({xs.code}, {self._keys(node, xs, key)}"
                                                     f"{', true' if rev else ''})"), P_ATOM, t)
            return Expr(f"qt::py::sorted({a.code}{', true' if rev else ''})", P_ATOM, t)
        if name == "list":
            self._arity(node, node.args, 1)
            return Expr(f"{t.cpp()}({a.code})", P_ATOM, t)
        return None

    def _keys(self, node, items, key) -> str:
        """The C++ for the list of keys key=... gives the elements of `items`, computed once each, in
        order: key=lambda x: ..., key=abs / len / a function of the module, key=d.get (over d)."""
        elem = items.type.elem
        if isinstance(key, ast.Lambda):
            a = key.args
            if len(a.args) != 1 or a.posonlyargs or a.kwonlyargs or a.vararg or a.kwarg or a.defaults:
                raise Unsupported(key, "a key function takes one argument: key=lambda x: ...")
            var, body = a.args[0].arg, key.body
        elif isinstance(key, ast.Name):
            var = "x"
            while self.is_variable(var) or var in getattr(self, "comp_vars", {}):
                var += "_"
            body = ast.Call(func=ast.Name(id=key.id, ctx=ast.Load()), args=[ast.Name(id=var, ctx=ast.Load())],
                            keywords=[])
        elif isinstance(key, ast.Attribute) and key.attr == "get" and isinstance(key.value, ast.Name) \
                and isinstance(self.expr(key.value).type, DictType):
            # max(d, key=d.get): over d's own keys, every lookup finds its value
            var = "k"
            while self.is_variable(var) or var in getattr(self, "comp_vars", {}):
                var += "_"
            body = ast.Subscript(value=ast.Name(id=key.value.id, ctx=ast.Load()),
                                 slice=ast.Name(id=var, ctx=ast.Load()), ctx=ast.Load())
            if not self._is_keys_of(node.args[0] if node.args else None, key.value.id):
                raise Unsupported(key, "key=d.get is supported over d itself (max(d, key=d.get)): elsewhere a "
                                       "missing key gives None")
        else:
            raise Unsupported(key, "key= takes a lambda, a function's name, or d.get",
                              "key=lambda x: x[1]")
        for n in ast.walk(body):
            if isinstance(n, (ast.Lambda, ast.NamedExpr, ast.Yield, ast.Await)):
                raise Unsupported(n, "a key function holding a lambda or := is not supported")
        ast.copy_location(body, key)
        for n in ast.walk(body):
            if "lineno" in n._attributes and not hasattr(n, "lineno"):
                ast.copy_location(n, key)
        ident = var if not self._clashes(var) else var + "_"
        outer = self.__dict__.setdefault("comp_vars", {})
        saved = outer.get(var)
        outer[var] = (elem, ident)
        try:
            k = self.expr(body)
        finally:
            if saved is None:
                outer.pop(var, None)
            else:
                outer[var] = saved
        if not list_elem_ok(k.type) and k.type != BOOL:
            raise Unsupported(key, f"a key of type {k.type} can't be compared as Python compares it")
        decl = "const std::string&" if elem == STR else ("const auto&" if isinstance(elem, TupleType) else elem.cpp())
        kt = ListType(k.type).cpp()
        return (f"[&] {{ {kt} qt_keys; qt_keys.reserve({wrap(items, P_ATOM)}.size()); for ({decl} {ident} : {items.code}) "
                f"qt_keys.push_back({k.code}); return qt_keys; }}()")

    @staticmethod
    def _items_once(a, make) -> str:
        """make(items) where items is `a` evaluated once: a variable as it is, anything else (a call,
        sorted(d.items())) into one named copy first, as Python evaluates it once."""
        if a.simple:
            return make(a)
        tmp = Expr("qt_items", P_ATOM, a.type, simple=True)
        return f"[&] {{ const auto qt_items = {a.code}; return {make(tmp)}; }}()"

    @staticmethod
    def _is_keys_of(arg, d) -> bool:
        if isinstance(arg, ast.Name) and arg.id == d:
            return True
        return isinstance(arg, ast.Call) and isinstance(arg.func, ast.Attribute) and arg.func.attr == "keys" \
            and isinstance(arg.func.value, ast.Name) and arg.func.value.id == d

    def pairs_iterable(self, it):
        """zip(xs, ys) / enumerate(xs) / d.items(), read to the end by list() / sorted() / dict() /
        min() / max(): the list of tuples Python's iterator gives. None if `it` is none of these."""
        if not isinstance(it, ast.Call) or it.keywords and not (
                isinstance(it.func, ast.Name) and it.func.id == "enumerate" and all(k.arg == "start" for k in it.keywords)):
            return None
        f = it.func
        if isinstance(f, ast.Name) and f.id == "zip" and not self.is_variable("zip") and it.args:
            parts = [self.expr(a) for a in it.args]
            for a, n in zip(parts, it.args):
                if not list_of(a.type):
                    raise Unsupported(n, f"zip() of a {a.type} as a value is supported for lists",
                                      "loop over it directly: for a, b in zip(xs, ys)")
                self._need_list(a, n, "zip()")
            t = ListType(TupleType(tuple(a.type.elem for a in parts)))
            return Expr(f"qt::py::zip({', '.join(a.code for a in parts)})", P_ATOM, t)
        if isinstance(f, ast.Name) and f.id == "enumerate" and not self.is_variable("enumerate") \
                and 1 <= len(it.args) + len(it.keywords) <= 2 and it.args:
            xs = self.expr(it.args[0])
            if not list_of(xs.type):
                raise Unsupported(it.args[0], f"enumerate() of a {xs.type} as a value is supported for lists",
                                  "loop over it directly: for i, x in enumerate(xs)")
            self._need_list(xs, it.args[0], "enumerate()")
            start = it.args[1] if len(it.args) == 2 else (it.keywords[0].value if it.keywords else None)
            extra = ""
            if start is not None:
                st = self.expr(start)
                if st.type not in (INT, BOOL):
                    raise Unsupported(start, "enumerate()'s start must be an int")
                extra = f", {self.as_type(st, INT).code}"
            return Expr(f"qt::py::enumerate({xs.code}{extra})", P_ATOM, ListType(TupleType((INT, xs.type.elem))))
        if isinstance(f, ast.Attribute) and f.attr == "items" and not it.args:
            d = self.expr(f.value)
            if isinstance(d.type, DictType):
                self._need_dict(d, f.value, "items()")
                return Expr(f"qt::py::items({d.code})", P_ATOM, ListType(TupleType((d.type.key, d.type.val))))
        return None

    def _sum_from(self, node, a, t):
        """sum(xs, start): start + the elements, in order. A float start makes the result a float
        whatever xs holds (empty too): the usual way to keep sum() a float."""
        if node.keywords and not all(k.arg == "start" for k in node.keywords):
            raise Unsupported(node, "sum() takes a list and a start")
        st = self.expr(node.args[1])
        if t.elem == FLOAT and st.type == FLOAT:
            if self.maybe_int(node.args[1]):
                raise Unsupported(node.args[1], "the start may be a Python int here", "write it as a float: 0.0")
            fn = "qt::py::sum" if sys.version_info >= (3, 12) else "qt::py::sum_left"
            self.note("listsum", "sum() of a list of floats is CPython's: compensated (Neumaier) from 3.12, "
                                 "left to right before; this translation follows the Python running qtrans "
                                 f"({sys.version_info.major}.{sys.version_info.minor})")
            return Expr(f"{fn}({a.code}, {st.code})", P_ATOM, FLOAT)
        if t.elem == INT and st.type in (INT, BOOL):  # ints add exactly: start + their sum
            return Expr(f"qt::iadd({self.as_type(st, INT).code}, qt::py::sum({a.code}))", P_ATOM, INT)
        if t.elem == FLOAT and st.type in (INT, BOOL) and st.value == 0:  # sum(xs, 0) is sum(xs)
            return self.list_builtin(ast.copy_location(ast.Call(func=node.func, args=[node.args[0]], keywords=[]),
                                                       node), "sum")
        raise Unsupported(node, f"sum() of a list of {t.elem} from a {st.type} start is not supported",
                          "a float start for floats (0.0), an int start for ints")

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
        if mode in ("any", "all"):
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
            if isinstance(src.type, DictType):  # over a dict: its keys
                self._need_dict(src, it, "a comprehension")
                src = Expr(f"{src.code}.keys()", P_ATOM, ListType(src.type.key))
            if src.type == STR:  # over a str: its characters
                src = self.str_chars(src)
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
        parts = [node.key, node.value] if mode == "dict" else [node.elt]
        safe_saved = self._safe_index_push(var, self.range_over_len(it, parts + list(g.ifs)) if is_range else set())
        try:
            if mode == "dict":
                ke, e = self.expr(node.key), self.expr(node.value)
            else:
                e = self.expr(node.elt)
            conds = [self.truth(self.expr(c), c) for c in g.ifs]
        finally:
            self._safe_index_pop(var, safe_saved)
            if saved is None:
                outer.pop(var, None)
            else:
                outer[var] = saved
        if mode == "dict":
            kt = self._one_type(node.key, [ke], "keys", (STR, INT))
            vt_ = self._one_type(node.value, [e], "values", ELEMS)
            if self._list_maybe_int(node.value, e.type):
                raise Unsupported(node.value, "this value may be a Python int for some keys and a float for others",
                                  "make it a float on every path (0.0 rather than 0)")
        elif mode != "list":
            test = self.truth(e, node.elt)
        elif not list_elem_ok(e.type):
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
        if mode in ("any", "all"):
            hit = wrap(test, 13) if mode == "any" else "!" + wrap(test, P_UNARY)
            if not conds and mode == "any":
                hit = test.code
            body = f"if ({' && '.join([wrap(c, 13) for c in conds] + [hit])}) return {'true' if mode == 'any' else 'false'};"
            return Expr(f"[&] {{ {loop} {body} return {'false' if mode == 'any' else 'true'}; }}()", P_ATOM, BOOL)
        if mode == "dict":
            # Python computes the key, then the value (C++'s `out[k] = v` computes v first)
            k_code = self._key(ke, kt, node.key).code
            v_code = self._val(e, vt_).code
            body = f"{out}[{k_code}] = {v_code};" if ke.simple or e.simple else \
                f"{{ {kt.cpp()} key = {k_code}; {out}[key] = {v_code}; }}"
            if conds:
                body = f"if ({' && '.join(wrap(c, 13) for c in conds)}) {body}"
            t = DictType(kt, vt_)
            return Expr(f"[&] {{ {t.cpp()} {out}; {loop} {body} return {out}; }}()", P_ATOM, t)
        body = f"{out}.push_back({e.code});"
        if conds:
            body = f"if ({' && '.join(wrap(c, 13) for c in conds)}) {body}"
        code = f"[&] {{ {ListType(e.type).cpp()} {out}; {loop} {body} return {out}; }}()"
        return Expr(code, P_ATOM, ListType(e.type))

    def _clashes(self, name) -> bool:
        """Would a C++ variable of this name hide another one (-Wshadow)?"""
        return self.is_variable(name) or name in getattr(self.mod, "constants", {}) \
            or name in getattr(self.mod, "sigs", {})
