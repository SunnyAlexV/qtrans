"""Which numbers are Python's and which are NumPy's.

`np.sum(v)` is a NumPy float64, `float(np.sum(v))` a Python float, `v[0]` a NumPy float64,
`len(v)` a Python int, `v.argmax()` a NumPy int64. They compute the same and print() / str() /
format() show them the same, but repr() doesn't: repr(np.float64(2.5)) is 'np.float64(2.5)'
(NumPy 2), and so is a tuple holding one, `print((m, s))`. So wherever repr is shown, the
translation needs to know which kind the value is. This works it out from the Python itself,
the way the value was made: an origin is

  PY   a Python float / int / bool
  NP   made by NumPy (a NumPy scalar, or an array: indexing one gives a NumPy scalar)
  RNG  a numpy.random.Generator (its scalar draws are Python floats)
  None it depends on the run (x = 0.0, then x += v[i] in a loop: a Python float if the loop
       never ran, a NumPy float64 if it did), or qtrans can't tell

and a tuple's origin is ("tuple", (origin of each element)). Variables are solved as a fixpoint
over all their assignments, so a variable is PY only if every value it can hold is.
"""
from __future__ import annotations

import ast

from .types import RNG, ArrayType, DictType, ListType, Scalar, TupleType, is_array, is_arraylike

PY, NP, RNG_ = "py", "np", "rng"
_BOTTOM = "bottom"  # no assignment seen yet (the start of the fixpoint)

_PY_BUILTINS = {"float", "int", "bool", "round", "len", "str", "repr", "format", "isinstance", "divmod"}
_PY_METHODS = {"item", "tolist", "upper", "lower", "capitalize", "title", "swapcase", "casefold", "istitle", "strip", "lstrip", "rstrip", "format", "join",
               "replace", "startswith", "endswith", "find", "zfill", "ljust", "rjust", "center", "isdigit",
               "isdecimal", "isnumeric", "isalpha", "isalnum", "isspace", "isupper", "islower", "isprintable"}
_SCALAR_DRAWS = {"random": 0, "standard_normal": 0, "normal": 2, "uniform": 2}  # positional args before size


def join(a, b):
    if a == _BOTTOM:
        return b
    if b == _BOTTOM:
        return a
    if isinstance(a, tuple) and isinstance(b, tuple) and a[0] == b[0] and a[0] in ("list", "dict"):
        return (a[0], join(a[1], b[1]))  # a list's elements / a dict's values
    if isinstance(a, tuple) and isinstance(b, tuple) and a[0] == b[0] == "tuple" and len(a[1]) == len(b[1]):
        return ("tuple", tuple(join(x, y) for x, y in zip(a[1], b[1])))
    return a if a == b else None


def arith(*xs):
    """An operator's result: NumPy's if any operand is NumPy's (np.float64 + 1 is a float64)."""
    if any(x == _BOTTOM for x in xs):
        return _BOTTOM
    if any(x is None or isinstance(x, tuple) or x == RNG_ for x in xs):
        return None
    return NP if NP in xs else PY


def binop(left, right):
    """An operator's result, lists included (a + b holds both lists' elements)."""
    if isinstance(left, tuple) and left[0] == "list" or isinstance(right, tuple) and right[0] == "list":
        if isinstance(left, tuple) and isinstance(right, tuple) and left[0] == right[0] == "list":
            return ("list", join(left[1], right[1]))
        return left if isinstance(left, tuple) and left[0] == "list" else right  # xs * n: xs's elements
    return arith(left, right)


def of_type(t):
    """The origin of a parameter of type t: the caller's value (verify passes Python numbers)."""
    if isinstance(t, Scalar):
        return PY
    if is_arraylike(t):
        return NP
    if t == RNG:
        return RNG_
    if isinstance(t, TupleType):
        return ("tuple", tuple(of_type(e) for e in t.elts))
    if isinstance(t, ListType):
        return ("list", of_type(t.elem) if t.elem is not None else PY)  # tuples: ("tuple", ...)
    if isinstance(t, DictType):
        return ("dict", of_type(t.val) if t.val is not None else PY)
    return None


class Origins:
    """One per module: origins of the values in each function, and of what each function returns."""

    def __init__(self, mod):
        self.mod = mod
        self._functions = {}
        self._returns = {}
        self._busy = set()

    def function(self, fdef, sig=None):
        key = id(fdef)
        if key not in self._functions:
            self._functions[key] = _FunctionOrigins(self, fdef, sig)
        return self._functions[key]

    def returns(self, name):
        """The origin of what user function `name` returns."""
        if name in self._returns:
            return self._returns[name]
        fdef = self.mod.fdef_named(name) if hasattr(self.mod, "fdef_named") else self.mod.fdefs.get(name)
        if name in self._busy or fdef is None:
            return None  # recursion: can't tell
        sig = self.mod.sigs.get(name)
        if sig is not None and (is_array(sig.ret) or isinstance(sig.ret, ArrayType)):
            return NP
        self._busy.add(name)
        try:
            fo = self.function(fdef, sig)
            out = _BOTTOM
            for n in _own_nodes(fo.fdef):
                if isinstance(n, ast.Return) and n.value is not None:
                    out = join(out, fo.of(n.value))
            out = None if out == _BOTTOM else out
        finally:
            self._busy.discard(name)
        self._returns[name] = out
        return out


def _own_nodes(fdef):
    """The nodes of a function's body, not entering nested functions or classes."""
    stack = list(fdef.body)
    while stack:
        n = stack.pop()
        yield n
        for c in ast.iter_child_nodes(n):
            if not isinstance(c, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef, ast.Lambda)):
                stack.append(c)


class _FunctionOrigins:
    def __init__(self, owner, fdef, sig):
        self.owner, self.mod, self.fdef = owner, owner.mod, fdef
        self.cls = getattr(sig, "owner", None)  # the class of a method (for super())
        self.env = {}
        if sig is not None:
            for p in sig.params:
                self.env[p.name] = of_type(p.type)
        self._solve()

    # --- the assignments ---------------------------------------------------------------
    def _records(self):
        recs = []  # (target node, a function giving the value's origin)
        for n in _own_nodes(self.fdef):
            if isinstance(n, ast.Assign):
                for t in n.targets:
                    if isinstance(t, ast.Subscript) and isinstance(t.value, ast.Name) and not isinstance(t.slice, ast.Slice):
                        # xs[i] = v, d[k] = v: the list / dict now also holds v
                        recs.append((t.value, lambda v=n.value: ("item", self.of(v)), None))
                    else:
                        recs.append((t, lambda v=n.value: self.of(v), n.value))
            elif isinstance(n, ast.AnnAssign) and n.value is not None:
                recs.append((n.target, lambda v=n.value: self.of(v), n.value))
            elif isinstance(n, ast.AugAssign):
                recs.append((n.target, lambda n=n: binop(self.of(n.target), self.of(n.value)), None))
            elif isinstance(n, ast.For):
                recs.append((n.target, lambda it=n.iter: self._element(it), None))
            elif isinstance(n, ast.NamedExpr):
                recs.append((n.target, lambda v=n.value: self.of(v), n.value))
            elif isinstance(n, (ast.With, ast.AsyncWith)):
                for it in n.items:
                    if it.optional_vars is not None:
                        recs.append((it.optional_vars, lambda: None, None))
            elif isinstance(n, ast.Expr) and isinstance(n.value, ast.Call) and isinstance(n.value.func, ast.Attribute) \
                    and isinstance(n.value.func.value, ast.Name) and n.value.func.attr in ("append", "insert", "extend") \
                    and n.value.args:
                c = n.value  # xs.append(x): the list now also holds x
                arg = c.args[-1] if c.func.attr == "insert" else c.args[0]
                ext = c.func.attr == "extend"
                recs.append((c.func.value, lambda a=arg, ext=ext: self._as_list(a, ext), None))
            elif isinstance(n, ast.Expr) and isinstance(n.value, ast.Call) and isinstance(n.value.func, ast.Attribute) \
                    and isinstance(n.value.func.value, ast.Name) and n.value.func.attr == "setdefault" \
                    and len(n.value.args) == 2:
                recs.append((n.value.func.value, lambda a=n.value.args[1]: ("item", self.of(a)), None))
            elif isinstance(n, ast.Expr) and isinstance(n.value, ast.Call) and isinstance(n.value.func, ast.Attribute) \
                    and isinstance(n.value.func.value, ast.Name) and n.value.func.attr == "update" \
                    and len(n.value.args) == 1:
                recs.append((n.value.func.value, lambda a=n.value.args[0]: self.of(a), None))
            elif isinstance(n, ast.ExceptHandler) and n.name:
                recs.append((ast.Name(id=n.name, ctx=ast.Store()), lambda: None, None))
        return recs

    def _as_list(self, arg, ext):
        o = self.of(arg)
        if ext:
            return o if isinstance(o, tuple) and o[0] == "list" or o == _BOTTOM else None
        return ("list", o)

    def _element(self, it):
        """The origin of one element of a for-loop's iterable."""
        if isinstance(it, (ast.List, ast.Tuple)) and it.elts:
            out = _BOTTOM
            for e in it.elts:
                out = join(out, self.of(e))
            return out
        if isinstance(it, ast.Call) and isinstance(it.func, ast.Name) and it.func.id in ("range", "enumerate"):
            return PY if it.func.id == "range" else None
        o = self.of(it)
        if o == NP:
            return NP
        if isinstance(o, tuple) and o[0] == "dict":
            return PY  # a dict's keys (str or int): Python's
        if isinstance(o, tuple) and o[0] == "list":
            return o[1]
        if isinstance(o, tuple) and len(set(o[1])) == 1:
            return o[1][0]
        return None if o != _BOTTOM else _BOTTOM

    def _store(self, env, target, origin, value):
        if isinstance(target, ast.Name) and isinstance(origin, tuple) and origin[0] == "item":
            old = env.get(target.id, _BOTTOM)
            if isinstance(old, tuple) and old[0] in ("list", "dict"):
                env[target.id] = (old[0], join(old[1], origin[1]))
            elif old != _BOTTOM:
                env[target.id] = None
            return
        if isinstance(target, ast.Name):
            env[target.id] = join(env.get(target.id, _BOTTOM), origin)
        elif isinstance(target, (ast.Tuple, ast.List)):
            n = len(target.elts)
            if isinstance(value, (ast.Tuple, ast.List)) and len(value.elts) == n:
                parts = [self.of(v) for v in value.elts]
            elif isinstance(origin, tuple) and len(origin[1]) == n:
                parts = list(origin[1])
            elif origin == "shape":
                parts = [PY] * n
            elif origin == _BOTTOM:
                parts = [_BOTTOM] * n
            else:
                parts = [NP if origin == NP else None] * n  # unpacking an array's rows / elements
            for t, o in zip(target.elts, parts):
                self._store(env, t, o, None)
        # attributes and subscripts change an object or array, not what a name holds

    def _solve(self):
        recs = self._records()
        names = set()
        for t, _, _ in recs:
            names |= {x.id for x in ast.walk(t) if isinstance(x, ast.Name)}
        for n in names:
            self.env.setdefault(n, _BOTTOM)
        for _ in range(len(names) * 3 + 5):  # monotone: each name only moves up, so this ends
            before = dict(self.env)
            for t, value_origin, value in recs:
                self._store(self.env, t, value_origin(), value)
            if self.env == before:
                break
        for n, o in self.env.items():
            if o == _BOTTOM:
                self.env[n] = None

    # --- one expression ----------------------------------------------------------------
    def qualify(self, node):
        if isinstance(node, ast.Name):
            if node.id in self.env:
                return None
            return self.mod.imports.get(node.id)
        if isinstance(node, ast.Attribute):
            base = self.qualify(node.value)
            return f"{base}.{node.attr}" if base else None
        return None

    def of(self, node):
        from . import library as lib
        if isinstance(node, ast.Constant):
            return PY
        if isinstance(node, (ast.JoinedStr, ast.FormattedValue)):
            return PY
        if isinstance(node, ast.Name):
            if node.id in self.env:
                return self.env[node.id]
            if node.id in self.mod.constants:
                return PY
            q = self.mod.imports.get(node.id)
            return PY if q in lib.CONSTANTS else None
        if isinstance(node, ast.BinOp):
            return binop(self.of(node.left), self.of(node.right))
        if isinstance(node, ast.UnaryOp):
            return PY if isinstance(node.op, ast.Not) else arith(self.of(node.operand))
        if isinstance(node, ast.BoolOp):
            out = _BOTTOM
            for v in node.values:
                out = join(out, self.of(v))
            return out
        if isinstance(node, ast.Compare):
            if any(isinstance(op, (ast.Is, ast.IsNot, ast.In, ast.NotIn)) for op in node.ops):
                return PY
            return arith(self.of(node.left), *(self.of(c) for c in node.comparators))
        if isinstance(node, ast.IfExp):
            return join(self.of(node.body), self.of(node.orelse))
        if isinstance(node, ast.Tuple):
            return ("tuple", tuple(self.of(e) for e in node.elts))
        if isinstance(node, ast.List):
            out = _BOTTOM
            for e in node.elts:
                out = join(out, self.of(e))
            return ("list", out)
        if isinstance(node, ast.Dict):
            out = _BOTTOM
            for v in node.values:
                out = join(out, self.of(v))
            return ("dict", out)
        if isinstance(node, (ast.ListComp, ast.DictComp)) and len(node.generators) == 1 \
                and isinstance(node.generators[0].target, ast.Name):
            g = node.generators[0]
            saved = self.env.get(g.target.id, "<unset>")
            self.env[g.target.id] = self._element(g.iter)
            try:
                if isinstance(node, ast.DictComp):
                    return ("dict", self.of(node.value))
                return ("list", self.of(node.elt))
            finally:
                if saved == "<unset>":
                    self.env.pop(g.target.id, None)
                else:
                    self.env[g.target.id] = saved
        if isinstance(node, ast.Attribute):
            q = self.qualify(node)
            if q is not None:
                return PY if q in lib.CONSTANTS else None
            if node.attr in ("shape",):
                return "shape"
            if node.attr in ("size", "ndim"):
                return PY
            owners = [c for c in self.mod.classes.values() if c.field(node.attr) is not None and not c.external]
            if owners and hasattr(self.mod, "struct_repr_problem") and isinstance(node.value, (ast.Name, ast.Attribute)):
                f = owners[0].field(node.attr)
                if isinstance(f.type, Scalar) and all(self.mod.struct_repr_problem(c) is None for c in owners):
                    return PY  # obj.x: every value the file puts in x is a Python number
            base = self.of(node.value)
            if base not in (NP, _BOTTOM) and isinstance(node.value, (ast.Name, ast.Attribute)):
                out = self._method_returns(node.attr, ("property",))
                if out is not None:
                    return out
            return NP if base == NP else (_BOTTOM if base == _BOTTOM else None)
        if isinstance(node, ast.Subscript):
            base = self.of(node.value)
            if base == "shape":
                return PY
            if base == NP:
                return NP
            if isinstance(base, tuple) and base[0] == "list":
                return base if isinstance(node.slice, ast.Slice) else base[1]
            if isinstance(base, tuple) and base[0] == "dict":
                return base[1]
            if isinstance(base, tuple) and isinstance(node.slice, ast.Constant) and isinstance(node.slice.value, int):
                k = node.slice.value
                return base[1][k] if -len(base[1]) <= k < len(base[1]) else None
            return _BOTTOM if base == _BOTTOM else None
        if isinstance(node, ast.Call):
            return self._call(node)
        return None

    def _call(self, node):
        f = node.func
        if isinstance(f, ast.Name) and f.id not in self.env:
            if f.id in self.mod.fdefs or f.id in getattr(self.mod, "variants", {}):
                return self.owner.returns(f.id)
            if f.id in self.mod.classes:
                return None
            if f.id == "range":
                return ("list", PY)  # its ints are Python's (list(range(n)), for i in range(n))
            if f.id == "zip" and node.args:  # list(zip(xs, ys)): pairs of their elements
                parts = [self.of(a) for a in node.args]
                if any(p == _BOTTOM for p in parts):
                    return _BOTTOM
                return ("list", ("tuple", tuple(p[1] if isinstance(p, tuple) and p[0] == "list" else None
                                                for p in parts)))
            if f.id == "enumerate" and node.args:
                o = self.of(node.args[0])
                if o == _BOTTOM:
                    return _BOTTOM
                return ("list", ("tuple", (PY, o[1] if isinstance(o, tuple) and o[0] == "list" else None)))
            if f.id == "dict" and len(node.args) == 1:  # dict(pairs): their second elements are the values
                o = self.of(node.args[0])
                if o == _BOTTOM:
                    return _BOTTOM
                if isinstance(o, tuple) and o[0] == "dict":
                    return o
                if isinstance(o, tuple) and o[0] == "list" and isinstance(o[1], tuple) and o[1][0] == "tuple" \
                        and len(o[1][1]) == 2:
                    return ("dict", o[1][1][1])
                return None
            if f.id in ("sum", "min", "max", "sorted", "list") and len(node.args) == 1:
                o = self.of(node.args[0])
                if isinstance(o, tuple) and o[0] == "dict":  # over its keys
                    return ("list", PY) if f.id in ("sorted", "list") else PY
                if isinstance(o, tuple) and o[0] == "list":
                    if f.id in ("sorted", "list"):
                        return o
                    return arith(o[1]) if f.id == "sum" else o[1]  # sum: 0 + x keeps x's kind
                if o == _BOTTOM:
                    return _BOTTOM
            if f.id == "round" and len(node.args) == 2 and not (
                    isinstance(node.args[1], ast.Constant) and node.args[1].value is None):
                return arith(self.of(node.args[0]))  # round(np.float64(x), 2) is a NumPy float64
            if f.id in _PY_BUILTINS:
                return PY
            if f.id == "abs" and len(node.args) == 1:
                return arith(self.of(node.args[0]))
            if f.id in ("min", "max") and len(node.args) > 1:
                out = _BOTTOM
                for a in node.args:  # returns one of its arguments
                    out = join(out, self.of(a))
                return out
        q = self.qualify(f)
        if q is not None:
            if q == "numpy.random.default_rng":
                return RNG_
            if q.startswith("math."):
                return PY  # the math module always returns Python numbers
            if q.startswith("random.") and q.count(".") == 1:
                if q.split(".")[1] in ("choice", "choices", "sample") and node.args:
                    o = self.of(node.args[0])  # an element (or elements) of the population
                    if q.endswith(".choice"):
                        return o[1] if isinstance(o, tuple) and o[0] == "list" else (None if o != _BOTTOM else _BOTTOM)
                    return o if isinstance(o, tuple) and o[0] == "list" else (None if o != _BOTTOM else _BOTTOM)
                return PY  # random(), gauss(), randint(), ...: Python's own floats and ints
            if q.startswith(("numpy.", "scipy.")):
                return NP
            return None
        if isinstance(f, ast.Attribute) and isinstance(f.value, ast.Call) and isinstance(f.value.func, ast.Name) \
                and f.value.func.id == "super" and self.cls in self.mod.classes:
            base = self.mod.classes[self.cls].base  # super().f(...): the base class's f
            m = base.find_method(f.attr) if base is not None else None
            return self.owner.returns(m.key) if m is not None and not m.abstract else None
        if isinstance(f, ast.Attribute):
            base = self.of(f.value)
            if base == _BOTTOM:
                return _BOTTOM
            if base == RNG_:
                if f.attr == "integers":
                    return NP  # an np.int64, or an int64 array
                k = _SCALAR_DRAWS.get(f.attr)
                if k is None:
                    return None
                sized = len(node.args) > k or any(a.arg == "size" and not (
                    isinstance(a.value, ast.Constant) and a.value.value is None) for a in node.keywords)
                return NP if sized else PY  # a draw without size is a Python float
            if isinstance(base, tuple) and base[0] == "dict":
                if f.attr in ("get", "pop", "setdefault"):
                    return base[1] if len(node.args) < 2 else join(base[1], self.of(node.args[1]))
                if f.attr == "copy":
                    return base
                if f.attr == "values":
                    return ("list", base[1])
                if f.attr == "keys":
                    return ("list", PY)
                if f.attr == "items":
                    return ("list", ("tuple", (PY, base[1])))
                return None
            if isinstance(base, tuple) and base[0] == "list":
                if f.attr == "pop":
                    return base[1]
                if f.attr == "copy":
                    return base
                if f.attr in ("index", "count"):
                    return PY
            if f.attr in _PY_METHODS:
                return PY
            if f.attr == "split":
                return ("list", PY)
            if base == NP:
                return NP
            out = self._method_returns(f.attr, ("instance", "static", "class"))
            if out is not None:
                return out
        return None

    def _method_returns(self, name, kinds):
        """obj.f(...): what any class's method f returns (the object may be of any class that has
        one, as with a virtual method), when they all agree."""
        keys = [c.methods[name].key for c in self.mod.classes.values()
                if name in c.methods and not c.external and c.methods[name].kind in kinds
                and not c.methods[name].abstract and c.rejected is None]
        if not keys:
            return None
        out = _BOTTOM
        for k in keys:
            r = self.owner.returns(k)
            if r is None:
                return None
            out = join(out, r)
        return None if out == _BOTTOM else out
