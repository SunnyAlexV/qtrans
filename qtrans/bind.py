"""Python bindings for a translated module (pybind11): call the C++ from Python.

    python -m qtrans bind black_scholes.py            # builds black_scholes_cpp next to it
    python -m qtrans bind black_scholes.py --check    # and checks it against the Python

    import black_scholes_cpp as bs
    bs.bs_call(100.0, 95.0, 0.5, 0.2, 0.04)

The compiled module has the translated functions and classes under their Python names, and
keeps Python's semantics at the boundary, not just its types:

* arrays are converted where the function only reads them (any array-like of numbers works);
  an array the function modifies in place must be your own float64 C-contiguous array, and the
  change is copied back into it, so you see it as in Python (anything else is refused, since a
  converted copy would hide the change);
* a Generator argument is read at its exact state and written back afterwards: your generator
  has advanced exactly as far as the Python function would have advanced it;
* your Python objects (instances of the dataclasses / classes the C++ was made from) can be
  passed where a translated class is expected: they are read field by field and, when the
  function modifies them, written back. The compiled classes are also available (Option(...)
  builds the C++ object) and are faster to pass;
* arguments that share memory, where the function modifies one of them, are refused: in Python
  the change shows through both, the compiled code works on separate copies;
* exceptions come back as the ones the Python code raised (numpy.linalg.LinAlgError,
  ValueError, ZeroDivisionError, ...). Shape and bounds errors raise ShapeError, which is both
  a ValueError and an IndexError (NumPy raises one or the other; Eigen's checks can't tell).

Where this is not exactly Python (each is documented in README.md): a function that returns
its argument returns a copy, not the same array; reading an array field of a compiled object
gives a read-only copy; storing an array into a compiled object's field stores a copy; a
generator field reads as a copy; a generator passed to a compiled class's constructor is
copied, so the object and you no longer share it.

`check_bindings` runs every translated function, method and constructor through the compiled
module and through Python on the same random and hand-picked inputs (QTRANS_CASES), with your
Python objects and with compiled ones, and compares results, in-place changes, generator
states and exceptions; with bench=True it times both end to end, conversions included.
"""
from __future__ import annotations

import ast
import copy
import importlib.util
import inspect
import math
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
import warnings
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from . import verify as V
from .translator import ARRAYS_HEADER, LIST_HEADER, UNICODE_HEADER, PYTHON_HEADER, PYRANDOM_HEADER, PRINT_HEADER, RANDOM_HEADER, RUNTIME_HEADER, VERSION, TranslationResult, translate_file
from .types import (ListType, DictType, BOOL, FLOAT, INT, IVEC, MAT, RNG, STR, VEC, ArrayType, IntArrayType, OptionalType, Scalar, StructType, TupleType,
                    VoidType)

PYBIND_HEADER = Path(__file__).parent / "runtime" / "qtrans_pybind.hpp"


class BindError(Exception):
    """Something the bindings can't express; the function or class is left out, with this note."""


# --- what the C++ wrapper does with each kind of value ------------------------------------------
def _desc(t) -> str:
    if isinstance(t, OptionalType):
        return _desc(t.inner) + " or None"
    if isinstance(t, StructType):
        return _article(t.name)
    return {FLOAT: "a float", INT: "an int", BOOL: "a bool", STR: "a str", VEC: "a 1-D float array",
            MAT: "a 2-D float array", RNG: "a numpy Generator", IVEC: "a 1-D int64 array"}.get(t, str(t))


def _article(name: str) -> str:
    return ("an " if name[:1].lower() in "aeiou" else "a ") + name


def _is_leafy(t) -> bool:
    """Can a value of type t hold memory the function could share with another argument?"""
    inner = t.inner if isinstance(t, OptionalType) else t
    return isinstance(inner, (ArrayType, StructType, IntArrayType)) or inner == RNG


def _native(t) -> bool:
    """Passed straight through pybind11's own conversions."""
    inner = t.inner if isinstance(t, OptionalType) else t
    # a list is copied in (pybind11's list caster): a function never changes a list passed to it
    return isinstance(inner, Scalar) or inner == STR or isinstance(inner, (ListType, DictType))


def _cstr(s: str) -> str:
    return '"' + s.replace("\\", "\\\\").replace('"', '\\"') + '"'


def _doc(text) -> str:
    if not text:
        return ""
    text = inspect.cleandoc(text)
    delim = "qtdoc"
    while f"){delim}\"" in text:
        delim += "_"
    return f'R"{delim}({text}){delim}"'


@dataclass
class _Param:
    name: str           # Python name (keyword)
    cpp: str            # local variable in the wrapper
    type: object
    mode: str           # cref / value / ref
    default_code: str | None


class _Wrap:
    """The pieces of one wrapper lambda."""

    def __init__(self):
        self.decls, self.pre, self.args, self.post, self.pyargs, self.leaves = [], [], [], [], [], []
        self.notes = []


class Binder:
    def __init__(self, result: TranslationResult, source: str, project=None):
        """project: the ProjectResult when the module is part of one; then the classes it uses
        from other modules are known here too (converted, but bound by their own module), and
        the analysis of what a call can run follows calls into the other modules."""
        self.result = result
        self.ns = result.namespace or result.header.split("namespace ", 1)[1].split(" ", 1)[0]
        self.classes = dict(result.classes or {})
        self.own = set(self.classes)
        self.project = project
        self.tree = ast.parse(source)
        self.extra_trees = []
        if project is not None and len(project.modules) > 1:
            t = project.translators[result.module]
            for name, info in t.classes.items():  # imported classes, as this module sees them
                if info.external and info.rejected is None and name in project.classes():
                    self.classes[name] = info
            order = [n for n in project.classes() if n in self.classes]  # dependency order, project-wide
            self.classes = {n: self.classes[n] for n in order}
            self.extra_trees = [project.translators[m].tree for m in project.order if m != result.module]
            self.imported_fdefs = {local: project.translators[owner].fdefs.get(oname)
                                   for local, (owner, oname, kind) in t.imported.items() if kind == "function"}
        else:
            self.imported_fdefs = {}
        self.notes = {}       # name -> [why it is left out / how it differs]
        self.bound = []       # Python names of the functions bound
        self.bound_classes = []
        self._index_functions()
        self.variants = {frozenset(): 0}  # rebind-only field sets -> write-back variant number
        self.variant = 0                  # the one for the callable being generated

    # --- which object fields Python only ever re-points (never changes in place) -----------------
    def _index_functions(self):
        self.top = {n.name: n for n in self.tree.body if isinstance(n, ast.FunctionDef)}
        self.top.update({k: v for k, v in self.imported_fdefs.items() if v is not None and k not in self.top})
        self.methods = {}   # (class, name) -> FunctionDef
        self.by_name = {}   # method name -> [FunctionDef]
        self.cls_defs = {}
        for n in [n for t in [self.tree] + self.extra_trees for n in t.body]:
            if isinstance(n, ast.ClassDef):
                self.cls_defs[n.name] = n
                for m in n.body:
                    if isinstance(m, ast.FunctionDef):
                        self.methods[(n.name, m.name)] = m
                        self.by_name.setdefault(m.name, []).append(m)

    def _reach(self, start) -> list:
        """The functions a call may run: start, and everything it calls, transitively (a method
        by name, in any class: a doubt counts as reached)."""
        seen, todo = [], [start]
        while todo:
            f = todo.pop()
            if f is None or any(f is g for g in seen):
                continue
            seen.append(f)
            for n in ast.walk(f):
                if not isinstance(n, ast.Call):
                    continue
                if isinstance(n.func, ast.Name):
                    todo.append(self.top.get(n.func.id))
                    if n.func.id in self.cls_defs:
                        todo += [self.methods.get((n.func.id, k)) for k in ("__init__", "__post_init__")]
                elif isinstance(n.func, ast.Attribute):
                    todo += self.by_name.get(n.func.attr, [])
        return seen

    def _rebind_only(self, fdefs) -> frozenset:
        """Field names that these functions only ever assign anew, never change in place: they are
        written back by re-pointing the attribute, as Python does. Any other array or generator
        field is written back into the object the caller's object already holds. Fields are
        matched by name, so a doubt counts as 'changed in place'."""
        rebinds, inplace = set(), set()

        def attr_of(n):
            while isinstance(n, ast.Subscript):
                n = n.value
            return n.attr if isinstance(n, ast.Attribute) else None

        def construction(fdef, target):
            return fdef.name in ("__init__", "__post_init__") and isinstance(target, ast.Attribute) and \
                isinstance(target.value, ast.Name) and target.value.id == "self"

        for fdef in fdefs:
            for node in ast.walk(fdef):
                if isinstance(node, (ast.Assign, ast.AnnAssign)):
                    targets = node.targets if isinstance(node, ast.Assign) else [node.target]
                    flat = []
                    for t in targets:
                        flat += t.elts if isinstance(t, (ast.Tuple, ast.List)) else [t]
                    for t in flat:
                        if isinstance(t, ast.Attribute) and not construction(fdef, t):
                            rebinds.add(t.attr)
                        elif isinstance(t, ast.Subscript) and attr_of(t):
                            inplace.add(attr_of(t))
                    if node.value is not None and isinstance(node.value, (ast.Attribute, ast.Subscript)) \
                            and attr_of(node.value):
                        inplace.add(attr_of(node.value))  # w = self.w (or a view of it): an alias
                elif isinstance(node, ast.AugAssign) and attr_of(node.target):
                    inplace.add(attr_of(node.target))
                elif isinstance(node, ast.Call):
                    f = node.func
                    if isinstance(f, ast.Attribute) and f.attr == "__setattr__" and len(node.args) >= 2 \
                            and isinstance(node.args[1], ast.Constant) and fdef.name not in ("__init__", "__post_init__"):
                        rebinds.add(node.args[1].value)
                    if isinstance(f, ast.Attribute) and isinstance(f.value, (ast.Attribute, ast.Subscript)) \
                            and attr_of(f.value):
                        inplace.add(attr_of(f.value))  # self.w.sort(), self.w.fill(0)
                    for a in list(node.args) + [k.value for k in node.keywords]:
                        if isinstance(a, (ast.Attribute, ast.Subscript)) and attr_of(a):
                            inplace.add(attr_of(a))  # passed on: the callee may change it
        return frozenset(rebinds - inplace)

    def _use_variant(self, start):
        """Pick the write-back variant for a callable starting at FunctionDef start."""
        relevant = {f.name for i in self.classes.values() for f in i.fields
                    if isinstance(f.type.inner if isinstance(f.type, OptionalType) else f.type, ArrayType)
                    or f.type == RNG}
        key = self._rebind_only(self._reach(start)) & relevant if start is not None else frozenset()
        self.variant = self.variants.setdefault(frozenset(key), len(self.variants))

    # --- parameters -------------------------------------------------------------------------------
    def _param(self, w: _Wrap, p: _Param):
        t, v = p.type, p.cpp
        if isinstance(t, ListType) and isinstance(t.elem, StructType):
            raise BindError(f"parameter '{p.name}' is a list of {t.elem} objects, which the bindings don't support yet")
        if isinstance(t, OptionalType) and not (_native(t) or isinstance(t.inner, ArrayType)):
            raise BindError(f"parameter '{p.name}' of type {t} can't be bound yet")
        pyarg = f"py::arg({_cstr(p.name)})"
        if _native(t):
            w.decls.append(f"{t.cpp()} {v}")
            w.args.append(f"std::move({v})" if t == STR and p.mode == "value" else v)
            if p.default_code is not None:
                pyarg += " = py::none()" if p.default_code == "std::nullopt" else f" = {p.default_code}"
            w.pyargs.append(pyarg)
            return
        o = f"{v}_obj"
        w.decls.append(f"py::object {o}")
        if p.default_code is not None:
            if p.default_code == "std::nullopt":
                pyarg += " = py::none()"
            elif isinstance(t, StructType):
                pyarg += f" = py::cast({p.default_code})"
            else:
                w.notes.append(f"'{p.name}' has a default in Python; here it must be given")
        w.pyargs.append(pyarg)
        modified = p.mode == "ref"
        if _is_leafy(t):
            w.leaves.append((o, modified, p.name))
        if isinstance(t, OptionalType):  # an optional array: read, never modified in place
            w.pre.append(f"std::optional<{t.inner.cpp()}> {v} = qtb::optional_array<{t.inner.cpp()}>({o}, "
                         f"{_cstr(p.name)}, {_cstr(_desc(t.inner))});")
            w.args.append(f"std::move({v})" if p.mode == "value" else v)
        elif isinstance(t, ArrayType):
            if modified:
                w.pre.append(f"py::array {v}_arr = qtb::own_array({o}, {_cstr(p.name)}, {t.rank});")
                w.pre.append(f"{t.cpp()} {v} = qtb::array_in<{t.cpp()}>({v}_arr);")
                w.post.append(f"qtb::array_back({v}_arr, {v});")
                w.args.append(v)
            else:
                w.pre.append(f"{t.cpp()} {v} = qtb::array_arg<{t.cpp()}>({o}, {_cstr(p.name)}, {_cstr(_desc(t))});")
                w.args.append(f"std::move({v})" if p.mode == "value" else v)
        elif isinstance(t, IntArrayType):
            if modified:
                w.pre.append(f"py::array {v}_arr = qtb::own_ivec({o}, {_cstr(p.name)});")
                w.pre.append(f"qt::IVec {v} = qtb::ivec_arg({v}_arr, {_cstr(p.name)});")
                w.post.append(f"qtb::ivec_back({v}_arr, {v});")
                w.args.append(v)
            else:
                w.pre.append(f"qt::IVec {v} = qtb::ivec_arg({o}, {_cstr(p.name)});")
                w.args.append(f"std::move({v})" if p.mode == "value" else v)
        elif t == RNG:
            w.pre.append(f"qt::Rng {v} = qtb::rng_in({o}, {_cstr(p.name)});")
            w.post.append(f"qtb::rng_back({o}, {v});")
            w.args.append(v)
        elif isinstance(t, StructType):
            info = self.classes.get(t.name)
            if info is None:
                raise BindError(f"uses class {t.name}, which isn't bound" if t.ref else
                                f"uses class {t.name}, which isn't translated")
            cpp = info.cpp
            w.pre.append(f"std::optional<{cpp}> {v}_tmp;")
            decl = f"const {cpp}&" if p.mode == "cref" else f"{cpp}&" if p.mode == "ref" else cpp
            w.pre.append(f"{decl} {v} = qtb::obtain<{cpp}>({o}, {v}_tmp, [](py::handle h) {{ return "
                         f"qtrans_from_py_{info.name}(h, {_cstr(p.name)}); }});")
            if modified:
                w.post.append(f"if ({v}_tmp) qtrans_to_py_{info.name}_{self.variant}({o}, *{v}_tmp);")
            w.args.append(f"std::move({v})" if p.mode == "value" else v)
        else:
            raise BindError(f"parameter '{p.name}' of type {t} can't be bound yet")

    def _params(self, sig):
        out, used = [], set()
        for p in sig.params:
            v = "a_" + p.cpp
            while v in used:
                v += "_"
            used.add(v)
            out.append(_Param(p.name, v, p.type, p.mode, p.default_code))
        return out

    def _ret_ok(self, t):
        if isinstance(t, TupleType):
            return all(self._ret_ok(e) for e in t.elts)
        inner = t.inner if isinstance(t, OptionalType) else t
        if inner == RNG:
            raise BindError("returning a generator can't be bound yet")
        if isinstance(inner, ListType) and isinstance(inner.elem, StructType):
            raise BindError(f"returns a list of {inner.elem} objects, which the bindings don't support yet")
        if isinstance(inner, StructType) and inner.name not in self.classes:
            raise BindError(f"returns a {inner.name}, which isn't " + ("bound" if inner.ref else "translated"))

    def _lambda(self, w: _Wrap, call: str, self_decl: str | None = None, self_leaf=None) -> str:
        """`[](decls) { pre; return call; }`, with the write-backs run afterwards (also when the
        call raises) and the shared-memory check first."""
        decls = ([self_decl] if self_decl else []) + w.decls
        body = []
        leaves = list(w.leaves)
        n_leaves = len(leaves) + (1 if self_leaf else 0)
        if n_leaves >= 2 and (any(m for _, m, _ in leaves) or (self_leaf and self_leaf[1])):
            body.append("std::vector<qtb::Leaf> qtrans_leaves;")
            if self_leaf:
                cpp, modified = self_leaf
                body.append(f"qtrans_leaves.push_back({{py::handle(), 2, &self, sizeof({cpp}), "
                            f"{'true' if modified else 'false'}, \"self\"}});")
            for o, modified, name in leaves:
                body.append(f"qtb::collect({o}, {'true' if modified else 'false'}, {_cstr(name)}, "
                            f"&qtrans_span, qtrans_leaves);")
            body.append("qtb::no_shared_memory(qtrans_leaves);")
        body += w.pre
        if w.post:
            body.append(f"return qtb::call_then([&] {{ return {call}; }}, [&] {{ {' '.join(w.post)} }});")
        else:
            body.append(f"return {call};")
        inner = "\n".join("        " + b for b in body)
        return f"[]({', '.join(decls)}) {{\n{inner}\n    }}"

    def _callable(self, sig, call_of, self_decl=None, self_leaf=None, params=None):
        """-> (lambda code, py::arg list) for a function / method / constructor."""
        w = _Wrap()
        for p in (params if params is not None else self._params(sig)):
            self._param(w, p)
        call = call_of(w.args)
        return self._lambda(w, call, self_decl, self_leaf), w.pyargs, w.notes

    # --- functions ---------------------------------------------------------------------------------
    def bind_function(self, name, sig) -> list:
        if getattr(sig, "py_random", False):
            raise BindError("it uses the random module's generator: the compiled module's would be a separate one "
                            "from your Python's, so the numbers wouldn't continue yours (qtrans verify checks it)")
        self._ret_ok(sig.ret)
        lam, pyargs, notes = self._callable(sig, lambda a: f"{sig.cpp}({', '.join(a)})")
        if notes:
            self.notes.setdefault(name, []).extend(notes)
        extra = "".join(", " + a for a in pyargs)
        if name in (self.result.prints or ()):
            # what it prints goes to Python's sys.stdout / sys.stderr (a notebook shows it), in
            # order with Python's own output, rather than straight to the process's stdout
            extra += ", py::call_guard<py::scoped_ostream_redirect, py::scoped_estream_redirect>()"
        doc = _doc(sig.doc)
        pyname = sig.name if getattr(sig, "overloaded", False) else name  # versions: one Python name, overloads
        return [f"    m.def({_cstr(pyname)}, {lam}{extra}{', ' + doc if doc else ''});"]

    # --- classes -----------------------------------------------------------------------------------
    def _mutable(self, info) -> bool:
        return not (info.frozen or info.namedtuple)

    def _immutable_value(self, info) -> bool:
        return not self._mutable(info) and not self._holds_mutable(info)

    def _holds_mutable(self, info, seen=None) -> bool:
        seen = set() if seen is None else seen
        seen.add(info.name)
        for f in info.fields:
            t = f.type.inner if isinstance(f.type, OptionalType) else f.type
            if isinstance(t, ArrayType) or t == RNG:
                return True
            if isinstance(t, StructType) and t.name not in seen:
                sub = self.classes[t.name]
                if self._mutable(sub) or self._holds_mutable(sub, seen):
                    return True
        return False

    def from_python(self, info) -> list:
        """C++ object from your Python object, field by field (no constructor runs: the fields
        are the object's state)."""
        if info.ref:
            raise BindError(f"{info.name} is part of a class hierarchy, which the bindings don't support yet "
                            "(qtrans verify checks the C++ of it)")
        cpp = info.cpp
        out = [f"static {cpp} qtrans_from_py_{info.name}(py::handle o, const char* arg) {{",
               f"    qtb::expect_class(o, {_cstr(info.name)}, {_cstr(_article(info.name))}, arg);",
               f"    {cpp} v;"]
        for f in info.fields:
            t, nm = f.type, _cstr(f.name)
            inner = t.inner if isinstance(t, OptionalType) else t
            if isinstance(t, OptionalType) and not (_native(t) or isinstance(inner, ArrayType)):
                raise BindError(f"field {f.name} of type {t} can't be bound yet")
            if _native(t) or isinstance(inner, ArrayType):
                out.append(f"    v.{f.cpp} = qtb::field<{t.cpp()}>(o, {_cstr(info.name)}, {nm}, {_cstr(_desc(t))});")
            elif t == RNG:
                out.append(f"    v.{f.cpp} = qtb::rng_in(qtb::field<py::object>(o, {_cstr(info.name)}, {nm}, "
                           f"\"a numpy Generator\"), {nm});")
            elif isinstance(t, StructType):
                sc = self.classes[t.name].cpp
                out.append(f"    {{ std::optional<{sc}> tmp; v.{f.cpp} = qtb::obtain<{sc}>(qtb::field<py::object>(o, "
                           f"{_cstr(info.name)}, {nm}, \"an object\"), tmp, [](py::handle h) {{ return "
                           f"qtrans_from_py_{t.name}(h, {nm}); }}); }}")
            else:
                raise BindError(f"field {f.name} of type {t} can't be bound yet")
        return out + ["    return v;", "}", ""]

    def to_python(self, info, k, rebind) -> list:
        """Write back into your Python object what a call may have changed. Arrays and
        generators go into the objects it already holds (anything else holding them sees the
        change, as when Python changes them in place), except fields in `rebind`, which the
        functions this call can run only ever assign anew: those are re-pointed, as in Python."""
        cpp = info.cpp
        mutable = self._mutable(info)
        out = [f"static void qtrans_to_py_{info.name}_{k}(py::handle o, const {cpp}& v) {{"]
        for f in info.fields:
            t, nm = f.type, _cstr(f.name)
            inner = t.inner if isinstance(t, OptionalType) else t
            re = "true" if f.name in rebind and mutable else "false"
            if isinstance(inner, ArrayType):
                out.append(f"    qtb::array_field_back(o, {nm}, v.{f.cpp}, {re});")
            elif _native(t):
                if mutable:  # (a frozen object's numbers can't have changed)
                    out.append(f"    qtb::set_attr(o, {nm}, py::cast(v.{f.cpp}));")
            elif t == RNG:
                out.append(f"    qtb::rng_field_back(o, {nm}, v.{f.cpp}, {re});")
            elif isinstance(t, StructType):
                sub = self.classes[t.name]
                sc = sub.cpp
                if self._immutable_value(sub):
                    if mutable:  # nothing inside it can change, but it may have been replaced
                        out.append(f"    qtb::set_attr(o, {nm}, py::cast(v.{f.cpp}));")
                else:
                    out.append(f"    {{ py::object c = o.attr({nm}); if (py::isinstance<{sc}>(c)) c.cast<{sc}&>() = "
                               f"v.{f.cpp}; else qtrans_to_py_{t.name}_{k}(c, v.{f.cpp}); }}")
        if len(out) == 1:
            out.append("    (void)o; (void)v;")
        return out + ["}", ""]

    def bind_class(self, info) -> list:
        cpp, sigs = info.cpp, self.result.signatures
        var = f"cls_{cpp}"
        lines = [f"    py::class_<{cpp}> {var}(m, {_cstr(info.name)}, py::module_local(){', ' + _doc(info.doc) if info.doc else ''});"]
        # constructor
        self._use_variant(self.methods.get((info.name, "__init__" if info.plain else "__post_init__")))
        if info.plain:
            sig = sigs[info.methods["__init__"].key]
            lam, pyargs, notes = self._callable(sig, lambda a: f"{cpp}({', '.join(a)})")
        else:
            params = [_Param(f.name, "a_" + f.cpp, f.type, "value", f.default_code) for f in info.init_fields]
            lam, pyargs, notes = self._callable(None, lambda a: f"{cpp}({', '.join(a)})", params=params)
            if any(f.type == RNG for f in info.init_fields):
                notes.append("a generator given to the constructor is copied: the object and you no longer share it")
        if notes:
            self.notes.setdefault(info.name, []).extend(notes)
        init = info.methods.get("__init__" if info.plain else "__post_init__")
        if init is not None and getattr(sigs.get(init.key), "py_random", False):
            raise BindError(f"creating a {info.name} uses the random module's generator, which the compiled module "
                            "would have a separate copy of")
        lines.append(f"    {var}.def(py::init({lam}){''.join(', ' + a for a in pyargs)});")
        # fields
        mutable = self._mutable(info)
        for f in info.fields:
            t, nm = f.type, _cstr(f.name)
            inner = t.inner if isinstance(t, OptionalType) else t
            if isinstance(inner, ArrayType):
                get = f"[](const {cpp}& s) {{ return qtb::readonly(s.{f.cpp}); }}"
                setter = (f"[]({cpp}& s, py::object x) {{ s.{f.cpp} = "
                          + (f"qtb::optional_array<{inner.cpp()}>" if isinstance(t, OptionalType) else f"qtb::array_arg<{t.cpp()}>")
                          + f"(x, {nm}, {_cstr(_desc(inner))}); }}")
            elif t == RNG:
                get, setter = f"[](const {cpp}& s) {{ return qtb::rng_out(s.{f.cpp}); }}", None
            elif isinstance(t, StructType):
                sc = self.classes[t.name].cpp
                get = (f"py::cpp_function([]({cpp}& s) -> {sc}& {{ return s.{f.cpp}; }}, "
                       "py::return_value_policy::reference_internal)")
                setter = (f"[]({cpp}& s, py::object x) {{ std::optional<{sc}> tmp; s.{f.cpp} = qtb::obtain<{sc}>(x, tmp, "
                          f"[](py::handle h) {{ return qtrans_from_py_{t.name}(h, {nm}); }}); }}")
            else:
                get = f"[](const {cpp}& s) {{ return s.{f.cpp}; }}"
                setter = f"[]({cpp}& s, {t.cpp()} x) {{ s.{f.cpp} = std::move(x); }}"
            if mutable and setter:
                lines.append(f"    {var}.def_property({nm}, {get}, {setter});")
            else:
                lines.append(f"    {var}.def_property_readonly({nm}, {get});")
        # methods
        for m in info.methods.values():
            if m.kind in ("post_init", "init") or m.key not in sigs:
                continue
            sig = sigs[m.key]
            try:
                self._use_variant(self.methods.get((info.name, m.name)))
                if getattr(sig, "py_random", False):
                    raise BindError("it uses the random module's generator, which the compiled module would have "
                                    "a separate copy of")
                self._ret_ok(sig.ret)
                doc = _doc(sig.doc)
                if m.kind == "property":
                    lines.append(f"    {var}.def_property_readonly({_cstr(m.name)}, [](const {cpp}& self) "
                                 f"{{ return self.{sig.cpp}(); }}{', ' + doc if doc else ''});")
                    continue
                if m.kind in ("static", "class"):
                    lam, pyargs, notes = self._callable(sig, lambda a, s=sig: f"{cpp}::{s.cpp}({', '.join(a)})")
                    how = "def_static"
                else:
                    const = sig.const
                    lam, pyargs, notes = self._callable(
                        sig, lambda a, s=sig: f"self.{s.cpp}({', '.join(a)})",
                        self_decl=f"{'const ' if const else ''}{cpp}& self", self_leaf=(cpp, not const))
                    how = "def"
                if notes:
                    self.notes.setdefault(f"{info.name}.{m.name}", []).extend(notes)
                lines.append(f"    {var}.{how}({_cstr(m.name)}, {lam}{''.join(', ' + a for a in pyargs)}"
                             f"{', ' + doc if doc else ''});")
            except BindError as e:
                self.notes.setdefault(f"{info.name}.{m.name}", []).append(f"left out: {e}")
        # dunders: equality, hash, repr, copies, and a NamedTuple's tuple protocol
        cmp = [f for f in info.fields if f.compare]
        fields_tuple = lambda o: f"py::make_tuple({', '.join(f'{o}.attr({_cstr(f.name)})' for f in cmp)})"  # noqa: E731
        if info.eq and not info.plain:
            if self._has_operator_eq(info):
                lines.append(f"    {var}.def(\"__eq__\", [](const {cpp}& a, py::object b) -> py::object {{ "
                             f"if (!py::isinstance<{cpp}>(b)) return py::reinterpret_borrow<py::object>(Py_NotImplemented); "
                             f"return py::bool_(a == b.cast<const {cpp}&>()); }}, py::is_operator());")
                lines.append(f"    {var}.def(\"__ne__\", [](const {cpp}& a, py::object b) -> py::object {{ "
                             f"if (!py::isinstance<{cpp}>(b)) return py::reinterpret_borrow<py::object>(Py_NotImplemented); "
                             f"return py::bool_(a != b.cast<const {cpp}&>()); }}, py::is_operator());")
            else:  # holds an array: compare the field tuples the way Python's dataclass == does
                for op, sym in (("__eq__", "Py_EQ"), ("__ne__", "Py_NE")):
                    lines.append(f"    {var}.def(\"{op}\", [](py::object a, py::object b) -> py::object {{ "
                                 f"if (!py::isinstance<{cpp}>(b)) return py::reinterpret_borrow<py::object>(Py_NotImplemented); "
                                 f"if (a.is(b)) return py::bool_({'true' if op == '__eq__' else 'false'}); "
                                 f"py::object r = py::reinterpret_steal<py::object>(PyObject_RichCompare("
                                 f"{fields_tuple('a')}.ptr(), {fields_tuple('b')}.ptr(), {sym})); "
                                 f"if (!r) throw py::error_already_set(); return r; }}, py::is_operator());")
            if info.frozen or info.namedtuple:
                lines.append(f"    {var}.def(\"__hash__\", [](py::object a) {{ return py::hash({fields_tuple('a')}); }});")
        if not info.plain:
            reprs = " + \", \" + ".join(f'std::string("{f.name}=") + py::repr(o.attr({_cstr(f.name)})).cast<std::string>()'
                                        for f in info.fields) or '""'
            lines.append(f"    {var}.def(\"__repr__\", [](py::object o) {{ return std::string({_cstr(info.name + '(')}) + "
                         f"{reprs} + \")\"; }});")
        lines.append(f"    {var}.def(\"__copy__\", [](const {cpp}& s) {{ return {cpp}(s); }});")
        lines.append(f"    {var}.def(\"__deepcopy__\", [](const {cpp}& s, py::dict) {{ return {cpp}(s); }}, py::arg(\"memo\"));")
        lines.append(f"    {var}.def_static(\"_from_python\", [](py::object o) {{ return qtrans_from_py_{info.name}(o, \"o\"); }}, "
                     f"\"The compiled copy of one of your Python {info.name} objects (no constructor runs).\");")
        if info.namedtuple:
            tup = f"py::make_tuple({', '.join(f'o.attr({_cstr(f.name)})' for f in info.fields)})"
            lines.append(f"    {var}.def(\"__len__\", [](const {cpp}&) {{ return {len(info.fields)}; }});")
            lines.append(f"    {var}.def(\"__iter__\", [](py::object o) {{ return {tup}.attr(\"__iter__\")(); }});")
            lines.append(f"    {var}.def(\"__getitem__\", [](py::object o, py::object i) {{ py::tuple t = {tup}; return py::object(t[i]); }});")
            lines.append(f"    {var}.attr(\"_fields\") = py::make_tuple({', '.join(_cstr(f.name) for f in info.fields)});")
        return lines

    def _has_operator_eq(self, info) -> bool:
        return info.eq and not info.has_array_field(self.classes) and any(f.compare for f in info.fields)

    # --- the whole file --------------------------------------------------------------------------
    def source(self, header_name: str, module_name: str) -> str:
        """The whole .cpp of a one-module extension."""
        res = self.result
        block = self.block()
        out = [f"// {module_name}_bind.cpp: Python bindings generated by qtrans {VERSION} for {header_name}.",
               "// Build with `python -m qtrans bind`; see qtrans/bind.py for what the bindings guarantee.",
               f'#include "{header_name}"']
        if res.uses_random:
            out.append("#define QTRANS_BIND_RNG 1")
        out += ['#include "qtrans_pybind.hpp"', "", "namespace py = pybind11;", ""] + block
        doc = _doc(f"{res.module}, compiled by qtrans {VERSION}: the translated functions and classes of "
                   f"{res.module}.py.")
        out += [f"PYBIND11_MODULE({module_name}, m) {{",
                f"    m.doc() = {doc};",
                "    qtb::register_exceptions(m);",
                f"    {self.ns}::qtrans_binding::bind(m);",
                f"    m.attr(\"__qtrans_version__\") = {_cstr(VERSION)};",
                "}", ""]
        return "\n".join(out)

    def block(self) -> list:
        """`namespace <module> { namespace qtrans_binding { ... void bind(module) } }`: the
        conversions for every class the module uses, and the bindings of its own classes and
        functions."""
        res = self.result
        order = list(self.classes)  # already in dependency order (class_order)
        ok_classes = []
        conv = {}
        for n in order:
            info = self.classes[n]
            try:
                conv[n] = self.from_python(info)
                ok_classes.append(info)
            except BindError as e:
                self.notes.setdefault(n, []).append(f"left out: {e}")
        good = {i.name for i in ok_classes}
        if len(good) != len(order):  # a class that needs a left-out one is left out too
            changed = True
            while changed:
                changed = False
                for info in list(ok_classes):
                    deps = {f.type.name for f in info.fields if isinstance(f.type, StructType)}
                    if not deps <= good:
                        ok_classes.remove(info)
                        good.discard(info.name)
                        self.notes.setdefault(info.name, []).append("left out: holds a class that is left out")
                        changed = True
        self.classes = {i.name: i for i in ok_classes}

        out = [f"namespace {self.ns} {{", "namespace qtrans_binding {", ""]
        body = []
        for info in ok_classes:
            if info.name not in self.own:
                continue  # bound by the module that defines it
            try:
                body += self.bind_class(info)
                self.bound_classes.append(info.name)
            except BindError as e:
                self.notes.setdefault(info.name, []).append(f"left out: {e}")
        def rank(item):  # versions of one function: bool, then int, then float arguments, so that
            _, sig = item  # pybind11 (trying overloads in order) gives a Python int to the int version
            order = {"bool": 0, "int": 1, "float": 2}
            return (sig.name, tuple(order.get(getattr(p.type, "name", ""), 3) for p in sig.params)) \
                if getattr(sig, "overloaded", False) else ("", ())
        items = list(res.signatures.items())
        versions = sorted((it for it in items if getattr(it[1], "overloaded", False)), key=rank)
        items = [it for it in items if not getattr(it[1], "overloaded", False)] + versions
        for name, sig in items:
            if sig.owner:
                continue
            try:
                self._use_variant(self.top.get(name))
                body += self.bind_function(name, sig)
                self.bound.append(name)
            except BindError as e:
                self.notes.setdefault(name, []).append(f"left out: {e}")

        for info in ok_classes:
            out.append(f"[[maybe_unused]] static {info.cpp} qtrans_from_py_{info.name}(py::handle o, const char* arg);")
            for k in self.variants.values():
                out.append(f"[[maybe_unused]] static void qtrans_to_py_{info.name}_{k}(py::handle o, const {info.cpp}& v);")
        out.append("")
        out.append("// Where a compiled object sits in memory, for the shared-memory check.")
        out.append("[[maybe_unused]] static bool qtrans_span(py::handle o, const void*& p, std::size_t& n) {")
        for info in ok_classes:
            out.append(f"    if (py::isinstance<{info.cpp}>(o)) {{ p = &o.cast<const {info.cpp}&>(); "
                       f"n = sizeof({info.cpp}); return true; }}")
        out += ["    (void)o; (void)p; (void)n;", "    return false;", "}", ""]
        for info in ok_classes:
            out.append(f"// {info.name}: from your Python object, and back after a call that changed it")
            out += conv[info.name]
            for key, k in self.variants.items():
                out += self.to_python(info, k, key)
        out.append("[[maybe_unused]] static void bind(py::module_& m) {")
        exc = getattr(res, "exceptions", {}) or {}
        for name, (base, doc) in exc.items():  # your exception classes, as Python classes of this module
            if base in exc:
                base_obj = f"m.attr({_cstr(base)})"
            elif base == "LinAlgError":
                base_obj = 'py::module_::import("numpy.linalg").attr("LinAlgError")'
            else:
                base_obj = f'py::module_::import("builtins").attr({_cstr(base)})'
            doc_code = _doc(doc) or '""'
            out += [f"    m.attr({_cstr(name)}) = qtb::exception_class(m, {_cstr(name)}, {base_obj}, {doc_code});"]
        out += body or (["    (void)m;"] if not exc else [])
        out += ["}", "", "}  // namespace qtrans_binding", f"}}  // namespace {self.ns}", ""]
        return out


def binding_source(result: TranslationResult, source: str, module_name: str | None = None):
    """-> (C++ source of the pybind11 module, Binder with what was bound and notes)."""
    b = Binder(result, source)
    name = module_name or f"{result.module}_cpp"
    return b.source(f"{result.module}.hpp", name), b


def project_binding_source(project, module_name: str):
    """One extension for a whole project: a submodule per module (`<name>.bl_core`), each with
    its functions and classes, and the names it imports from the others, as in Python. When one
    module was asked for, its names are also at the top level, so `import <name> as bl_core`
    works as a drop-in. One extension, so a class used by several modules is one Python type.
    -> (C++ source, {module: Binder})"""
    binders = {m: Binder(project.modules[m], Path(project.paths[m]).read_text(encoding="utf-8-sig"), project)
               for m in project.order}
    blocks = {m: b.block() for m, b in binders.items()}
    out = [f"// {module_name}_bind.cpp: Python bindings generated by qtrans {VERSION} for a project of "
           f"{len(project.modules)} modules.",
           "// Build with `python -m qtrans bind`; see qtrans/bind.py for what the bindings guarantee."]
    out += [f'#include "{m}.hpp"' for m in project.order]
    if project.uses_random:
        out.append("#define QTRANS_BIND_RNG 1")
    out += ['#include "qtrans_pybind.hpp"', "", "namespace py = pybind11;", ""]
    for m in project.order:
        out += blocks[m]
    doc = _doc(f"compiled by qtrans {VERSION}: the translated functions and classes of "
               + ", ".join(f"{m}.py" for m in project.order) + ", one submodule each.")
    out += [f"PYBIND11_MODULE({module_name}, m) {{",
            f"    m.doc() = {doc};",
            "    qtb::register_exceptions(m);",
            "    py::dict sysmods = py::module_::import(\"sys\").attr(\"modules\");",
            "    py::dict submodules;"]
    for mod in project.order:  # in import order: a module's classes exist before others use them
        ns = project.modules[mod].namespace
        out += [f"    py::module_ sub_{ns} = m.def_submodule({_cstr(mod)});",
                f"    {ns}::qtrans_binding::bind(sub_{ns});",
                f"    sub_{ns}.attr(\"ShapeError\") = m.attr(\"ShapeError\");",
                f"    sysmods[{_cstr(module_name + '.' + mod)}] = sub_{ns};  // import {module_name}.{mod} works",
                f"    submodules[{_cstr(mod)}] = sub_{ns};"]
    for mod in project.order:  # names imported from other modules: the same objects, as in Python
        t, ns = project.translators[mod], project.modules[mod].namespace
        bound_here = set(binders[mod].bound) | set(binders[mod].bound_classes)
        for local, (owner, oname, kind) in t.imported.items():
            if "." in local or local in bound_here or kind not in ("function", "class"):
                continue
            ob = binders[owner]
            if oname in ob.bound or oname in ob.bound_classes:
                ons = project.modules[owner].namespace
                out.append(f"    sub_{ns}.attr({_cstr(local)}) = sub_{ons}.attr({_cstr(oname)});")
    if len(project.requested) == 1:
        main = project.modules[project.requested[0]].namespace
        out += [f"    for (auto kv : sub_{main}.attr(\"__dict__\").cast<py::dict>()) {{  // the module asked for, at the top",
                "        std::string k = py::str(kv.first);",
                "        if (k.rfind(\"__\", 0) != 0) m.attr(kv.first) = kv.second;",
                "    }"]
    out += ["    m.attr(\"_qtrans_submodules\") = submodules;",
            f"    m.attr(\"__qtrans_version__\") = {_cstr(VERSION)};", "}", ""]
    return "\n".join(out), binders


# --- building --------------------------------------------------------------------------------------
_SETUP = r'''# generated by qtrans: builds the extension in place
import sys
from setuptools import setup
from pybind11.setup_helpers import Pybind11Extension, build_ext


class Build(build_ext):
    def build_extensions(self):
        for e in self.extensions:
            if self.compiler.compiler_type == "msvc":
                e.extra_compile_args += ["/fp:precise", "/bigobj", "/O2"]
            else:
                # no fused multiply-add: results stay bit-identical to NumPy's
                e.extra_compile_args += ["-ffp-contract=off", "-O2"]
        super().build_extensions()


setup(name=@@NAME@@, ext_modules=[Pybind11Extension(@@NAME@@, [@@SRC@@], include_dirs=[@@INCLUDES@@], cxx_std=17)],
      cmdclass={"build_ext": Build}, script_args=["build_ext", "--inplace", "-q"])
'''


@dataclass
class Extension:
    path: Path            # the compiled module file
    name: str             # its import name
    project: object       # the ProjectResult (one module, or several)
    binders: dict         # module -> Binder
    build_dir: Path
    seconds: float = 0.0  # build time

    @property
    def single(self) -> bool:
        return len(self.project.modules) == 1

    @property
    def result(self) -> TranslationResult:
        return self.project.modules[self.project.requested[0]]

    @property
    def binder(self) -> Binder:
        return self.binders[self.project.requested[0]]

    @property
    def bound(self) -> list:
        return [n if self.single else f"{m}.{n}" for m, b in self.binders.items() for n in b.bound]

    @property
    def bound_classes(self) -> list:
        return [n for b in self.binders.values() for n in b.bound_classes]

    @property
    def notes(self) -> dict:
        return {(k if self.single else f"{m}.{k}"): v for m, b in self.binders.items() for k, v in b.notes.items()}

    def submodules(self, mod) -> dict:
        """module name -> the compiled (sub)module holding its functions and classes."""
        if self.single:
            return {self.project.order[0]: mod}
        return dict(mod._qtrans_submodules)

    def load(self):
        spec = importlib.util.spec_from_file_location(self.name, self.path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return mod


def build_extension(path, out_dir=None, build_dir=None, eigen=None, module_name=None) -> Extension:
    """Translate path (a file, a list of files, or a folder), generate the bindings and compile
    them with the compiler Python's own build tools use (MSVC on Windows, the system compiler
    elsewhere). The module is written to out_dir (default: next to the first file) as
    <stem>_cpp.<platform suffix> (for several files, <folder>_cpp). A file that imports other
    modules of its project gets one extension with a submodule per module."""
    try:
        import pybind11
    except ImportError as e:  # pragma: no cover
        raise RuntimeError("Building bindings needs pybind11: pip install pybind11") from e
    from .project import translate_project
    paths = [path] if isinstance(path, (str, Path)) and not Path(path).is_dir() else path
    build = Path(build_dir) if build_dir else Path(tempfile.mkdtemp(prefix="qtrans_bind_"))
    build.mkdir(parents=True, exist_ok=True)
    project = translate_project(paths, build)
    first = Path(project.paths[project.requested[0]]) if project.requested else Path(project.paths[project.order[0]])
    if len(project.requested) == 1:
        default = f"{project.requested[0]}_cpp"
    else:
        default = re.sub(r"\W", "_", (Path(path) if isinstance(path, (str, Path)) else first.parent).resolve().name) + "_cpp"
    name = module_name or default
    if len(project.modules) == 1:
        (m, result), = project.modules.items()
        src, binder = binding_source(result, Path(project.paths[m]).read_text(encoding="utf-8-sig"), name)
        binders = {m: binder}
    else:
        src, binders = project_binding_source(project, name)
    if not any(b.bound or b.bound_classes for b in binders.values()):
        raise RuntimeError(f"nothing in {first.name} could be translated and bound")
    path = first
    for h in (RUNTIME_HEADER, PYTHON_HEADER, PYRANDOM_HEADER, ARRAYS_HEADER, RANDOM_HEADER, PRINT_HEADER, LIST_HEADER, UNICODE_HEADER, PYBIND_HEADER):
        shutil.copy(h, build / h.name)
    cpp = build / f"{name}_bind.cpp"
    cpp.write_text(src, encoding="utf-8")
    eig = V.find_eigen(eigen)
    setup = build / "qtrans_setup.py"
    setup.write_text(_SETUP.replace("@@NAME@@", repr(name)).replace("@@SRC@@", repr(cpp.name))
                     .replace("@@INCLUDES@@", ", ".join(repr(str(d)) for d in (build, eig, pybind11.get_include()))),
                     encoding="utf-8")
    t0 = time.perf_counter()
    proc = subprocess.run([sys.executable, setup.name], cwd=build, capture_output=True, text=True,
                          encoding="utf-8", errors="replace")
    seconds = time.perf_counter() - t0
    built = sorted(build.glob(f"{name}*.pyd" if os.name == "nt" else f"{name}*.so"))
    if proc.returncode != 0 or not built:
        raise RuntimeError(f"building the bindings failed:\n{proc.stdout[-6000:]}\n{proc.stderr[-6000:]}")
    out = Path(out_dir) if out_dir else path.parent
    out.mkdir(parents=True, exist_ok=True)
    target = out / built[-1].name
    if target.resolve() != built[-1].resolve():
        try:
            shutil.copy(built[-1], target)
        except PermissionError:  # Windows: the old module is loaded by a running Python
            target = built[-1]
    return Extension(target, name, project, binders, build, seconds)


# --- checking ------------------------------------------------------------------------------------------
@dataclass
class CallableCheck:
    name: str
    kind: str                       # "function", "method", "constructor"
    cases: int = 0
    outcomes: dict = field(default_factory=dict)  # match / known / illcond / mismatch
    examples: list = field(default_factory=list)  # (args, Python result, compiled result) of mismatches
    py_ns: float | None = None
    cpp_ns: float | None = None
    cpp_obj_ns: float | None = None  # with compiled objects instead of your Python ones

    def add(self, outcome, example=None):
        self.cases += 1
        self.outcomes[outcome] = self.outcomes.get(outcome, 0) + 1
        if outcome == "mismatch" and example is not None and len(self.examples) < 3:
            self.examples.append(example)


@dataclass
class BindReport:
    path: str
    extension: Extension
    checks: list
    notes: dict

    @property
    def mismatches(self) -> int:
        return sum(c.outcomes.get("mismatch", 0) for c in self.checks)

    @property
    def ok(self) -> bool:
        return self.mismatches == 0

    def to_text(self) -> str:
        e = self.extension
        lines = [f"qtrans bind: {Path(self.path).name} -> {e.path.name} (built in {e.seconds:.1f} s)",
                 f"bound {len(e.bound)} function(s), {len(e.bound_classes)} class(es)"
                 + ("" if e.single else f" from {len(e.project.modules)} modules"), ""]
        timed = any(c.py_ns for c in self.checks)
        head = f"{'':36} {'cases':>6} {'match':>6} {'known':>6} {'illcond':>7} {'MISMATCH':>8}"
        if timed:
            head += f" {'Python':>10} {'compiled':>10} {'speedup':>8}"
        lines.append(head)
        for c in self.checks:
            row = (f"{c.name[:36]:36} {c.cases:6d} {c.outcomes.get('match', 0):6d} {c.outcomes.get('known', 0):6d} "
                   f"{c.outcomes.get('illcond', 0):7d} {c.outcomes.get('mismatch', 0):8d}")
            if timed and c.py_ns and c.cpp_ns:
                best = min(x for x in (c.cpp_ns, c.cpp_obj_ns) if x)
                row += f" {_fmt_ns(c.py_ns):>10} {_fmt_ns(best):>10} {c.py_ns / best:7.1f}x"
            lines.append(row)
        for c in self.checks:
            for args, py, cpp in c.examples:
                lines.append(f"  MISMATCH {c.name}{args!r}: Python {py!r}, compiled {cpp!r}")
        if self.notes:
            lines.append("")
            lines.append("Notes:")
            for k, ns in self.notes.items():
                for n in ns:
                    lines.append(f"  {k}: {n}")
        lines.append("")
        lines.append("OK: the compiled module agrees with Python on every case." if self.ok else
                     f"FAILED: {self.mismatches} case(s) disagree.")
        return "\n".join(lines)


def _fmt_ns(ns):
    if ns < 1e3:
        return f"{ns:.0f} ns"
    if ns < 1e6:
        return f"{ns / 1e3:.1f} us"
    if ns < 1e9:
        return f"{ns / 1e6:.1f} ms"
    return f"{ns / 1e9:.2f} s"


def _related(a, b) -> bool:
    """The same exception, as far as an `except` clause can tell (ShapeError is both
    ValueError and IndexError; RuntimeError stands for a raised Exception)."""
    return isinstance(a, type(b)) or isinstance(b, type(a))


def _call(fn, args):
    """-> ("ok", value) | ("exc", exception)"""
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            with np.errstate(all="ignore"):
                return ("ok", fn(*args))
    except Exception as e:  # noqa: BLE001 - every exception is an outcome
        return ("exc", e)


def _fresh(args, types):
    """Independent copies of the arguments (each side gets its own): arrays and objects copied,
    generator seeds turned into generators."""
    out = []
    for a, t in zip(args, types):
        if t == RNG and not isinstance(a, np.random.Generator):
            out.append(np.random.default_rng(int(a)))
        else:
            out.append(copy.deepcopy(a))
    return out


def _to_compiled(v, t, classes):
    """One of your Python objects as the compiled object (fields copied, no constructor run).
    classes: class name -> the compiled class."""
    if isinstance(t, StructType) and V._is_object(v) and t.name in classes:
        return classes[t.name]._from_python(v)
    if isinstance(t, OptionalType) and v is not None:
        return _to_compiled(v, t.inner, classes)
    return v


def _held(o, path=(), out=None, depth=0):
    """(path, object) for every array and generator your Python object holds, at any depth."""
    out = [] if out is None else out
    if isinstance(o, (np.ndarray, np.random.Generator)):
        out.append((path, o))
    elif depth < 6 and V._is_object(o) and not type(o).__module__.endswith("_cpp"):
        names = o._fields if isinstance(o, tuple) else list(vars(o))
        for n in names:
            _held(getattr(o, n), path + (n,), out, depth + 1)
    return out


def _snapshot(args, idx):
    """Before a call: what the objects passed for modified parameters hold, with copies of the
    values, so afterwards we can see what was changed in place and what was re-pointed."""
    return [(i, path, obj, copy.deepcopy(obj)) for i in idx for path, obj in _held(args[i], (i,))]


def _trace(args, snap):
    """After a call: for each array / generator held before, is it still the one held there,
    and what does that old object contain now?"""
    def key(x):
        if isinstance(x, np.random.Generator):
            return V.rng_fingerprint(copy.deepcopy(x))
        return x.tobytes() + str(x.shape).encode() if isinstance(x, np.ndarray) else None

    out = []
    for i, path, obj, before in snap:
        cur = args[i]
        try:
            for n in path[1:]:
                cur = getattr(cur, n)
        except AttributeError:
            cur = None
        now = key(obj)
        # nothing to see but identity: the old object is untouched and the field holds the same
        # values (re-pointed to an equal array, which the compiled side can't tell from no change)
        untouched = now == key(before) and cur is not None and key(cur) == now
        out.append((path, cur is obj, now if isinstance(obj, np.random.Generator) else obj, untouched))
    return out


def _same_trace(a, b, rtol, atol) -> bool:
    if len(a) != len(b):
        return False
    for (pa, ia, va, ua), (pb, ib, vb, ub) in zip(a, b):
        if pa != pb or (ia != ib and not (ua and ub)):
            return False
        if isinstance(va, np.ndarray):
            if not isinstance(vb, np.ndarray) or not V._compare_arrays(va, vb, rtol, atol)[0]:
                return False
        elif va != vb:
            return False
    return True


def _outcome(py, cpp, t, rtol, atol, sens_fn=None, flip_fn=None, exact_floats=False):
    """Compare ("ok", value) / ("exc", e) pairs the way verify does."""
    if py[0] == "exc" and cpp[0] == "exc" and _related(py[1], cpp[1]):
        return "match"
    if py[0] == "exc" and cpp[0] == "exc" and isinstance(py[1], (ValueError, IndexError, ZeroDivisionError)) \
            and isinstance(cpp[1], (ValueError, IndexError, ArithmeticError)):
        return "known"  # two broken checks in one expression, reported in a different order (see verify.classify)
    err = py[1] if py[0] == "exc" else cpp[1]
    if flip_fn is not None and (py[0] == "ok") != (cpp[0] == "ok") and isinstance(err, ValueError) \
            and flip_fn("ingular" in str(err) or "positive definite" in str(err)):
        return "illcond"  # a singular matrix on one side only, and Python's own verdict is a knife edge
    if py[0] == "ok" and V._has_complex(py[1]):
        # x ** y with x < 0 and fractional y: Python's value went complex, C++'s is NaN
        # (verify's documented difference); never compared as if it were real
        return "known"
    case = V.Case("", (), "")
    case.py = ("ok", py[1]) if py[0] == "ok" else ("exc", V._py_category(py[1]), str(py[1]))
    case.cpp = ("ok", cpp[1]) if cpp[0] == "ok" else ("exc", type(cpp[1]).__name__, str(cpp[1]))
    if py[0] == "ok" and cpp[0] == "ok":
        try:
            eq = V._compare_values(py[1], cpp[1], t, rtol, atol)
        except Exception:  # noqa: BLE001 - a value of the wrong shape or kind
            return "mismatch"
        if eq[0]:
            return "match"
        if sens_fn is not None:
            s = sens_fn()
            if eq[1] <= 100 * s:
                return "illcond"
        return "mismatch"
    if cpp[0] == "exc" and isinstance(cpp[1], OverflowError) and "beyond 64 bits" in str(cpp[1]):
        return "known"  # Python's int grew past 64 bits; C++ raises OverflowError (documented)
    if py[0] == "exc" and cpp[0] == "exc":
        case.cpp = ("exc", "unrelated", str(cpp[1]))  # so classify only accepts a known float error
    V.classify(case, t, rtol, atol, exact_floats=exact_floats)
    return case.outcome


def _result(fn, args, sig_params, ret, ref_idx):
    """Call, then collect (return value, final state of what was modified in place)."""
    r = _call(fn, args)
    if r[0] != "ok":
        return r, None
    v = r[1]
    refs = []
    for i in ref_idx:
        a = args[i]
        refs.append(V.rng_fingerprint(copy.deepcopy(a)) if isinstance(a, np.random.Generator) else a)
    vals = [] if isinstance(ret, VoidType) else (list(v) if isinstance(ret, TupleType) else [v])
    if isinstance(ret, TupleType) and (not isinstance(v, tuple) or len(v) != len(ret.elts)):
        return ("ok", v), None
    return ("ok", tuple(vals + refs)), TupleType(tuple(
        ([] if isinstance(ret, VoidType) else V._flatten(ret)) +
        [INT if sig_params[i].type == RNG else sig_params[i].type for i in ref_idx]))


def check_bindings(path, n_random=200, seed=0, rtol=1e-10, atol=1e-13, bench=False,
                   build_dir=None, eigen=None, ext: Extension | None = None, all_modules=False) -> BindReport:
    """Build (or take) the extension for path and check every function, constructor and method
    of the modules asked for (all_modules=True: of every module of the project)."""
    ext = ext or build_extension(path, out_dir=build_dir, build_dir=build_dir, eigen=eigen)
    project = ext.project
    subs = ext.submodules(ext.load())
    pymods = V.load_project(project)
    V.setup_inputs(project, pymods)
    compiled_classes = {}
    for m, b in ext.binders.items():
        for cname in b.bound_classes:
            compiled_classes[cname] = getattr(subs[m], cname)
    normwise = V.lapack_functions(project.paths[project.order[0]]) if ext.single else V.project_lapack_functions(project)
    rng = np.random.default_rng(seed)
    nudge_rng = np.random.default_rng(seed + 1)
    checks = []
    targets = project.order if all_modules else project.requested
    for module in targets:
        checks += _check_module(project.modules[module], pymods[module], subs[module], ext.binders[module],
                                compiled_classes, "" if ext.single else f"{module}.", n_random, rtol, atol, bench,
                                rng, nudge_rng, normwise, project)
    shown = project.paths[project.requested[0]] if len(project.requested) == 1 else Path(project.paths[targets[0]]).parent
    return BindReport(str(shown), ext, checks, ext.notes)


def _check_module(result, pymod, mod, binder, compiled_classes, prefix, n_random, rtol, atol, bench,
                  rng, nudge_rng, normwise, project=None) -> list:
    if project is not None:
        given, tolerances = V.module_cases(project, result.module, pymod)
    else:
        given = getattr(pymod, "QTRANS_CASES", {}) or {}
        tolerances = getattr(pymod, "QTRANS_TOLERANCE", {}) or {}
    bound = set(binder.bound)
    checks = []
    linalg = V.linalg_functions(project.paths.values()) if project is not None else set()

    def tol_of(name):
        tol = tolerances.get(name, rtol)
        return (tol, atol) if isinstance(tol, (int, float)) else tuple(tol)

    # functions
    for name, sig in result.signatures.items():
        if sig.owner or name not in bound:
            continue
        f_rtol, f_atol = tol_of(name)
        fc = CallableCheck(prefix + name, "function")
        fn, bfn = getattr(pymod, sig.name), getattr(mod, sig.name)
        types = [p.type for p in sig.params]
        ref_idx = [i for i, p in enumerate(sig.params) if p.mode == "ref"]
        cases = []
        defaults = [p.default for p in inspect.signature(fn).parameters.values()]
        for args in given.get(sig.name, []) if not getattr(sig, "overloaded", False) else []:
            args = list(args) + [defaults[i] for i in range(len(args), len(sig.params))]
            cases.append(tuple(V._py_value(a, p.type) for a, p in zip(args, sig.params)))
        plans = V.shape_plans(fn, sig, rng)
        for _ in range(n_random):
            try:
                cases.append(V.random_args(rng, sig, plans))
            except V._CantBuild:
                continue
        has_obj = any(isinstance(t, StructType) or (isinstance(t, OptionalType) and isinstance(t.inner, StructType))
                      for t in types)
        obj_refs = [i for i in ref_idx if isinstance(types[i], StructType)]
        for args in cases:
            pargs = _fresh(args, types)
            psnap = _snapshot(pargs, obj_refs)
            py, out_t = _result(fn, pargs, sig.params, sig.ret, ref_idx)
            ptrace = _trace(pargs, psnap)
            if out_t is None and py[0] == "ok":
                fc.add("known")  # Python returned something outside its annotation: not comparable
                continue

            def sens(args=args):
                c = V.Case(name, tuple(args), "")
                c.py = V.run_python(fn, c.args, sig)
                if c.py[0] != "ok":
                    return math.inf
                return V.python_sensitivity(fn, c, sig, V.output_type(sig), nudge_rng,
                                            normwise_too=name in normwise)

            def flips(singular, args=args):
                if name not in linalg:
                    return False
                c = V.Case(name, tuple(args), "")
                c.py = V.run_python(fn, c.args, sig)
                return V.python_flips(fn, c, sig, nudge_rng, normwise_too=name in normwise) \
                    or (singular and V.python_near_singular(fn, c, sig))
            variants = [_fresh(args, types)]
            if has_obj:
                variants.append([_to_compiled(a, t, compiled_classes) for a, t in zip(_fresh(args, types), types)])
            worst = "match"
            for k, bargs in enumerate(variants):
                bsnap = _snapshot(bargs, obj_refs) if k == 0 else []
                cpp, _ = _result(bfn, bargs, sig.params, sig.ret, ref_idx)
                o = _outcome(py, cpp, out_t or sig.ret, f_rtol, f_atol, sens, flips,
                             exact_floats=getattr(sig, "float_errors", False))
                if k == 0 and o != "mismatch" and not _same_trace(ptrace, _trace(bargs, bsnap), f_rtol, f_atol):
                    o = "mismatch"  # changed in place where Python re-points, or the other way round
                    cpp = ("ok", ("arrays held by the objects passed in:", _trace(bargs, bsnap), "Python:", ptrace))
                if o == "mismatch" and V.python_went_complex(fn, _fresh(args, types), sig):
                    o = "known"  # a ** b went complex in Python, NaN in C++ (see verify)
                if o == "mismatch":
                    fc.add("mismatch", (tuple(args), py[1], cpp[1]) + (("compiled objects",) if k else ()))
                    worst = None
                    break
                if o != "match":
                    worst = o
            if worst is not None:
                fc.add(worst)
        if bench:
            _bench(fc, fn, bfn, cases, types, compiled_classes, n_given=len(given.get(name, [])))
        checks.append(fc)

    # classes: constructors, methods and properties, on objects built the same way on both sides
    for cname in binder.bound_classes:
        info = result.classes[cname]
        pycls, bcls = getattr(pymod, cname), getattr(mod, cname)
        if info.plain:
            ctor_types = [p.type for p in V._INIT_PARAMS[cname]]
        else:
            ctor_types = [f.type for f in info.init_fields]
        cc = CallableCheck(f"{prefix}{cname}()", "constructor")
        pairs = []
        for _ in range(max(20, n_random // 4)):
            n = int(rng.integers(1, 7))
            try:
                raw = [V.random_value(rng, RNG) if t == RNG else V.random_value(rng, t, n) for t in ctor_types]
            except V._CantBuild:
                continue
            pa = _fresh(raw, ctor_types)
            ba = _fresh(raw, ctor_types)
            py, cpp = _call(pycls, pa), _call(bcls, ba)
            o = _outcome(py, cpp, info.type, rtol, atol)
            if py[0] == "ok" and cpp[0] == "ok":
                pairs.append((py[1], cpp[1], n))
            cc.add(o, (tuple(raw), py[1], cpp[1]))
        checks.append(cc)
        for m in info.methods.values():
            if m.kind in ("post_init", "init") or m.key not in result.signatures:
                continue
            msig = result.signatures[m.key]
            mname = f"{prefix}{cname}.{m.name}"
            if not hasattr(bcls, m.name):
                continue
            mc = CallableCheck(mname, "method")
            f_rtol, f_atol = tol_of(f"{cname}.{m.name}")
            types = [p.type for p in msig.params]
            ref_idx = [i for i, p in enumerate(msig.params) if p.mode == "ref"]
            for po, bo, n in (pairs * (1 + n_random // max(1, 4 * len(pairs))) if pairs else []):
                try:
                    args = [V.random_value(rng, t, n) for t in types]
                except V._CantBuild:
                    continue
                po2, bo2 = copy.deepcopy(po), copy.deepcopy(bo)
                if m.kind == "property":
                    py, cpp = _call(lambda: getattr(po2, m.name), ()), _call(lambda: getattr(bo2, m.name), ())
                    out_t = msig.ret
                else:
                    if m.kind in ("static", "class"):
                        pf, bf = getattr(pycls, m.name), getattr(bcls, m.name)
                    else:
                        pf, bf = getattr(po2, m.name), getattr(bo2, m.name)
                    py, out_t = _result(pf, _fresh(args, types), msig.params, msig.ret, ref_idx)
                    cpp, _ = _result(bf, _fresh(args, types), msig.params, msig.ret, ref_idx)
                if out_t is None and py[0] == "ok":
                    mc.add("known")
                    continue
                o = _outcome(py, cpp, out_t, f_rtol, f_atol, exact_floats=getattr(msig, "float_errors", False))
                if o in ("match", "known") and m.kind not in ("static", "class", "property"):
                    # and the object itself ends up the same (a method may change it)
                    eq = V._compare_values(po2, bo2, info.type, f_rtol, f_atol)[0]
                    if not eq:
                        o = "mismatch"
                mc.add(o, (tuple(args), py[1], cpp[1]))
            checks.append(mc)
    return checks


def _bench(fc, fn, bfn, cases, types, classes, n_given=0):
    """Time one representative call end to end on each side, as verify does: the hand-picked
    case with the most data if there is one, else the random case with the most data."""
    def runs(args):
        return _call(fn, _fresh(args, types))[0] == "ok" and _call(bfn, _fresh(args, types))[0] == "ok"
    ok = [a for a in cases[:n_given] if runs(a)]
    if not ok:
        ok = [a for a in cases[n_given:n_given + 60] if runs(a)]
    if not ok:
        return
    case = max(ok, key=lambda a: sum(np.size(x) for x in a if isinstance(x, np.ndarray)))
    fc.py_ns = V._bench_python(fn, _fresh(case, types))
    fc.cpp_ns = V._bench_python(bfn, _fresh(case, types))
    if any(isinstance(t, StructType) for t in types):
        fc.cpp_obj_ns = V._bench_python(bfn, [_to_compiled(a, t, classes) for a, t in zip(_fresh(case, types), types)])
