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
from .translator import ARRAYS_HEADER, RANDOM_HEADER, RUNTIME_HEADER, VERSION, TranslationResult, translate_file
from .types import (BOOL, FLOAT, INT, MAT, RNG, STR, VEC, ArrayType, OptionalType, Scalar, StructType, TupleType,
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
            MAT: "a 2-D float array", RNG: "a numpy Generator"}.get(t, str(t))


def _article(name: str) -> str:
    return ("an " if name[:1].lower() in "aeiou" else "a ") + name


def _is_leafy(t) -> bool:
    """Can a value of type t hold memory the function could share with another argument?"""
    inner = t.inner if isinstance(t, OptionalType) else t
    return isinstance(inner, (ArrayType, StructType)) or inner == RNG


def _native(t) -> bool:
    """Passed straight through pybind11's own conversions."""
    inner = t.inner if isinstance(t, OptionalType) else t
    return isinstance(inner, Scalar) or inner == STR


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
    def __init__(self, result: TranslationResult, source: str):
        self.result = result
        self.ns = result.header.split("namespace ", 1)[1].split(" ", 1)[0]
        self.classes = dict(result.classes or {})
        self.tree = ast.parse(source)
        self.notes = {}       # name -> [why it is left out / how it differs]
        self.bound = []       # Python names of the functions bound
        self.bound_classes = []
        self._index_functions()
        self.variants = {frozenset(): 0}  # rebind-only field sets -> write-back variant number
        self.variant = 0                  # the one for the callable being generated

    # --- which object fields Python only ever re-points (never changes in place) -----------------
    def _index_functions(self):
        self.top = {n.name: n for n in self.tree.body if isinstance(n, ast.FunctionDef)}
        self.methods = {}   # (class, name) -> FunctionDef
        self.by_name = {}   # method name -> [FunctionDef]
        self.cls_defs = {}
        for n in self.tree.body:
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
        elif t == RNG:
            w.pre.append(f"qt::Rng {v} = qtb::rng_in({o}, {_cstr(p.name)});")
            w.post.append(f"qtb::rng_back({o}, {v});")
            w.args.append(v)
        elif isinstance(t, StructType):
            info = self.classes.get(t.name)
            if info is None:
                raise BindError(f"uses class {t.name}, which isn't translated")
            cpp = info.cpp
            w.pre.append(f"std::optional<{cpp}> {v}_tmp;")
            decl = f"const {cpp}&" if p.mode == "cref" else f"{cpp}&" if p.mode == "ref" else cpp
            w.pre.append(f"{decl} {v} = qtb::obtain<{cpp}>({o}, {v}_tmp, [](py::handle h) {{ return "
                         f"qtrans_from_py_{cpp}(h, {_cstr(p.name)}); }});")
            if modified:
                w.post.append(f"if ({v}_tmp) qtrans_to_py_{cpp}_{self.variant}({o}, *{v}_tmp);")
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
        if isinstance(inner, StructType) and inner.name not in self.classes:
            raise BindError(f"returns a {inner.name}, which isn't translated")

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
        self._ret_ok(sig.ret)
        lam, pyargs, notes = self._callable(sig, lambda a: f"{sig.cpp}({', '.join(a)})")
        if notes:
            self.notes.setdefault(name, []).extend(notes)
        extra = "".join(", " + a for a in pyargs)
        doc = _doc(sig.doc)
        return [f"    m.def({_cstr(name)}, {lam}{extra}{', ' + doc if doc else ''});"]

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
        cpp = info.cpp
        out = [f"static {cpp} qtrans_from_py_{cpp}(py::handle o, const char* arg) {{",
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
                           f"qtrans_from_py_{sc}(h, {nm}); }}); }}")
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
        out = [f"static void qtrans_to_py_{cpp}_{k}(py::handle o, const {cpp}& v) {{"]
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
                               f"v.{f.cpp}; else qtrans_to_py_{sc}_{k}(c, v.{f.cpp}); }}")
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
                          f"[](py::handle h) {{ return qtrans_from_py_{sc}(h, {nm}); }}); }}")
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
        lines.append(f"    {var}.def_static(\"_from_python\", [](py::object o) {{ return qtrans_from_py_{cpp}(o, \"o\"); }}, "
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

        out = [f"// {module_name}_bind.cpp: Python bindings generated by qtrans {VERSION} for {header_name}.",
               "// Build with `python -m qtrans bind`; see qtrans/bind.py for what the bindings guarantee.",
               f'#include "{header_name}"']
        if res.uses_random:
            out.append("#define QTRANS_BIND_RNG 1")
        out += ['#include "qtrans_pybind.hpp"', "", "namespace py = pybind11;", "", f"namespace {self.ns} {{",
                "namespace qtrans_binding {", ""]
        body = []
        for info in ok_classes:
            try:
                body += self.bind_class(info)
                self.bound_classes.append(info.name)
            except BindError as e:
                self.notes.setdefault(info.name, []).append(f"left out: {e}")
        for name, sig in res.signatures.items():
            if sig.owner:
                continue
            try:
                self._use_variant(self.top.get(name))
                body += self.bind_function(name, sig)
                self.bound.append(name)
            except BindError as e:
                self.notes.setdefault(name, []).append(f"left out: {e}")

        for info in ok_classes:
            out.append(f"static {info.cpp} qtrans_from_py_{info.cpp}(py::handle o, const char* arg);")
            for k in self.variants.values():
                out.append(f"[[maybe_unused]] static void qtrans_to_py_{info.cpp}_{k}(py::handle o, const {info.cpp}& v);")
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
        out.append("static void bind(py::module_& m) {")
        out += body
        out += ["}", "", "}  // namespace qtrans_binding", f"}}  // namespace {self.ns}", ""]
        doc = _doc(f"{res.module}, compiled by qtrans {VERSION}: the translated functions and classes of "
                   f"{res.module}.py.")
        out += [f"PYBIND11_MODULE({module_name}, m) {{",
                f"    m.doc() = {doc};",
                "    qtb::register_exceptions(m);",
                f"    {self.ns}::qtrans_binding::bind(m);",
                f"    m.attr(\"__qtrans_version__\") = {_cstr(VERSION)};",
                "}", ""]
        return "\n".join(out)


def binding_source(result: TranslationResult, source: str, module_name: str | None = None):
    """-> (C++ source of the pybind11 module, Binder with what was bound and notes)."""
    b = Binder(result, source)
    name = module_name or f"{result.module}_cpp"
    return b.source(f"{result.module}.hpp", name), b


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
    result: TranslationResult
    binder: Binder
    build_dir: Path
    seconds: float = 0.0  # build time

    def load(self):
        spec = importlib.util.spec_from_file_location(self.name, self.path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return mod


def build_extension(path, out_dir=None, build_dir=None, eigen=None, module_name=None) -> Extension:
    """Translate path, generate the bindings and compile them with the compiler Python's own
    build tools use (MSVC on Windows, the system compiler elsewhere). The module is written to
    out_dir (default: next to path) as <stem>_cpp.<platform suffix>."""
    try:
        import pybind11
    except ImportError as e:  # pragma: no cover
        raise RuntimeError("Building bindings needs pybind11: pip install pybind11") from e
    path = Path(path).resolve()
    build = Path(build_dir) if build_dir else Path(tempfile.mkdtemp(prefix="qtrans_bind_"))
    build.mkdir(parents=True, exist_ok=True)
    result = translate_file(path, build)
    name = module_name or f"{path.stem}_cpp"
    src, binder = binding_source(result, path.read_text(encoding="utf-8"), name)
    if not binder.bound and not binder.bound_classes:
        raise RuntimeError(f"nothing in {path.name} could be translated and bound")
    for h in (RUNTIME_HEADER, ARRAYS_HEADER, RANDOM_HEADER, PYBIND_HEADER):
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
    return Extension(target, name, result, binder, build, seconds)


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
                 f"bound {len(e.binder.bound)} function(s), {len(e.binder.bound_classes)} class(es)", ""]
        timed = any(c.py_ns for c in self.checks)
        head = f"{'':32} {'cases':>6} {'match':>6} {'known':>6} {'illcond':>7} {'MISMATCH':>8}"
        if timed:
            head += f" {'Python':>10} {'compiled':>10} {'speedup':>8}"
        lines.append(head)
        for c in self.checks:
            row = (f"{c.name[:32]:32} {c.cases:6d} {c.outcomes.get('match', 0):6d} {c.outcomes.get('known', 0):6d} "
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


def _to_compiled(v, t, mod):
    """One of your Python objects as the compiled object (fields copied, no constructor run)."""
    if isinstance(t, StructType) and V._is_object(v):
        return getattr(mod, t.name)._from_python(v)
    if isinstance(t, OptionalType) and v is not None:
        return _to_compiled(v, t.inner, mod)
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
            return int(copy.deepcopy(x).bit_generator.random_raw())
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


def _outcome(py, cpp, t, rtol, atol, sens_fn=None):
    """Compare ("ok", value) / ("exc", e) pairs the way verify does."""
    if py[0] == "exc" and cpp[0] == "exc" and _related(py[1], cpp[1]):
        return "match"
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
    if py[0] == "exc" and cpp[0] == "exc":
        case.cpp = ("exc", "unrelated", str(cpp[1]))  # so classify only accepts a known float error
    V.classify(case, t, rtol, atol)
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
        refs.append(int(copy.deepcopy(a).bit_generator.random_raw()) if isinstance(a, np.random.Generator) else a)
    vals = [] if isinstance(ret, VoidType) else (list(v) if isinstance(ret, TupleType) else [v])
    if isinstance(ret, TupleType) and (not isinstance(v, tuple) or len(v) != len(ret.elts)):
        return ("ok", v), None
    return ("ok", tuple(vals + refs)), TupleType(tuple(
        ([] if isinstance(ret, VoidType) else V._flatten(ret)) +
        [INT if sig_params[i].type == RNG else sig_params[i].type for i in ref_idx]))


def check_bindings(path, n_random=200, seed=0, rtol=1e-10, atol=1e-13, bench=False,
                   build_dir=None, eigen=None, ext: Extension | None = None) -> BindReport:
    path = Path(path).resolve()
    ext = ext or build_extension(path, out_dir=build_dir, build_dir=build_dir, eigen=eigen)
    result = ext.result
    mod = ext.load()
    pymod = V.load_module(path)
    # the verifier's input generation, set up for this module
    src = path.read_text(encoding="utf-8")
    V._STRINGS = sorted({n.value for n in ast.walk(ast.parse(src))
                         if isinstance(n, ast.Constant) and isinstance(n.value, str) and len(n.value) < 80}) + \
        ["", "no such option"]
    V._CLASSES = {n: (info, getattr(pymod, n)) for n, info in (result.classes or {}).items()}
    V._INIT_PARAMS = {n: result.signatures[f"{n}.__init__"].params for n, info in (result.classes or {}).items()
                      if info.plain}
    given = getattr(pymod, "QTRANS_CASES", {}) or {}
    tolerances = getattr(pymod, "QTRANS_TOLERANCE", {}) or {}
    normwise = V.lapack_functions(path)
    rng = np.random.default_rng(seed)
    nudge_rng = np.random.default_rng(seed + 1)
    bound = set(ext.binder.bound)
    checks = []

    def tol_of(name):
        tol = tolerances.get(name, rtol)
        return (tol, atol) if isinstance(tol, (int, float)) else tuple(tol)

    # functions
    for name, sig in result.signatures.items():
        if sig.owner or name not in bound:
            continue
        f_rtol, f_atol = tol_of(name)
        fc = CallableCheck(name, "function")
        fn, bfn = getattr(pymod, name), getattr(mod, name)
        types = [p.type for p in sig.params]
        ref_idx = [i for i, p in enumerate(sig.params) if p.mode == "ref"]
        cases = []
        defaults = [p.default for p in inspect.signature(fn).parameters.values()]
        for args in given.get(name, []):
            args = list(args) + [defaults[i] for i in range(len(args), len(sig.params))]
            cases.append(tuple(V._py_value(a, p.type) for a, p in zip(args, sig.params)))
        for _ in range(n_random):
            n = 0 if rng.random() < 0.05 else int(rng.integers(1, 7))
            try:
                cases.append(tuple(V.random_value(rng, t, n) for t in types))
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
            variants = [_fresh(args, types)]
            if has_obj:
                variants.append([_to_compiled(a, t, mod) for a, t in zip(_fresh(args, types), types)])
            worst = "match"
            for k, bargs in enumerate(variants):
                bsnap = _snapshot(bargs, obj_refs) if k == 0 else []
                cpp, _ = _result(bfn, bargs, sig.params, sig.ret, ref_idx)
                o = _outcome(py, cpp, out_t or sig.ret, f_rtol, f_atol, sens)
                if k == 0 and o != "mismatch" and not _same_trace(ptrace, _trace(bargs, bsnap), f_rtol, f_atol):
                    o = "mismatch"  # changed in place where Python re-points, or the other way round
                    cpp = ("ok", ("arrays held by the objects passed in:", _trace(bargs, bsnap), "Python:", ptrace))
                if o == "mismatch":
                    fc.add("mismatch", (tuple(args), py[1], cpp[1]) + (("compiled objects",) if k else ()))
                    worst = None
                    break
                if o != "match":
                    worst = o
            if worst is not None:
                fc.add(worst)
        if bench:
            _bench(fc, fn, bfn, cases, types, mod, n_given=len(given.get(name, [])))
        checks.append(fc)

    # classes: constructors, methods and properties, on objects built the same way on both sides
    for cname in ext.binder.bound_classes:
        info = result.classes[cname]
        pycls, bcls = getattr(pymod, cname), getattr(mod, cname)
        if info.plain:
            ctor_types = [p.type for p in V._INIT_PARAMS[cname]]
        else:
            ctor_types = [f.type for f in info.init_fields]
        cc = CallableCheck(f"{cname}()", "constructor")
        pairs = []
        for _ in range(max(20, n_random // 4)):
            n = int(rng.integers(1, 7))
            raw = [V.random_value(rng, RNG) if t == RNG else V.random_value(rng, t, n) for t in ctor_types]
            pa, ba = _fresh(raw, ctor_types), _fresh(raw, ctor_types)
            py, cpp = _call(pycls, pa), _call(bcls, ba)
            if py[0] == "ok" and cpp[0] == "ok":
                o = _outcome(py, cpp, info.type, rtol, atol)
                pairs.append((py[1], cpp[1], n))
            else:
                o = _outcome(py, cpp, info.type, rtol, atol)
            cc.add(o, (tuple(raw), py[1], cpp[1]))
        checks.append(cc)
        for m in info.methods.values():
            if m.kind in ("post_init", "init") or m.key not in result.signatures:
                continue
            msig = result.signatures[m.key]
            mname = f"{cname}.{m.name}"
            if not hasattr(bcls, m.name):
                continue
            mc = CallableCheck(mname, "method")
            f_rtol, f_atol = tol_of(mname)
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
                o = _outcome(py, cpp, out_t, f_rtol, f_atol)
                if o in ("match", "known") and m.kind not in ("static", "class", "property"):
                    # and the object itself ends up the same (a method may change it)
                    eq = V._compare_values(po2, bo2, info.type, f_rtol, f_atol)[0]
                    if not eq:
                        o = "mismatch"
                mc.add(o, (tuple(args), py[1], cpp[1]))
            checks.append(mc)
    return BindReport(str(path), ext, checks, ext.binder.notes)


def _bench(fc, fn, bfn, cases, types, mod, n_given=0):
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
        fc.cpp_obj_ns = V._bench_python(bfn, [_to_compiled(a, t, mod) for a, t in zip(_fresh(case, types), types)])
