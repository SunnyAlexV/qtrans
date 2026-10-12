"""The pybind11 bindings: compiled modules must behave like the Python they came from, at the
boundary too (in-place changes, generator states, your own objects, exceptions)."""
import copy
from pathlib import Path

import numpy as np
import pytest

try:
    import pybind11  # noqa: F401
    HAVE_PYBIND = True
except ImportError:
    HAVE_PYBIND = False

pytestmark = pytest.mark.skipif(not HAVE_PYBIND, reason="pybind11 is not installed")

ROOT = Path(__file__).resolve().parent.parent
EXAMPLES = ROOT / "examples"
DATA = Path(__file__).resolve().parent / "data"


@pytest.fixture(scope="module")
def built(tmp_path_factory):
    """Each module is compiled once for all the tests here."""
    from qtrans.bind import build_extension
    cache = {}

    def get(path):
        if path not in cache:
            d = tmp_path_factory.mktemp(Path(path).stem)
            cache[path] = build_extension(path, out_dir=d, build_dir=d)
        return cache[path]
    return get


def _check(built, path, n=60):
    from qtrans.bind import check_bindings
    r = check_bindings(path, n_random=n, ext=built(path))
    assert r.ok, r.to_text()
    return r


# --- whole modules ------------------------------------------------------------------------------------
@pytest.mark.parametrize("name", ["instruments", "black_litterman_app"])
def test_example_bindings_match_python(built, name):
    r = _check(built, EXAMPLES / f"{name}.py")
    assert r.extension.binder.notes == {}
    assert sum(c.cases for c in r.checks) > 500


def test_partial_update_before_a_later_error_matches_python(built):
    """The release smoke test's case: rebalance() broadcasts a length-1 weights array against
    the target (NumPy stretches it), re-points weights and pays the cost, and only the second
    call fails (weights @ prices, 4 against 1). The object must be left as Python leaves it."""
    import importlib.util
    ext = built(EXAMPLES / "instruments.py")
    mod = ext.load()
    spec = importlib.util.spec_from_file_location("instruments_py", EXAMPLES / "instruments.py")
    py = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(py)
    states = []
    for fn in (py.run_rebalances, mod.run_rebalances):
        p = py.Portfolio(np.array([0.5]), 44.5)
        with pytest.raises(ValueError):
            fn(p, np.array([1.0, 7.0, 0.8, 0.37]), np.array([-0.5]), 90)
        states.append((p.weights.tolist(), p.cash, p.trades))
    assert states[0] == states[1] and states[0][2] == 1


def test_semantics_module_matches_python(built):
    r = _check(built, DATA / "bind_semantics.py", n=100)
    assert set(r.extension.binder.bound) >= {"scale_inplace", "draws", "solve2", "maybe", "add_into"}


# --- projects (several modules) ----------------------------------------------------------------------
PROJECT = DATA / "project"


def test_project_bindings_match_python(built):
    r = _check(built, PROJECT / "book.py", n=60)
    ext = r.extension
    assert not ext.single and ext.name == "book_cpp"
    assert {c.name.split(".")[0] for c in r.checks} == {"book"}  # the module asked for
    from qtrans.bind import check_bindings
    r_all = check_bindings(PROJECT / "book.py", n_random=40, ext=ext, all_modules=True)
    assert r_all.ok, r_all.to_text()
    assert {c.name.split(".")[0] for c in r_all.checks} == {"rates", "core", "book"}


def test_project_extension_layout(built):
    import sys
    m = built(PROJECT / "book.py").load()
    subs = m._qtrans_submodules
    rates, core = subs["rates"], subs["core"]
    assert m.book(100.0, np.ones(3)) == subs["book"].book(100.0, np.ones(3))  # the module asked for, at the top
    assert sys.modules["book_cpp.core"] is core                               # import book_cpp.core works
    assert core.discount is rates.discount                                    # imported names are the same objects
    assert core.Option is rates.Option and core.Portfolio is rates.Portfolio
    p = m.fresh(4)
    assert type(p) is rates.Portfolio                                         # one Python type project-wide
    assert m.strike_of(rates.Option(1.0, 2.0, 3.0)) == 2.0


def test_project_objects_cross_modules(built):
    import rates as py_rates
    m = built(PROJECT / "book.py").load()
    core = m._qtrans_submodules["core"]
    p = py_rates.Portfolio(np.array([1.0, 2.0]), 1.0)                         # your Python object, from rates
    w0 = p.w
    assert core.grow(p, np.array([1.0, 1.0]), 2.0) == 7.0
    assert p.w.tolist() == [2.0, 4.0] and w0 is p.w                           # scaled in place, as in Python
    w = np.array([1.0, 3.0])
    core.weights(w)                                                           # rates.normalise, through core
    assert w.tolist() == [0.25, 0.75]


def test_real_app_project_bindings_match_python(built):
    r = _check(built, EXAMPLES / "bl_app", n=60)
    assert len(r.extension.bound) == 22 and r.extension.name == "bl_app_cpp"


# --- arrays ---------------------------------------------------------------------------------------------
@pytest.fixture(scope="module")
def sem(built):
    return built(DATA / "bind_semantics.py").load()


def test_in_place_change_reaches_your_array(sem):
    x = np.array([1.0, 2.0, 3.0])
    view = x[1:]
    sem.scale_inplace(x, 2.0)
    assert x.tolist() == [2.0, 4.0, 6.0]
    assert view.tolist() == [4.0, 6.0]  # the same memory, as in Python


@pytest.mark.parametrize("bad", [[1.0, 2.0], np.array([1, 2]), np.array([1.0, 2.0], dtype=np.float32),
                                 np.arange(6.0)[::2], np.ones((2, 2))])
def test_in_place_needs_your_own_float64_array(sem, bad):
    # a converted copy would be changed instead of your data: refused, not silently wrong
    with pytest.raises(TypeError, match="modified in place"):
        sem.scale_inplace(bad, 2.0)


def test_read_only_arrays_are_converted(sem):
    assert sem.total([[1, 2], [3, 4]]) == 10.0
    assert sem.total(np.arange(6.0).reshape(2, 3)[:, ::2]) == 0.0 + 2 + 3 + 5


def test_partial_change_before_an_exception_is_kept(sem):
    x = np.array([5.0, 5.0])
    with pytest.raises(ValueError, match="n must be"):
        sem.fill_then_fail(x, 1)
    assert x.tolist() == [1.0, 5.0]  # as Python leaves it


def test_aliased_arguments_are_refused(sem):
    a = np.array([1.0, 2.0, 3.0])
    with pytest.raises(ValueError, match="share an array"):
        sem.add_into(a, a)
    with pytest.raises(ValueError, match="share an array"):
        sem.add_into(a, a[::-1])
    b = np.array([1.0, 1.0, 1.0])
    sem.add_into(a, b)
    assert a.tolist() == [2.0, 3.0, 4.0]


# --- generators -------------------------------------------------------------------------------------
def test_generator_state_is_written_back(sem):
    import bind_semantics as py  # noqa: F401  (tests/data is on sys.path below)
    g1, g2 = np.random.default_rng(42), np.random.default_rng(42)
    assert np.array_equal(sem.draws(g1, 5), py.draws(g2, 5))
    assert g1.bit_generator.state == g2.bit_generator.state
    assert g1.random() == g2.random()  # and it carries on in step


def test_generator_advances_even_when_the_call_raises(sem):
    import bind_semantics as py
    g1, g2 = np.random.default_rng(7), np.random.default_rng(7)
    with pytest.raises(ValueError):
        sem.draw_then_check(g1, -10.0)
    with pytest.raises(ValueError):
        py.draw_then_check(g2, -10.0)
    assert g1.bit_generator.state == g2.bit_generator.state


def test_generator_keeps_a_buffered_half_word(sem):
    import bind_semantics as py
    g1, g2 = np.random.default_rng(3), np.random.default_rng(3)
    for g in (g1, g2):
        g.integers(0, 10, dtype=np.int32)  # leaves a buffered 32-bit half in PCG64
    sem.draws(g1, 3)
    py.draws(g2, 3)
    assert g1.bit_generator.state == g2.bit_generator.state


def test_generator_argument_must_be_a_generator(sem):
    with pytest.raises(TypeError, match="Generator"):
        sem.draws(42, 3)


# --- exceptions, defaults -----------------------------------------------------------------------------
def test_exceptions_are_the_python_ones(sem):
    with pytest.raises(np.linalg.LinAlgError):
        sem.solve2(np.array([[1.0, 2.0], [2.0, 4.0]]), np.ones(2))
    with pytest.raises(ZeroDivisionError):
        sem.idiv(1, 0)
    with pytest.raises(IndexError):
        sem.pick(np.ones(2), 5)
    with pytest.raises(ValueError):  # ShapeError: both ValueError and IndexError, as NumPy may raise either
        sem.solve2(np.eye(3), np.ones(2))
    assert issubclass(sem.ShapeError, ValueError) and issubclass(sem.ShapeError, IndexError)


def test_defaults_none_and_keywords(sem):
    assert sem.maybe() == 0.5
    assert sem.maybe(None, "x") == 1.5
    assert sem.maybe(2.0) == 3.0
    assert sem.maybe(s="abc", x=1) == 2.0
    with pytest.raises(TypeError):
        sem.maybe("not a number")


def test_int_parameters_refuse_floats(sem):
    with pytest.raises(TypeError):
        sem.idiv(1.5, 2)


# --- classes ------------------------------------------------------------------------------------------
@pytest.fixture(scope="module")
def inst(built):
    return built(EXAMPLES / "instruments.py").load()


def test_your_objects_are_written_back_as_python_would(inst):
    import instruments as py
    for mod in (inst, py):
        p = py.Portfolio(np.array([0.5, 0.5]), 100.0)
        before = p.weights
        cost = mod.run_rebalances(p, np.array([0.2, 0.8]), np.array([10.0, 20.0]), 2)
        assert p.weights.tolist() == [0.2, 0.8] and p.trades == 2 and p.cash == pytest.approx(100.0 - cost)
        assert before.tolist() == [0.5, 0.5]  # rebalance re-points weights: the old array is untouched


def test_compiled_objects_change_in_place(inst):
    p = inst.Portfolio(np.array([1.0, 2.0]))
    p.scale(3.0)
    acc = inst.Account(p)
    acc.port.scale(2.0)  # a field of a compiled object is the object itself, not a copy
    assert acc.port.weights.tolist() == [6.0, 12.0]
    assert p.weights.tolist() == [3.0, 6.0]  # the constructor took a copy, as the C++ does
    with pytest.raises(ValueError):
        acc.port.weights[0] = 1.0  # a read-only copy: writing to it would change nothing


def test_dataclass_behaviour(inst):
    import instruments as py
    o, q = inst.Option(100.0, 95.0, 0.5, 0.2, 0.04), py.Option(100.0, 95.0, 0.5, 0.2, 0.04)
    assert repr(o) == repr(q)
    assert o == inst.Option(100.0, 95.0, 0.5, 0.2, 0.04) and o != inst.Option(100.0, 96.0, 0.5, 0.2, 0.04)
    assert hash(o) == hash(q)  # frozen: hashable, by its fields
    assert o.forward == q.forward and o.price() == q.price()
    with pytest.raises(AttributeError):
        o.S = 1.0  # frozen
    with pytest.raises(ValueError, match="must be positive"):
        inst.Option(-1.0, 95.0, 0.5, 0.2)  # __post_init__ runs
    with pytest.raises(TypeError):
        hash(inst.Portfolio(np.ones(2)))  # eq=True and not frozen: unhashable, as in Python
    assert inst.Bond(100.0, 0.05, 7.0).n_periods == 14
    c = copy.deepcopy(inst.Account(inst.Portfolio(np.ones(2))))
    assert c.name == "main" and c.port.weights.tolist() == [1.0, 1.0]


def test_namedtuple_unpacks(inst):
    q = inst.Quote(99.0, 101.0)
    bid, ask = q
    assert (bid, ask, len(q), q[1], q[-1], q.mid) == (99.0, 101.0, 2, 101.0, 101.0, 100.0)
    assert q._fields == ("bid", "ask")


def test_engine_generator_moves_on(inst):
    import instruments as py
    e1, e2 = inst.GBMEngine(100.0, 0.04, 0.2, 7), py.GBMEngine(100.0, 0.04, 0.2, 7)
    assert inst.reprice(e1, 100.0) == py.reprice(e2, 100.0)
    assert e1.paths_run == e2.paths_run == 2000
    assert e1.rng.random() == e2.rng.random()
    e3 = py.GBMEngine(100.0, 0.04, 0.2, 7)  # your own engine, through the compiled function
    assert inst.reprice(e3, 100.0) == inst.reprice(inst.GBMEngine(100.0, 0.04, 0.2, 7), 100.0)
    assert e3.paths_run == 2000


def test_wrong_object_type_is_a_type_error(inst):
    with pytest.raises(TypeError, match="must be an Account"):
        inst.leveraged_nav("x", np.ones(2), 1.0)


@pytest.fixture(autouse=True, scope="module")
def _python_modules_importable():
    import sys
    sys.path[:0] = [str(DATA), str(EXAMPLES), str(PROJECT)]
    yield
    for p in (str(DATA), str(EXAMPLES), str(PROJECT)):
        sys.path.remove(p)


def test_what_a_compiled_function_prints_goes_to_python_stdout(built, tmp_path, capsys):
    """A function that prints, called from Python: its output goes through sys.stdout (so a
    notebook shows it, in order with Python's own prints), exactly as Python's print writes it."""
    import contextlib
    import io
    f = tmp_path / "talk.py"
    f.write_text("import sys\n\n\ndef greet(n: int, x: float) -> float:\n    print('hello', n, x / 3)\n"
                 "    print(f'{x:.2f}', file=sys.stderr)\n    return x * n\n", encoding="utf-8")
    mod = built(f).load()
    out, err = io.StringIO(), io.StringIO()
    with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
        print("before")
        r = mod.greet(3, 2.5)
        print("after")
    assert r == 7.5 and out.getvalue() == "before\nhello 3 0.8333333333333334\nafter\n" and err.getvalue() == "2.50\n"
