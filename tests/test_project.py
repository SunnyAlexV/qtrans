"""Multi-module projects: modules that import from each other become one header each, with the
analysis (in-place changes, generators, rejections) carried across the imports."""
import textwrap
from pathlib import Path

import numpy as np
import pytest

from qtrans import ProjectError, translate_file, translate_project
from tests.test_semantics import HAVE_CXX

DATA = Path(__file__).resolve().parent / "data" / "project"


def write(folder, **files):
    for name, src in files.items():
        (folder / f"{name}.py").write_text(textwrap.dedent(src), encoding="utf-8")
    return folder


@pytest.fixture(scope="module")
def toy():
    return translate_project([DATA / "book.py"])


def test_imports_are_followed_in_order(toy):
    assert toy.order == ["rates", "core", "book"]  # each after the modules it imports
    assert toy.requested == ["book"]
    assert toy.modules["core"].deps == ["rates"]
    assert set(toy.modules["book"].deps) == {"core", "rates"}


def test_headers_include_and_qualify(toy):
    core, book = toy.modules["core"].header, toy.modules["book"].header
    assert '#include "rates.hpp"' in core and "namespace core {" in core
    assert "return x * static_cast<double>(rates::DAYS);" in core          # star-imported constant
    assert "double pv(double cf, double t, double rf = rates::DEFAULT_RF);" in core  # default from rates
    assert "cf * rates::discount(rates::DEFAULT_RF, t)" in core              # import rates as R; R.discount
    assert "rates::Option(S, K, T).forward() - rates::Option(S, K, T, 0.0).forward()" in core
    assert "rates::Option make(double S)" in core
    assert "double strike_of(const rates::Option& o)" in book                # a class two modules away
    assert "rates::Portfolio fresh(long long n)" in book
    assert "core::fwd(o.S, o.K, o.T) + core::weights(w)" in book


def test_in_place_changes_cross_modules(toy):
    """rates.normalise modifies its argument, so core.weights(w), which passes w on, modifies
    the caller's array too, and so does book.book, two modules up: all take it by reference."""
    sigs = {m: r.signatures for m, r in toy.modules.items()}
    assert sigs["rates"]["normalise"].params[0].mode == "ref"
    assert sigs["core"]["weights"].params[0].mode == "ref"
    assert sigs["book"]["book"].params[1].mode == "ref"
    assert sigs["core"]["grow"].params[0].mode == "ref"   # p.scale(k): a method of rates.Portfolio
    assert sigs["core"]["draws"].params[0].mode == "ref"  # passes the generator to rates.noisy


def test_rejections_say_why_across_modules(toy):
    why = {f.name: f.reason for f in toy.modules["core"].report.rejected}
    assert "bad (in rates) could not be translated" in why["uses_bad"]
    assert "dicts are not supported" in why["uses_bad"]
    assert "R.TABLE" in why["uses_table"] and "(in rates)" in why["uses_table"]
    # rates.safe_log may raise a math domain error: catching ValueError around it is refused
    assert "except ValueError" in why["guarded"]


def test_star_import_skips_private_names_and_respects_all(tmp_path):
    write(tmp_path,
          a='''
          __all__ = ["f"]

          def f(x: float) -> float:
              return x + 1.0

          def g(x: float) -> float:
              return x * 2.0
          ''',
          b='''
          from a import *

          def h(x: float) -> float:
              return f(x)

          def k(x: float) -> float:
              return g(x)
          ''')
    p = translate_project([tmp_path / "b.py"])
    rep = p.modules["b"].report
    assert [f.name for f in rep.translated] == ["h"]
    assert "unknown function 'g'" in next(f.reason for f in rep.rejected if f.name == "k")


def test_own_definition_shadows_an_imported_one(tmp_path):
    write(tmp_path,
          a='''
          def f(x: float) -> float:
              return x + 1.0
          ''',
          b='''
          from a import *

          def f(x: float) -> float:
              return x - 1.0

          def g(x: float) -> float:
              return f(x)
          ''')
    h = translate_project([tmp_path / "b.py"]).modules["b"].header
    assert "return f(x);" in h and "a::f" not in h


def test_a_name_like_a_module_gets_the_global_namespace(tmp_path):
    write(tmp_path,
          rates='''
          def discount(r: float) -> float:
              return 1.0 / (1.0 + r)
          ''',
          pricing='''
          from rates import discount

          def rates(r: float) -> float:
              return discount(r)
          ''')
    h = translate_project([tmp_path / "pricing.py"]).modules["pricing"].header
    assert "return ::rates::discount(r);" in h


def test_import_cycle_is_refused(tmp_path):
    write(tmp_path,
          a='''
          from b import g

          def f(x: float) -> float:
              return g(x)
          ''',
          b='''
          from a import f

          def g(x: float) -> float:
              return x
          ''')
    with pytest.raises(ProjectError, match="import cycle: a -> b -> a"):
        translate_project([tmp_path / "a.py"])


def test_class_names_must_be_unique(tmp_path):
    write(tmp_path,
          a='''
          from dataclasses import dataclass

          @dataclass
          class Leg:
              x: float
          ''',
          b='''
          from dataclasses import dataclass
          import a

          @dataclass
          class Leg:
              y: float
          ''')
    with pytest.raises(ProjectError, match="class Leg is defined in both"):
        translate_project([tmp_path / "b.py"])


def test_class_imported_under_another_name_is_refused(tmp_path):
    write(tmp_path,
          a='''
          from dataclasses import dataclass

          @dataclass(frozen=True)
          class Leg:
              x: float
          ''',
          b='''
          from a import Leg as L

          def f(x: float) -> float:
              return L(x).x
          ''')
    rep = translate_project([tmp_path / "b.py"]).modules["b"].report
    assert rep.rejected and "import it as Leg" in rep.rejected[0].reason


def test_relative_imports_in_a_package(tmp_path):
    pkg = tmp_path / "quant"
    pkg.mkdir()
    (pkg / "__init__.py").write_text("")
    write(pkg,
          maths='''
          import math

          def disc(r: float, t: float) -> float:
              return math.exp(-r * t)
          ''',
          pricer='''
          from .maths import disc
          from . import maths

          def pv(c: float, r: float, t: float) -> float:
              return c * disc(r, t) + 0.0 * maths.disc(r, t)
          ''')
    p = translate_project([pkg / "pricer.py"])
    assert "c * maths::disc(r, t) + 0.0 * maths::disc(r, t)" in p.modules["pricer"].header


def test_translate_file_pulls_in_its_imports(tmp_path):
    r = translate_file(DATA / "core.py", tmp_path)
    assert r.module == "core"
    assert (tmp_path / "rates.hpp").exists() and (tmp_path / "core.hpp").exists()


def test_single_module_unchanged():
    ex = Path(__file__).resolve().parent.parent / "examples" / "black_scholes.py"
    p = translate_project([ex])
    assert p.order == ["black_scholes"] and p.modules["black_scholes"].deps == []


# --- compiled -----------------------------------------------------------------------------------------
@pytest.mark.skipif(not HAVE_CXX, reason="no C++ compiler")
def test_toy_project_matches_python(tmp_path):
    from qtrans import verify_project
    r = verify_project([DATA / "book.py"], n_random=80, build_dir=tmp_path, all_modules=True)
    assert set(r.modules) == {"rates", "core", "book"}
    assert all(m.warnings == [] for m in r.modules.values())
    assert r.ok, r.to_text()


@pytest.mark.skipif(not HAVE_CXX, reason="no C++ compiler")
def test_package_matches_python(tmp_path):
    from qtrans import verify_project
    pkg = tmp_path / "src" / "quant"
    pkg.mkdir(parents=True)
    (pkg / "__init__.py").write_text("")
    write(pkg,
          maths='''
          import math
          import numpy as np
          from qtrans.typing import Vec

          def disc(r: float, t: float) -> float:
              return math.exp(-r * t)

          def demean(x: Vec) -> None:
              x -= np.mean(x)
          ''',
          pricer='''
          import numpy as np
          from qtrans.typing import Vec
          from .maths import disc, demean

          def pv(c: float, r: float, t: float) -> float:
              return c * disc(r, t)

          def spread(x: Vec) -> float:
              demean(x)
              return float(np.max(x) - np.min(x)) if len(x) else 0.0
          ''')
    r = verify_project([pkg / "pricer.py"], n_random=80, build_dir=tmp_path / "b", all_modules=True)
    assert r.ok, r.to_text()
    assert r.modules["pricer"].translation.signatures["spread"].params[0].mode == "ref"


@pytest.mark.skipif(not HAVE_CXX, reason="no C++ compiler")
def test_real_app_project_matches_python(tmp_path):
    """The app's own files (type hints added in place), as a project: bl_core and bl_backtest
    star-import bl_config, bl_backtest star-imports bl_core."""
    from qtrans import verify_project
    app = Path(__file__).resolve().parent.parent / "examples" / "bl_app"
    r = verify_project(app, n_random=120, build_dir=tmp_path, all_modules=True)
    assert set(r.modules) == {"bl_config", "bl_core", "bl_inference", "bl_backtest"}
    assert sum(len(m.translation.report.translated) for m in r.modules.values()) == 22
    assert all(m.warnings == [] for m in r.modules.values())
    assert r.ok, r.to_text()
    np.testing.assert_equal(len(r.mismatches), 0)
