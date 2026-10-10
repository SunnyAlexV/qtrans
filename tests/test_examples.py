"""Every bundled example must translate fully and match Python with zero mismatches."""
from pathlib import Path

import pytest

from qtrans import verify_file
from tests.test_semantics import HAVE_CXX

EXAMPLES = Path(__file__).resolve().parent.parent / "examples"
FILES = ["black_scholes", "implied_vol", "futures_options", "bonds", "exotics",
         "portfolio", "lattice_pde", "risk", "monte_carlo", "black_litterman_app"]


@pytest.mark.skipif(not HAVE_CXX, reason="no C++ compiler")
@pytest.mark.parametrize("name", FILES)
def test_example_matches_python(name, tmp_path):
    r = verify_file(EXAMPLES / f"{name}.py", n_random=200, build_dir=tmp_path)
    assert not r.translation.report.rejected
    assert r.warnings == []
    assert r.ok, r.to_text()


def test_unsupported_example_reports_each_reason():
    from qtrans import translate_file
    r = translate_file(EXAMPLES / "not_yet_supported.py")
    ok = {f.name for f in r.report.translated}
    assert ok == {"discount", "forward"}
    assert len(r.report.rejected) == 8
