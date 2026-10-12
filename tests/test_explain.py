"""qtrans explain / inspect / doctor, and the source map under them: every C++ line of a function
comes from a line of that function, every Python line is in exactly one row, and every reason
is about C++ that is actually in its row."""
import glob
import html.parser
import json
import re
from pathlib import Path

import pytest

from qtrans import doctor
from qtrans.cli import main as cli
from qtrans.explain import _COMPILED, explain_file, explain_result, to_html, to_json, to_text
from qtrans.inspect_ import inspect_file
from qtrans.translator import translate_file, translate_source
from tests.test_semantics import HAVE_CXX

ROOT = Path(__file__).resolve().parents[1]
FILES = sorted(glob.glob(str(ROOT / "examples" / "*.py"))) + sorted(glob.glob(str(ROOT / "examples" / "bl_app" / "*.py"))) \
    + [str(ROOT / "tests" / "data" / "fuzz_regressions.py")]
_DECL = re.compile(r"^[\w:<>, &*]+ \w+\(.*\);$")  # a forward declaration at namespace level


@pytest.mark.parametrize("path", FILES, ids=lambda p: Path(p).name)
def test_source_map_is_complete_and_local(path):
    r = translate_file(path)
    assert "\x1f" not in r.header
    lines = r.header.splitlines()
    assert len(r.line_map) == len(lines)
    view = explain_result(r, Path(path).read_text(encoding="utf-8"), path)
    for f in view.functions:
        if not f.translated:
            continue
        seen = []
        for row in f.rows:
            assert f.first <= row.py_first <= row.py_last <= f.last, (f.name, row.py_first)
            seen += range(row.py_first, row.py_last + 1)
            for why in row.reasons:  # a reason about C++ is next to that C++
                if why.token and why.kind != "passing":
                    assert any(why.token in c for c in row.cpp), (f.name, why.token)
        assert seen == list(range(f.first, f.last + 1)), f.name  # every line once, in order
    # every line inside a function body comes from some Python line
    depth, unmapped = 0, []
    for line, span in zip(lines, r.line_map):
        if line.startswith("inline ") and line.rstrip().endswith("{"):
            depth = 1
        elif depth:
            if span is None and line.strip():
                unmapped.append(line)
            if line == "}":
                depth = 0
    assert unmapped == []


def test_statements_map_to_their_own_lines():
    src = ("import numpy as np\nfrom qtrans.typing import Vec\n\n"
           "def f(v: Vec, k: int) -> float:\n"   # 4
           "    s = 0.0\n"                        # 5
           "    for i in range(len(v)):\n"        # 6
           "        if v[i] > 0:\n"               # 7
           "            s += v[i]\n"              # 8
           "        else:\n"                      # 9
           "            s -= 1.0\n"               # 10
           "    v[::-1] = v\n"                    # 11
           "    return s * k\n")                  # 12
    r = translate_source(src, "m")
    by = {}
    for line, span in zip(r.header.splitlines(), r.line_map):
        if span:
            by.setdefault(span[0], []).append(line.strip())
    assert by[4] == ["inline double f(qt::Vec& v, long long k) {"]
    assert by[6][-1].startswith("for (long long i = 0;")
    assert by[7] == ["if (v(i) > 0) {"]
    assert by[9] == ["} else {"]
    assert by[10] == ["s -= 1.0;", "}", "}"]  # the closing braces of the if and the for come after its last line
    assert by[11] == ["qt::store(qt::slice(v, qt::none, qt::none, -1), qt::Vec(v));"]
    assert by[12] == ["return s * static_cast<double>(k);", "}"]


def test_reasons_name_what_they_explain(tmp_path):
    f = tmp_path / "m.py"
    f.write_text("import numpy as np\nfrom qtrans.typing import Mat, Vec\n\n"
                 "def g(x: Vec) -> None:\n    x[0] = 1.0\n\n"
                 "def f(A: Mat, v: Vec, i: int, n: int) -> float:\n"
                 "    g(v * 2)\n"
                 "    B = A - v\n"
                 "    return float(B[0, 0]) + v[i] + n // 3 + n * n\n", encoding="utf-8")
    view = explain_file(f, "f")
    kinds = {r.kind for row in view.functions[0].rows for r in row.reasons}
    assert {"sharing", "broadcasting", "indexing", "division", "int64", "passing"} <= kinds
    text = to_text(view, 160)
    assert "qt::lvalue" in text and "NumPy broadcasting" in text and "counts it from the end" in text


def test_refused_function_shows_reason_at_its_line(tmp_path):
    f = tmp_path / "m.py"
    f.write_text("import numpy as np\nfrom qtrans.typing import Vec\n\n"
                 "def bad(a: Vec) -> float:\n    b = a\n    b[0] = 1.0\n    return float(b[0])\n", encoding="utf-8")
    view = explain_file(f)
    fv = view.functions[0]
    assert not fv.translated and fv.line == 5 and "share memory" in fv.reason
    assert [r.py_first for r in fv.rows if r.error] == [5]
    assert "b = a.copy()" in fv.hint
    info = inspect_file(f)[0]
    assert not info.translated and info.reason_line == 5


def test_html_is_escaped_and_complete(tmp_path):
    f = tmp_path / "m.py"
    f.write_text("import numpy as np\nfrom qtrans.typing import Vec\n\n"
                 "def f(v: Vec) -> float:\n    # <script>alert(1)</script> & \"quotes\"\n    return float(v[0])\n",
                 encoding="utf-8")
    page = to_html(explain_file(f))
    assert "<script>alert" not in page and "&lt;script&gt;alert" in page

    class P(html.parser.HTMLParser):
        sections = 0

        def handle_starttag(self, tag, attrs):
            P.sections += tag == "section"
    P().feed(page)
    assert P.sections == 2  # the module (imports) and f


def test_json_and_cli(tmp_path, capsys):
    f = ROOT / "examples" / "portfolio.py"
    data = json.loads(to_json(explain_file(f, "sample_cov")))
    assert [x["name"] for x in data["functions"]] == ["sample_cov"]
    row = data["functions"][0]["rows"][0]
    assert set(row) >= {"py_first", "py_last", "python", "cpp", "reasons"}
    out = tmp_path / "p.html"
    assert cli(["explain", f"{f}:portfolio_vol", "--html", str(out)]) == 0
    assert "portfolio_vol" in out.read_text(encoding="utf-8")
    assert cli(["inspect", str(f), "--json"]) == 0
    info = json.loads(capsys.readouterr().out.split("\n", 1)[1])
    sc = next(x for x in info["functions"] if x["name"] == "sample_cov")
    assert sc["params"][0]["mode"] == "cref" and "only reads" in sc["params"][0]["why"]


def test_inspect_reports_modes_linalg_and_differences():
    infos = {i.name: i for i in inspect_file(ROOT / "examples" / "black_litterman_app.py")}
    np_ = infos["nearest_psd"]
    assert np_.translated and "eigh" in np_.linalg
    assert np_.params[0]["mode"] == "value"
    assert any("eigenvector" in d["note"] for d in np_.differences)


def test_every_rule_is_valid_and_documented():
    for rx, kind, why in _COMPILED:
        assert kind and why.strip() and rx.pattern
        assert not why.endswith(".")  # shown as a clause after the construct


@pytest.mark.skipif(not HAVE_CXX, reason="no C++ compiler")
def test_doctor_passes_here_and_reports_what_is_missing(monkeypatch):
    checks = {c.name: c for c in doctor.run()}
    assert checks["translate, compile, verify"].status == doctor.OK, checks["translate, compile, verify"].detail
    assert checks["C++17"].status == doctor.OK

    import qtrans.verify as V

    def no_eigen(explicit=None):
        raise RuntimeError("Array code needs the Eigen headers, which weren't found.")
    monkeypatch.setattr(V, "find_eigen", no_eigen)
    checks = {c.name: c for c in doctor.run()}
    assert checks["Eigen"].status == doctor.FAIL and "Eigen" in checks["Eigen"].fix
    assert "translate, compile, verify" not in checks  # not tried without Eigen
    assert "1 problem(s) to fix" in doctor.to_text(list(checks.values()))


def test_doctor_catches_a_compiler_that_cannot_build(monkeypatch, tmp_path):
    fake = tmp_path / ("g++.bat" if __import__("os").name == "nt" else "g++")
    fake.write_text("@exit /b 1\n" if fake.suffix else "#!/bin/sh\nexit 1\n", encoding="utf-8")
    fake.chmod(0o755)
    monkeypatch.setenv("CXX", str(fake))
    checks = {c.name: c for c in doctor.run()}
    assert checks["C++17"].status == doctor.FAIL


def test_text_views_survive_a_console_that_cannot_show_their_marks():
    """A Windows console on cp1252 has no ✓ ✗ │ ↳: the commands print ASCII there instead of
    failing with UnicodeEncodeError (CI found this on Windows)."""
    import os
    import subprocess
    import sys
    env = dict(os.environ, PYTHONIOENCODING="cp1252")
    f = str(ROOT / "examples" / "portfolio.py")
    for args in (["inspect", f], ["explain", f + ":sample_cov"]):
        r = subprocess.run([sys.executable, "-m", "qtrans", *args], capture_output=True, env=env, cwd=ROOT)
        assert r.returncode == 0, r.stderr.decode("utf-8", "replace")[-500:]
        assert b"|" in r.stdout or b"ok" in r.stdout


def test_one_version_everywhere():
    """pyproject.toml's version is the one the headers carry (`generated by qtrans X`)."""
    from qtrans.translator import VERSION
    text = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
    assert re.search(r'^version = "([^"]+)"', text, re.M).group(1) == VERSION
    assert f"generated by qtrans {VERSION}" in translate_source("def f(x: float) -> float:\n    return x\n", "m").header
