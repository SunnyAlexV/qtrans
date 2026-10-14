"""verify's bench=N (1.0.1): N rounds, Python and C++ taking turns, medians reported; the calls
timed in each round recorded (1.0.2)."""
import statistics
import textwrap

import qtrans.cli as cli
from qtrans.verify import verify_file

SRC = textwrap.dedent('''
    def pair(x: float, y: float) -> tuple[float, float]:
        return x + y, x * y
    QTRANS_CASES = {"pair": [(1.5, 2.0)]}
''')


def test_rounds_and_medians(tmp_path):
    p = tmp_path / "m.py"
    p.write_text(SRC)
    rep = verify_file(p, n_random=5, bench=3, build_dir=tmp_path / "b")
    (fc,) = rep.functions
    assert fc.count("mismatch") == 0
    assert len(fc.py_rounds) == 3 and len(fc.cpp_rounds) == 3
    assert fc.py_ns == statistics.median(fc.py_rounds)
    assert fc.cpp_ns == statistics.median(fc.cpp_rounds)
    assert fc.bench_case is not None
    # the calls timed in each round are recorded (1.0.2): runs of 1, 4, 16, ... calls
    assert len(fc.py_reps) == 3 and len(fc.cpp_reps) == 3
    assert all(n >= 1 and (n & (n - 1)) == 0 and n.bit_length() % 2 == 1 for n in fc.py_reps + fc.cpp_reps)


def test_cli_bench_optional_rounds(tmp_path, monkeypatch):
    seen = []

    class Rep:
        ok = True

        def to_text(self):
            return ""

    def fake(inputs, **kw):
        seen.append(kw["bench"])
        return Rep()

    monkeypatch.setattr(cli, "verify_project", fake)
    p = tmp_path / "m.py"
    p.write_text(SRC)
    for extra, want in (([], 0), (["--bench"], 1), (["--bench", "--rounds", "5"], 5), (["--rounds", "5"], 0)):
        assert cli.main(["verify", str(p), *extra]) == 0
        assert seen[-1] == want
