"""docs/SPEC.md is checked, example by example, so it can't drift from what qtrans does.

A fenced block whose info string starts with `python spec` is one example:
- `python spec translated "text" ...`: it translates with nothing refused, and the C++ holds each text;
- `python spec refused "text"`: something in it is refused, and a reason holds the text;
- `python spec program`: it compiles and runs, and prints exactly what Python prints."""
import re
import shlex
from pathlib import Path

import pytest

from qtrans.translator import translate_source
from qtrans.verify import verify_file
from tests.test_semantics import HAVE_CXX

SPEC = Path(__file__).resolve().parent.parent / "docs" / "SPEC.md"
BLOCK = re.compile(r"^```python spec (.*?)\n(.*?)^```", re.S | re.M)


def _examples():
    text = SPEC.read_text(encoding="utf-8")
    out = []
    for m in BLOCK.finditer(text):
        words = shlex.split(m.group(1))
        line = text.count("\n", 0, m.start()) + 1
        out.append(pytest.param(words[0], words[1:], m.group(2), id=f"line{line}-{words[0]}"))
    return out


EXAMPLES = _examples()


def test_the_spec_has_examples_of_each_kind():
    kinds = {p.values[0] for p in EXAMPLES}
    assert kinds == {"translated", "refused", "program"}, kinds
    assert len(EXAMPLES) >= 30


@pytest.mark.parametrize("kind, texts, source", EXAMPLES)
def test_spec_example(kind, texts, source, tmp_path):
    if kind == "program":
        if not HAVE_CXX:
            pytest.skip("no C++ compiler")
        f = tmp_path / "spec_program.py"
        f.write_text(source, encoding="utf-8")
        rep = verify_file(f, n_random=5, build_dir=tmp_path / "build")
        p = rep.program
        assert p is not None and p.identical, rep.to_text() + "\n--- Python:\n" + (p.py_out if p else "") + \
            "\n--- C++:\n" + (p.cpp_out if p else "")
        return
    r = translate_source(source, "m")
    reasons = [f"{f.name}: {f.reason}" for f in r.report.rejected]
    if kind == "translated":
        assert not reasons, reasons
        code = r.header + (r.program or "")
        for t in texts:
            assert t in code, f"{t!r} not in the C++:\n{code}"
    elif kind == "refused":
        assert reasons, "nothing was refused:\n" + r.header
        for t in texts:
            assert any(t in x for x in reasons), (t, reasons)
    else:
        raise AssertionError(f"unknown spec example kind {kind!r}")
