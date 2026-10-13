"""A UTF-8 file that starts with a byte-order mark (Windows Notepad and Windows PowerShell 5's
Out-File write one) is read as Python reads it: the mark is not part of the code."""
import pytest

from qtrans.live import Document
from qtrans.translator import translate_file
from qtrans.verify import verify_file
from tests.test_semantics import HAVE_CXX

SOURCE = 'def f(x: float) -> float:\n    return x * 2.0\n\n\nprint("straße", f(1.5))\n'


def _bom_file(tmp_path, crlf=False):
    p = tmp_path / "bom_demo.py"
    text = SOURCE.replace("\n", "\r\n") if crlf else SOURCE
    p.write_bytes(b"\xef\xbb\xbf" + text.encode("utf-8"))
    return p


def test_a_file_with_a_byte_order_mark_translates(tmp_path):
    r = translate_file(_bom_file(tmp_path), tmp_path / "out")
    assert all(f.translated for f in r.report.functions), r.report.to_text()
    assert "﻿" not in r.header + r.program


@pytest.mark.skipif(not HAVE_CXX, reason="no C++ compiler")
def test_a_file_with_a_byte_order_mark_verifies(tmp_path):
    rep = verify_file(_bom_file(tmp_path), n_random=20, build_dir=tmp_path / "build")
    assert rep.ok and rep.program is not None and rep.program.identical, rep.to_text()


def test_the_live_editor_keeps_the_mark_and_line_endings(tmp_path):
    p = _bom_file(tmp_path, crlf=True)
    doc = Document(p)
    text, mtime = doc.read()
    assert text == SOURCE  # what the editor shows: no mark, \n
    ok, _, _ = doc.write(text.replace("2.0", "3.0"), mtime)
    assert ok and p.read_bytes() == b"\xef\xbb\xbf" + SOURCE.replace("2.0", "3.0").replace("\n", "\r\n").encode("utf-8")


def test_the_build_command_is_the_one_verify_used(monkeypatch, tmp_path):
    """`qtrans translate` prints how to build a program: the compiler qtrans found, by its full
    path when it isn't on the PATH (winlibs unzipped on the Desktop), and no fused multiply-add."""
    import qtrans.verify as V
    from pathlib import Path
    gxx = tmp_path / "Desktop" / "mingw64" / "bin" / "g++"
    monkeypatch.setattr(V, "find_compiler", lambda explicit=None: V.Compiler(str(gxx), "gnu"))
    monkeypatch.setenv("PATH", str(tmp_path / "empty"))
    hint = V.build_hint(Path("out") / "demo.cpp", "demo", Path("out"))
    assert hint.startswith(str(gxx) + " -std=c++17 -O2 -ffp-contract=off")
    assert hint.endswith("-o demo" + (".exe" if V.os.name == "nt" else ""))
    monkeypatch.setattr(V, "find_compiler", lambda explicit=None: V.Compiler("cl", "msvc"))
    assert V.build_hint(Path("out") / "demo.cpp", "demo", Path("out")).startswith("cl /nologo /std:c++17 /EHsc /O2 /fp:precise")
