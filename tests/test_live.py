"""qtrans live: the server (translate as you type, keep translating while a function is half
written, save without losing an edit made elsewhere, only answer this computer and the page it
served) and, when Playwright and a browser are available, the page itself."""
import json
import re
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path

import pytest

from qtrans.live import salvage, serve, translate_text
from tests.test_semantics import HAVE_CXX

ROOT = Path(__file__).resolve().parents[1]
SAMPLE = ("import numpy as np\nfrom qtrans.typing import Vec\n\n\n"
          "def norm2(v: Vec) -> float:\n    return float(v @ v)\n\n\n"
          "def halve(n: int) -> int:\n    return n // 2\n")


@pytest.fixture
def live(tmp_path):
    f = tmp_path / "model.py"
    f.write_text(SAMPLE, encoding="utf-8")
    box = {}
    started = threading.Event()

    def ready(url, server):
        box["url"], box["server"] = url, server
        started.set()
    t = threading.Thread(target=serve, args=(f,), kwargs={"open_browser": False, "ready": ready}, daemon=True)
    t.start()
    assert started.wait(10)
    page = urllib.request.urlopen(box["url"]).read().decode("utf-8")
    box["token"] = re.search(r'const TOKEN = "([^"]+)"', page).group(1)
    box["file"], box["page"] = f, page
    yield box
    box["server"].shutdown()


def call(box, path, body=None, token=True, host=None):
    req = urllib.request.Request(box["url"].rstrip("/") + path, method="POST" if body is not None else "GET",
                                 data=json.dumps(body).encode() if body is not None else None)
    if token:
        req.add_header("X-Qtrans-Token", box["token"])
    if host:
        req.add_header("Host", host)
    try:
        with urllib.request.urlopen(req) as r:
            return r.status, json.loads(r.read())
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read() or b"{}")


def test_only_this_computer_and_the_page_it_served(live):
    assert call(live, "/api/file", token=False)[0] == 403        # another page can't read the file
    assert call(live, "/api/save", {"text": "x"}, token=False)[0] == 403
    assert call(live, "/api/file", host="evil.example")[0] == 403  # DNS rebinding
    status, body = call(live, "/api/file")
    assert status == 200 and body["text"] == SAMPLE and isinstance(body["mtime"], str)
    assert live["file"].read_text(encoding="utf-8") == SAMPLE


def test_translates_the_editor_text(live):
    status, d = call(live, "/api/translate", {"text": SAMPLE})
    assert status == 200 and d["ok"]
    assert len(d["map"]) == len(d["header"])
    names = {f["name"]: f for f in d["functions"]}
    assert names["norm2"]["translated"] and names["halve"]["translated"]
    lines = [h for h, s in zip(d["header"], d["map"]) if s and s[0] == 10]
    assert lines == ["    return qt::floordiv(n, 2LL);", "}"]
    row = next(r for r in names["halve"]["rows"] if r["first"] == 10)
    assert any(x["kind"] == "division" for x in row["reasons"])


def test_a_half_written_function_leaves_the_rest_translated():
    text = SAMPLE + "\n\ndef draft(x: float) -> float:\n    return (x +\n"
    source, editing = salvage(text)
    assert editing[:2] == (13, 15) and "never closed" in editing[3]
    d = translate_text(text, ROOT / "model.py")
    assert d["ok"] and d["editing"]["first"] == 13
    assert {f["name"] for f in d["functions"] if f["translated"]} == {"norm2", "halve"}
    # a def with no body yet: the message names the def's line, and that block is set aside
    d = translate_text(SAMPLE + "\n\ndef empty(x: float) -> float:\n\n\ndef ok(y: float) -> float:\n    return y\n",
                       ROOT / "model.py")
    assert d["ok"] and {f["name"] for f in d["functions"] if f["translated"]} == {"norm2", "halve", "ok"}
    # broken at module level: nothing to translate, the last good C++ stays on the page
    d = translate_text("x = (\n", ROOT / "model.py")
    assert not d["ok"] and d["error"]["line"] == 1


def test_saving_never_loses_an_edit_made_elsewhere(live):
    _, f = call(live, "/api/file")
    status, s = call(live, "/api/save", {"text": SAMPLE + "# one\n", "mtime": f["mtime"]})
    assert status == 200 and live["file"].read_text(encoding="utf-8").endswith("# one\n")
    time.sleep(0.02)
    live["file"].write_text(SAMPLE + "# from another editor\n", encoding="utf-8")
    status, c = call(live, "/api/save", {"text": SAMPLE + "# two\n", "mtime": s["mtime"]})
    assert status == 409 and c["text"].endswith("# from another editor\n")  # the page asks which to keep
    assert live["file"].read_text(encoding="utf-8").endswith("# from another editor\n")
    status, _ = call(live, "/api/save", {"text": SAMPLE + "# two\n", "mtime": None})  # "keep mine"
    assert status == 200 and live["file"].read_text(encoding="utf-8").endswith("# two\n")


def test_windows_line_endings_are_kept(tmp_path):
    from qtrans.live import Document
    f = tmp_path / "w.py"
    f.write_bytes(SAMPLE.replace("\n", "\r\n").encode())
    doc = Document(f)
    text, mtime = doc.read()
    assert "\r" not in text
    assert doc.write(text + "# x\n", mtime)[0]
    assert f.read_bytes().endswith(b"# x\r\n") and b"\r\n" in f.read_bytes()


@pytest.mark.skipif(not HAVE_CXX, reason="no C++ compiler")
def test_verify_from_the_page(live):
    status, v = call(live, "/api/verify", {"text": SAMPLE, "cases": 10})
    assert status == 200 and v["ok"], v.get("text")
    assert {f["name"] for f in v["functions"]} == {"norm2", "halve"}
    _, v = call(live, "/api/verify", {"text": SAMPLE + "def x(:\n"})
    assert not v["ok"] and "finish the code" in v["text"]


def test_the_page_in_a_browser(live):
    pw = pytest.importorskip("playwright.sync_api")
    with pw.sync_playwright() as p:
        try:
            browser = p.chromium.launch()
        except Exception as e:  # noqa: BLE001 - no browser installed here
            pytest.skip(f"no browser for Playwright: {e}")
        page = browser.new_page(viewport={"width": 1400, "height": 900})
        errors = []
        page.on("pageerror", lambda e: errors.append(str(e)))
        page.goto(live["url"])
        page.wait_for_selector("#cpp .cl")
        page.wait_for_function("document.getElementById('status').textContent.includes('translated')")
        # the cursor on a Python line lights its C++
        page.evaluate("goToLine(10)")
        assert page.locator("#cpp .cl.active .ct").all_inner_texts()[0].strip() == "return qt::floordiv(n, 2LL);"
        assert "rounds toward minus infinity" in page.inner_text("#why")
        # type a new function: its C++ appears without pressing anything
        page.evaluate("src.focus(); src.setSelectionRange(src.value.length, src.value.length)")
        page.keyboard.type("\n\ndef cube(x: float) -> float:\nreturn x * x * x")
        page.wait_for_function("[...document.querySelectorAll('#cpp .ct')].some(e => e.textContent.includes('double cube(double x)'))")
        assert page.locator("#cpp .cl.active .ct").all_inner_texts()[0].strip() == "return x * x * x;"
        # Ctrl+S writes the file
        page.keyboard.press("Control+s")
        page.wait_for_function("document.getElementById('dirty').textContent === 'Saved'")
        assert "def cube(x: float) -> float:\n    return x * x * x" in live["file"].read_text(encoding="utf-8")
        assert errors == []
        browser.close()


def test_verify_without_a_compiler_says_how_to_get_one(monkeypatch, tmp_path):
    import qtrans.verify as V
    from qtrans.live import verify_text

    def none(explicit=None):
        raise RuntimeError("No C++ compiler found. On Windows the simplest is GCC from winlibs.com")
    monkeypatch.setattr(V, "find_compiler", none)
    v = verify_text(SAMPLE, tmp_path / "model.py")
    assert v["setup"] and not v["ok"] and "winlibs" in v["text"]


def test_a_mingw_gcc_off_the_path_is_found_on_windows(monkeypatch, tmp_path):
    import os
    import qtrans.verify as V
    exe = tmp_path / "mingw64" / "bin" / "g++.exe"
    exe.parent.mkdir(parents=True)
    exe.write_text("")
    monkeypatch.setattr(V.shutil, "which", lambda name: None)
    monkeypatch.setattr(V.os, "name", "nt")
    monkeypatch.setattr(V, "_windows_gcc_candidates", lambda: iter([exe]))
    monkeypatch.setenv("PATH", "/usr/bin")
    monkeypatch.delenv("CXX", raising=False)
    cxx = V.find_compiler()
    assert cxx.path == str(exe) and cxx.kind == "gnu"
    assert os.environ["PATH"].split(os.pathsep)[0] == str(exe.parent)  # its DLLs, for the programs it builds


def test_a_program_shows_its_main():
    """Typed in as a script, it is a program: the page shows the whole .cpp, main() and all,
    each line next to the Python it came from, and a line left out says why."""
    d = translate_text("A = 1+1\nprint(A)\n", ROOT / "model.py")
    assert d["ok"] and d["file"] == "model.cpp"
    by_line = {}
    for text, span in zip(d["header"], d["map"]):
        if span:
            by_line.setdefault(span[0], []).append(text.strip())
    assert by_line == {1: ["long long A = 1LL + 1;"], 2: ["std::cout << A << '\\n';"]}
    prog = next(f for f in d["functions"] if f["name"] == "(program)")
    assert prog["translated"] and prog["spans"] == [[1, 1], [2, 2]]
    d = translate_text("from .x import y\nprint(1)\n", ROOT / "model.py")
    assert any(r["skipped"] and r["first"] == 1 and "relative import" in r["reasons"][0]["text"] for r in d["module"])


@pytest.mark.skipif(not HAVE_CXX, reason="no C++ compiler")
def test_verify_runs_the_program_both_ways(live):
    status, v = call(live, "/api/verify", {"text": "import math\nx = math.sqrt(2.0)\nprint(f'{x:.4f}', x)\n"})
    assert status == 200 and v["ok"], v.get("text")
    p = v["program"]
    assert p["identical"] and p["python"] == p["cpp"] == "1.4142 1.4142135623730951\n" and p["py_code"] == 0
