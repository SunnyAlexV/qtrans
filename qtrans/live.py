"""`qtrans live`: edit Python on the left, see its C++ on the right, updated as you type.

    python -m qtrans live FILE.py [--port N] [--no-browser]

A small web server on this computer (127.0.0.1 only) serves an editor page. Each pause in
typing sends the editor's text here; it is translated as `qtrans translate` would (the
project modules it imports included) and the page shows the C++ (the header, or for a program
the whole .cpp with its main()), the C++ of the statement under the cursor lit, and the reasons
for it.
While one function is half-written (it doesn't parse yet), the rest of the module is still
translated: that function's lines are left out until they parse, and the page says so.

Ctrl+S saves to the file. Edits made to the file in another editor show up on the page; if the
page has unsaved changes too, it asks which to keep. "Verify" compiles the module and checks it
against Python (`qtrans verify`, 50 random cases a function); a program is also run both ways
and what each printed is shown side by side.

Every request carries a random token the page was served with, and only requests addressed to
this computer are answered, so another web page open in the browser can't read or write files
through the server.
"""
from __future__ import annotations

import ast
import json
import re
import secrets
import shutil
import tempfile
import threading
import time
import webbrowser
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

_TOPLEVEL = re.compile(r"^(?:async\s+def|def|class|@)\b|^@")


# --- translating the editor's text --------------------------------------------------------------
def _sibling_imports(tree, folder: Path) -> set:
    names = set()
    for n in ast.walk(tree):
        if isinstance(n, ast.ImportFrom) and n.module and n.level == 0:
            names.add(n.module.split(".")[0])
        elif isinstance(n, ast.Import):
            names |= {a.name.split(".")[0] for a in n.names}
    return {m for m in names if (folder / f"{m}.py").is_file()}


def _chunks(lines):
    """Top-level blocks: (first, last) 1-based line ranges, each starting at a def, class or
    decorator in column 0 (decorators join the def they decorate)."""
    starts = []
    for i, line in enumerate(lines, 1):
        if _TOPLEVEL.match(line) and not (starts and starts[-1] == i - 1 and lines[i - 2].startswith("@")):
            starts.append(i)
    out = []
    for k, s in enumerate(starts):
        e = (starts[k + 1] - 1) if k + 1 < len(starts) else len(lines)
        out.append((s, e))
    return out


def salvage(text: str):
    """(text that parses, (first, last, line, message)) or (None, (...)) when it can't be saved.
    A block that doesn't parse yet (being typed) is blanked, line numbers kept, so the rest of
    the module still translates. Up to five blocks are set aside this way: the one holding the
    error, and the one a message points back to ("expected an indented block after function
    definition on line 102"). (text, None) when the text parses as it is."""
    try:
        ast.parse(text)
        return text, None
    except SyntaxError as e:
        first_error = (e.lineno or 1, e.msg)
    lines = text.split("\n")
    chunks = _chunks(lines)
    blanked, left_out = list(lines), []
    for _ in range(5):
        try:
            ast.parse("\n".join(blanked))
            break
        except SyntaxError as e:
            # a message that names an earlier line ("... after function definition on line 102")
            # points at the block to set aside; otherwise the block holding the error
            at = [int(m) for m in re.findall(r"on line (\d+)", e.msg or "")] or [e.lineno or 1]
            hit = [c for c in chunks if any(c[0] <= n <= c[1] for n in at) and c not in left_out]
            if not hit:
                return None, (first_error[0], first_error[0], first_error[0], first_error[1])
            for c in hit:
                left_out.append(c)
                blanked[c[0] - 1:c[1]] = [""] * (c[1] - c[0] + 1)
    else:
        return None, (first_error[0], first_error[0], first_error[0], first_error[1])
    first = min(c[0] for c in left_out)
    last = max(c[1] for c in left_out)
    return "\n".join(blanked), (first, last, first_error[0], first_error[1])


def translate_text(text: str, path: Path) -> dict:
    """The editor's text, translated as if it were the file at path: the page's data."""
    from .explain import explain_result
    from .translator import translate_file
    t0 = time.perf_counter()
    source, editing = salvage(text)
    if source is None:
        first, last, line, msg = editing
        return {"ok": False, "error": {"line": line, "message": msg, "first": first, "last": last},
                "ms": round(1000 * (time.perf_counter() - t0), 1)}
    work = Path(tempfile.mkdtemp(prefix="qtrans_live_"))
    try:
        target = work / path.name
        target.write_text(source, encoding="utf-8")
        from .overrides import path_for
        if path_for(path).is_file():  # your own C++ for some functions, as it is on disk
            shutil.copy(path_for(path), path_for(target))
        todo, seen = [target], set()
        while todo:  # the project modules it imports, as they are on disk
            f = todo.pop()
            try:
                tree = ast.parse(f.read_text(encoding="utf-8-sig"))
            except SyntaxError:
                continue
            for m in _sibling_imports(tree, path.parent) - seen:
                seen.add(m)
                shutil.copy(path.parent / f"{m}.py", work / f"{m}.py")
                todo.append(work / f"{m}.py")
        try:
            result = translate_file(target)
        except Exception as e:  # noqa: BLE001 - shown in the page, never a crash of the server
            return {"ok": False, "error": {"line": getattr(e, "lineno", None) or 1,
                                           "message": f"{type(e).__name__}: {e}"},
                    "ms": round(1000 * (time.perf_counter() - t0), 1)}
        view = explain_result(result, source, str(path))
    finally:
        shutil.rmtree(work, ignore_errors=True)
    funcs = []
    for f in view.functions:
        funcs.append({
            "name": f.name, "first": f.first, "last": f.last, "translated": f.translated, "override": f.override,
            "spans": [list(x) for x in f.spans] or [[f.first, f.last]],
            "reason": f.reason, "line": f.line, "hint": f.hint,
            "rows": [{"first": r.py_first, "last": r.py_last,
                      "reasons": [{"kind": x.kind, "text": x.text} for x in r.reasons]} for r in f.rows],
        })
    module_rows = [{"first": r.py_first, "last": r.py_last, "skipped": any(x.kind == "skipped" for x in r.reasons),
                    "reasons": [{"kind": x.kind, "text": x.text} for x in r.reasons]} for r in view.module_rows]
    program = result.program is not None
    lines = (result.program if program else result.header).split("\n")
    if lines and lines[-1] == "":
        lines.pop()
    lmap = (result.program_map if program else result.line_map) or []
    out = {"ok": True, "header": lines, "map": [list(s) if s else None for s in lmap],
           "file": f"{path.stem}.cpp" if program else f"{path.stem}.hpp",
           "functions": funcs, "module": module_rows, "notes": view.general_notes,
           "ms": round(1000 * (time.perf_counter() - t0), 1)}
    if editing:
        first, last, line, msg = editing
        out["editing"] = {"first": first, "last": last, "line": line, "message": msg}
    return out


def verify_text(text: str, path: Path, cases: int = 50) -> dict:
    """Compile the editor's text and check it against Python (qtrans verify)."""
    from .verify import find_compiler, find_eigen, verify_file
    try:
        ast.parse(text)
    except SyntaxError as e:
        return {"ok": False, "functions": [], "text": f"line {e.lineno}: {e.msg}: finish the code, then verify"}
    work = Path(tempfile.mkdtemp(prefix="qtrans_live_verify_"))
    try:
        for f in path.parent.glob("*.py"):
            if f.name != path.name:
                shutil.copy(f, work / f.name)
        target = work / path.name
        target.write_text(text, encoding="utf-8")
        try:
            find_compiler()
            find_eigen() if re.search(r"\bnumpy\b|\bnp\.", text) else None
        except RuntimeError as e:  # nothing to compile with: a setup problem, not a failed check
            return {"ok": False, "setup": True, "functions": [], "text": str(e)}
        try:
            rep = verify_file(target, n_random=cases, build_dir=work / "build")
        except Exception as e:  # noqa: BLE001
            return {"ok": False, "text": f"{type(e).__name__}: {e}"}
        rows = [{"name": f.name, "cases": len(f.cases), "match": f.count("match"), "illcond": f.count("illcond"),
                 "known": f.count("known"), "mismatch": f.count("mismatch")} for f in rep.functions]
        out = {"ok": rep.ok, "functions": rows, "warnings": rep.warnings[:10],
               "text": rep.to_text().replace(str(work), str(path.parent))}
        if rep.program is not None:
            p = rep.program
            out["program"] = {"ok": p.ok, "identical": p.identical, "line": p.line, "rounding": p.rounding,
                              "python": p.py_out[-20000:], "cpp": p.cpp_out[-20000:], "py_code": p.py_code,
                              "cpp_code": p.cpp_code, "error": p.error,
                              "py_err": p.py_err[-2000:].replace(str(work), str(path.parent)), "cpp_err": p.cpp_err[-2000:]}
        return out
    finally:
        shutil.rmtree(work, ignore_errors=True)


# --- the file on disk ---------------------------------------------------------------------------------
class Document:
    """The file being edited: read and written with its own line endings, edits on disk noticed."""

    def __init__(self, path: Path):
        self.path = path
        self.lock = threading.Lock()
        if not path.exists():
            path.write_text("import numpy as np\nfrom qtrans.typing import Mat, Vec\n\n\n"
                            "def f(x: float) -> float:\n    return x * x\n", encoding="utf-8")
        raw = path.read_bytes()
        self.bom = raw.startswith(b"\xef\xbb\xbf")  # Notepad and Windows PowerShell 5 start UTF-8 files with one
        self.crlf = b"\r\n" in raw

    def read(self):
        with self.lock:
            raw = self.path.read_bytes().decode("utf-8-sig")
            return raw.replace("\r\n", "\n"), str(self.path.stat().st_mtime_ns)

    def write(self, text: str, base_mtime: int | None):
        """-> (ok, mtime, text on disk when it changed since base_mtime)."""
        with self.lock:
            now = str(self.path.stat().st_mtime_ns)  # a string: nanoseconds don't fit a JS number
            if base_mtime is not None and now != str(base_mtime):
                return False, now, self.path.read_bytes().decode("utf-8-sig").replace("\r\n", "\n")
            data = text.replace("\r\n", "\n")
            if self.crlf:
                data = data.replace("\n", "\r\n")
            tmp = self.path.with_name(self.path.name + ".qtrans-live.tmp")
            tmp.write_bytes((b"\xef\xbb\xbf" if self.bom else b"") + data.encode("utf-8"))  # kept as it was
            tmp.replace(self.path)  # never a half-written file
            return True, str(self.path.stat().st_mtime_ns), None


# --- the server ------------------------------------------------------------------------------------------
def make_handler(doc: Document, token: str, page: bytes, port_ref: list):
    class Handler(BaseHTTPRequestHandler):
        server_version = "qtrans-live"

        def log_message(self, *args):  # quiet: the terminal shows only the address
            pass

        def _host_ok(self) -> bool:
            host = (self.headers.get("Host") or "").split(":")[0]
            return host in ("127.0.0.1", "localhost")

        def _send(self, status, body: bytes, ctype="application/json; charset=utf-8"):
            self.send_response(status)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.end_headers()
            self.wfile.write(body)

        def _json(self, status, obj):
            self._send(status, json.dumps(obj).encode("utf-8"))

        def _authorised(self) -> bool:
            if not self._host_ok():
                self._json(HTTPStatus.FORBIDDEN, {"error": "only requests to 127.0.0.1 / localhost are answered"})
                return False
            if not secrets.compare_digest(self.headers.get("X-Qtrans-Token", ""), token):
                self._json(HTTPStatus.FORBIDDEN, {"error": "missing or wrong token: reload the page from the "
                                                           "address qtrans live printed"})
                return False
            return True

        def do_GET(self):
            if self.path.split("?")[0] in ("/", "/index.html"):
                if not self._host_ok():
                    return self._json(HTTPStatus.FORBIDDEN, {"error": "only localhost"})
                return self._send(HTTPStatus.OK, page, "text/html; charset=utf-8")
            if self.path.startswith("/api/file"):
                if not self._authorised():
                    return
                text, mtime = doc.read()
                return self._json(HTTPStatus.OK, {"path": str(doc.path), "name": doc.path.name,
                                                  "text": text, "mtime": mtime})
            if self.path.startswith("/api/mtime"):
                if not self._authorised():
                    return
                return self._json(HTTPStatus.OK, {"mtime": str(doc.path.stat().st_mtime_ns)})
            self._json(HTTPStatus.NOT_FOUND, {"error": "not found"})

        def do_POST(self):
            if not self._authorised():
                return
            n = int(self.headers.get("Content-Length") or 0)
            if n > 20_000_000:
                return self._json(HTTPStatus.REQUEST_ENTITY_TOO_LARGE, {"error": "too large"})
            try:
                body = json.loads(self.rfile.read(n) or b"{}")
            except ValueError:
                return self._json(HTTPStatus.BAD_REQUEST, {"error": "not JSON"})
            text = body.get("text")
            if not isinstance(text, str):
                return self._json(HTTPStatus.BAD_REQUEST, {"error": "no text"})
            if self.path == "/api/translate":
                return self._json(HTTPStatus.OK, translate_text(text, doc.path))
            if self.path == "/api/save":
                ok, mtime, disk = doc.write(text, body.get("mtime"))
                if not ok:
                    return self._json(HTTPStatus.CONFLICT, {"error": "changed on disk", "mtime": mtime, "text": disk})
                return self._json(HTTPStatus.OK, {"mtime": mtime})
            if self.path == "/api/verify":
                return self._json(HTTPStatus.OK, verify_text(text, doc.path, int(body.get("cases") or 50)))
            self._json(HTTPStatus.NOT_FOUND, {"error": "not found"})

    return Handler


class _Server(ThreadingHTTPServer):
    daemon_threads = True

    def server_bind(self):
        # HTTPServer.server_bind looks up the host's full DNS name (socket.getfqdn), which on
        # macOS can wait seconds for a DNS timeout; the page only ever uses 127.0.0.1
        import socketserver
        socketserver.TCPServer.server_bind(self)
        self.server_name, self.server_port = "127.0.0.1", self.server_address[1]


def serve(path, port: int = 0, open_browser: bool = True, ready=None):
    """Run the server until interrupted. `ready(url, server)` is called once it listens."""
    PAGE = (Path(__file__).with_name("live_page.html")).read_text(encoding="utf-8-sig")
    path = Path(path).resolve()
    doc = Document(path)
    token = secrets.token_urlsafe(24)
    page = PAGE.replace("@@TOKEN@@", token).replace("@@NAME@@", path.name).encode("utf-8")
    port_ref = [port]
    server = _Server(("127.0.0.1", port), make_handler(doc, token, page, port_ref))
    url = f"http://127.0.0.1:{server.server_address[1]}/"
    if ready:
        ready(url, server)
    else:
        print(f"qtrans live: {path.name} at {url}  (Ctrl+C to stop)")
    if open_browser:
        threading.Timer(0.3, lambda: webbrowser.open(url)).start()
    try:
        server.serve_forever(poll_interval=0.2)
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
    return 0
