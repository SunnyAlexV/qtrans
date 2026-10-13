"""Shows what a Windows console window displays for a translated program that prints text beyond
ASCII, with and without qt::py::windows_console, next to what Python prints. Run on Linux with
MinGW and Wine (apt install g++-mingw-w64-x86-64-posix wine64); the console is Wine's, given a
terminal through `script`, so GetConsoleMode sees a console as on Windows.

    python tools/windows_console_check.py

On Windows itself: `qtrans translate demo.py -o out`, build out/demo.cpp, and run it in a
PowerShell or cmd window.
"""
import os
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from qtrans.translator import LIST_HEADER, PRINT_HEADER, RUNTIME_HEADER, UNICODE_HEADER, translate_source  # noqa: E402

DEMO = 'line = " AAPL, 190.5 ,straße "\nparts = [p.strip() for p in line.split(",")]\n' \
       'print(parts, float(parts[1]), parts[2].upper(), "ΟΔΟΣ".lower(), "١٢٣", repr("a\\u00a0b"))\n'


def shown(exe, work):
    """What the console shows, read back from the terminal (control sequences removed)."""
    env = dict(os.environ, WINEDEBUG="-all", LANG="C.UTF-8")
    raw = subprocess.run(["script", "-qc", f"wine {exe}", "/dev/null"], capture_output=True, cwd=work, env=env).stdout
    raw = re.sub(rb"\x1b\[K\x1b\[1C", b" ", raw)  # Wine draws a space as erase + cursor right
    text = re.sub(rb"\x1b\[[0-9;?]*[A-Za-z]", b"", raw).replace(b"\r", b"").decode("utf-8", "replace")
    return text.strip()


def main():
    for tool in ("x86_64-w64-mingw32-g++", "wine", "script"):
        if not shutil.which(tool):
            sys.exit(f"needs {tool}")
    work = Path(tempfile.mkdtemp(prefix="qtrans_console_"))
    program = translate_source(DEMO, "demo", "demo.py").program
    assert "qt::py::windows_console" in program
    for h in (RUNTIME_HEADER, PRINT_HEADER, LIST_HEADER, UNICODE_HEADER):
        shutil.copy(h, work / h.name)
    (work / "with.cpp").write_text(program, encoding="utf-8")
    (work / "without.cpp").write_text(re.sub(r".*qt::py::windows_console .*\n", "", program), encoding="utf-8")
    for name in ("with", "without"):
        subprocess.run(["x86_64-w64-mingw32-g++", "-std=c++17", "-O2", "-static", "-I.", f"{name}.cpp", "-o",
                        f"{name}.exe"], cwd=work, check=True)
    (work / "demo.py").write_text(DEMO, encoding="utf-8")
    python = subprocess.run([sys.executable, "demo.py"], capture_output=True, text=True, encoding="utf-8",
                            cwd=work).stdout.strip()
    with_, without = shown("with.exe", work), shown("without.exe", work)
    print("Python:                     ", python)
    print("C++ with windows_console:   ", with_)
    print("C++ writing bytes (before): ", without)
    print("same as Python:", with_ == python)
    return 0 if with_ == python else 1


if __name__ == "__main__":
    sys.exit(main())
