"""Screenshots of what a qtrans user sees, for each version: the `qtrans live` editor (the C++
next to the Python, the reasons for a line, a verification), the `qtrans explain` page, and the
terminal output of translate / verify / export.

    python tools/screenshots.py [out_folder]      (needs Playwright's Chromium)

It runs the real commands on a small demo file; nothing in the pictures is mocked up.
"""
import html
import os
import subprocess
import sys
import tempfile
import threading
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

DEMO = '''"""A small pricing script: what qtrans is for."""
import math
import random

import numpy as np
from qtrans.typing import Vec

RATE = 0.05


def discount(t: float) -> float:
    return math.exp(-RATE * t)


def annuity(n: int) -> float:
    total = 0.0
    for t in range(1, n + 1):  # qtrans: unroll 4
        total += discount(t)
    return total


class Payoff:
    def __init__(self, strike: float):
        self.strike = strike

    def value(self, spot: float) -> float:
        return 0.0


class Call(Payoff):
    def value(self, spot: float) -> float:
        return max(spot - self.strike, 0.0)


class Put(Payoff):
    def value(self, spot: float) -> float:
        return max(self.strike - spot, 0.0)


def mc_value(p: Payoff, s0: float, vol: float, n: int) -> float:
    total = 0.0
    for _ in range(n):
        total += p.value(s0 * math.exp(vol * random.gauss(0.0, 1.0)))
    return total / n


def bootstrap_mean(x: Vec, seed: int) -> float:
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(x), size=len(x))  # int64 indexes, as NumPy draws them
    return float(x[idx].mean())


def safe_ratio(a: float, b: float) -> float:
    try:
        return a / b
    except ZeroDivisionError as e:
        print("no ratio:", e)
        return 0.0


random.seed(7)
book: list[Payoff] = [Call(100.0), Put(95.0)]
for p in book:
    print(type(p).__name__, round(mc_value(p, 100.0, 0.2, 1000), 4))
print("annuity(10) =", round(annuity(10), 6))
print("bootstrap:", bootstrap_mean(np.linspace(90.0, 110.0, 50), 3))
print(safe_ratio(1.0, 4.0), safe_ratio(1.0, 0.0))
'''

def _line(text):
    """The demo's line number holding `text`."""
    return next(i for i, line in enumerate(DEMO.splitlines(), 1) if text in line)


TERMINAL_CSS = """
body { margin: 0; background: #0d1117; }
.win { margin: 24px; border-radius: 10px; overflow: hidden; box-shadow: 0 8px 30px #0008; width: 1180px; }
.bar { background: #2b3036; padding: 9px 14px; font: 13px system-ui, sans-serif; color: #c9d1d9; }
.dot { display: inline-block; width: 11px; height: 11px; border-radius: 50%; margin-right: 6px; }
pre { margin: 0; padding: 16px 18px; background: #161b22; color: #e6edf3;
      font: 13.5px/1.45 "DejaVu Sans Mono", Consolas, monospace; white-space: pre-wrap; }
.cmd { color: #7ee787; } .ok { color: #7ee787; } .bad { color: #ff7b72; } .dim { color: #8b949e; }
"""


def _terminal_html(title, blocks):
    body = []
    for cmd, out in blocks:
        body.append(f'<span class="cmd">PS C:\\Users\\you\\models&gt; {html.escape(cmd)}</span>')
        for line in out.rstrip().splitlines():
            e = html.escape(line)
            if "PASS" in line or "[ok]" in line or "passed" in line or "[your C++]" in line:
                e = f'<span class="ok">{e}</span>'
            elif "FAIL" in line or "rejected" in line or "NOT" in line:
                e = f'<span class="bad">{e}</span>'
            body.append(e)
        body.append("")
    dots = "".join(f'<span class="dot" style="background:{c}"></span>' for c in ("#ff5f56", "#ffbd2e", "#27c93f"))
    return (f"<!doctype html><meta charset=utf-8><style>{TERMINAL_CSS}</style>"
            f'<div class="win"><div class="bar">{dots} {html.escape(title)}</div><pre>{chr(10).join(body)}</pre></div>')


def _run(args, cwd):
    p = subprocess.run([sys.executable, "-m", "qtrans", *args], cwd=cwd, capture_output=True, text=True,
                       env={**os.environ, "PYTHONPATH": str(ROOT)})
    out = p.stdout + p.stderr
    return out.replace(str(cwd) + os.sep, "").replace(str(cwd), ".")


def main(out_dir):
    from playwright.sync_api import sync_playwright
    from qtrans.live import serve
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    work = Path(tempfile.mkdtemp(prefix="qtrans_shots_"))
    demo = work / "pricing.py"
    demo.write_text(DEMO, encoding="utf-8")
    shots = []

    # the terminal: translate, verify, export
    t1 = _run(["translate", "pricing.py", "-o", "cpp_out"], work)
    t2 = _run(["verify", "pricing.py", "--cases", "100"], work)
    t3 = _run(["export", "pricing.py", "-o", "pricing_cpp", "--cases", "50"], work)
    v_lines = [l for l in t2.splitlines() if not l.startswith("  platform")]
    (work / "terminal.html").write_text(_terminal_html("Windows PowerShell", [
        ("qtrans translate pricing.py -o cpp_out", t1),
        ("qtrans verify pricing.py --cases 100", "\n".join(v_lines)),
        ("qtrans export pricing.py -o pricing_cpp", t3)]), encoding="utf-8")
    _run(["explain", "pricing.py", "--html", "explain.html"], work)

    box, started = {}, threading.Event()

    def ready(url, server):
        box["url"], box["server"] = url, server
        started.set()
    threading.Thread(target=serve, args=(demo,), kwargs={"open_browser": False, "ready": ready}, daemon=True).start()
    started.wait(60)
    with sync_playwright() as p:
        browser = p.chromium.launch()
        page = browser.new_page(viewport={"width": 1500, "height": 920}, device_scale_factor=1)
        page.goto(box["url"])
        page.wait_for_selector("#cpp .cl")
        page.wait_for_function("document.getElementById('status').textContent.includes('translated')")
        page.evaluate(f"goToLine({_line('for t in range')})")  # the loop with its unroll hint
        page.wait_for_timeout(400)
        page.screenshot(path=str(out / "1_live_editor.png"))
        shots.append("1_live_editor.png")
        page.evaluate(f"goToLine({_line('return max(spot - self.strike')})")  # an override: virtual in C++
        page.wait_for_timeout(400)
        page.screenshot(path=str(out / "2_live_program_line.png"))
        shots.append("2_live_program_line.png")
        page.evaluate(f"goToLine({_line('random.gauss')})")  # Python's random module, number for number
        page.wait_for_timeout(400)
        page.screenshot(path=str(out / "2b_live_program_main.png"))
        shots.append("2b_live_program_main.png")
        page.evaluate(f"goToLine({_line('x[idx].mean()')})")  # int arrays: the bootstrap's gather
        page.wait_for_timeout(400)
        page.screenshot(path=str(out / "2c_live_int_arrays.png"))
        shots.append("2c_live_int_arrays.png")
        page.click("#verify")
        page.wait_for_function("document.querySelector('#why #closeverify') !== null", timeout=300000)
        page.wait_for_timeout(300)
        page.screenshot(path=str(out / "3_live_verified.png"))
        shots.append("3_live_verified.png")

        page = browser.new_page(viewport={"width": 1500, "height": 1000})
        page.goto((work / "explain.html").as_uri())
        page.wait_for_timeout(300)
        page.screenshot(path=str(out / "4_explain_page.png"))
        shots.append("4_explain_page.png")

        page = browser.new_page(viewport={"width": 1230, "height": 400})
        page.goto((work / "terminal.html").as_uri())
        page.screenshot(path=str(out / "5_terminal.png"), full_page=True)
        shots.append("5_terminal.png")
        browser.close()
    box["server"].shutdown()
    return [out / s for s in shots]


if __name__ == "__main__":
    for s in main(sys.argv[1] if len(sys.argv) > 1 else "screenshots"):
        print(s)
