"""The fuzzer: random typed programs, checked against Python, failures shrunk to minimal ones.

    python -m tools.fuzz [--seed S] [--minutes M] [--batch N] [--cases C] [--out DIR] [--no-shrink]

Each batch is --batch generated functions in one module (one C++ compile for all of them),
verified on --cases random inputs each (the verifier's own: non-square, empty and length-1
arrays; every other batch with the stress profile's inf, NaN and extreme values). A failure is
isolated to its function, shrunk, and written to DIR as a module that reproduces it (its
failing input frozen in QTRANS_CASES), plus a summary in DIR/report.md. Exit code 1 if any bug
was found. The same seed gives the same programs.
"""
from __future__ import annotations

import argparse
import random
import sys
import tempfile
import time
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from tools.fuzz.check import check, module_source  # noqa: E402
from tools.fuzz.gen import Gen  # noqa: E402
from tools.fuzz.shrink import shrink  # noqa: E402


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="python -m tools.fuzz")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--minutes", type=float, default=5.0)
    ap.add_argument("--batch", type=int, default=20)
    ap.add_argument("--cases", type=int, default=40)
    ap.add_argument("--out", default="fuzz_out")
    ap.add_argument("--no-shrink", action="store_true")
    ap.add_argument("--shrink-minutes", type=float, default=10.0)
    args = ap.parse_args(argv)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    work = Path(tempfile.mkdtemp(prefix="qtrans_fuzz_"))
    rng = random.Random(args.seed)
    gen = Gen(rng)
    deadline = time.time() + 60 * args.minutes
    stats, bugs, seen = Counter(), [], set()
    refusals = Counter()
    batch = 0
    while time.time() < deadline:
        names = [f"f{batch}_{k}" for k in range(args.batch)]
        gen.new_module()  # later functions of a batch may call earlier ones
        funcs = {n: gen.function(n) for n in names}
        profile = "stress" if batch % 2 else "strict"
        res = check(funcs, work, args.cases, args.seed * 1000 + batch, profile)
        stats["generated"] += len(funcs)
        stats["translated"] += len(res.translated)
        stats["refused"] += len(res.refused)
        stats["cases"] += res.cases
        for reason in res.refused.values():
            refusals[reason.split("(")[0][:70]] += 1
        for b in res.bugs:
            stats[b.kind] += 1
            print(f"[batch {batch}] {b.kind} in {b.name}: {b.detail.strip().splitlines()[-1][:160]}", flush=True)
            if b.signature() in seen and b.kind != "mismatch":
                continue
            seen.add(b.signature())
            if not args.no_shrink:
                shrink(b, work, budget_seconds=60 * args.shrink_minutes)
                if b.shrunk:
                    print("    minimal:\n" + "\n".join("      " + x for x in b.shrunk.splitlines()), flush=True)
            bugs.append(b)
        batch += 1
        print(f"[batch {batch}] {dict(stats)}", flush=True)

    _write(out, bugs, stats, refusals, args)
    return 1 if bugs else 0


def _write(out: Path, bugs, stats, refusals, args):
    lines = ["# Fuzzer report", "",
             f"Seed {args.seed}, {args.minutes:g} minutes, batches of {args.batch}, {args.cases} random cases per function.", "",
             f"Generated {stats['generated']} functions: {stats['translated']} translated and checked "
             f"({stats['cases']:,} cases), {stats['refused']} refused with a reason. "
             f"Bugs: {len(bugs)}.", ""]
    for k, b in enumerate(bugs):
        src = b.shrunk or b.source
        cases = {b.name: b.args} if b.args is not None else None
        path = out / f"bug_{k}_{b.kind}.py"
        path.write_text(f'"""fuzzer: {b.kind} ({b.profile} profile)\n\n{b.detail.strip()[-1500:]}\n"""\n'
                        + module_source({**b.deps, b.name: src}, cases), encoding="utf-8")
        shown = "\n\n".join([d.strip() for d in b.deps.values()] + [src.strip()])
        lines += [f"## {k}: {b.kind} in `{b.name}`" + ("" if b.shrunk else " (not shrunk)"), "",
                  "```python", shown, "```", "", "```", b.detail.strip()[-800:], "```", ""]
    lines += ["## Most common refusals", ""] + [f"- {n} x {r}" for r, n in refusals.most_common(15)]
    (out / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote {out / 'report.md'} ({len(bugs)} bug(s))")


if __name__ == "__main__":
    sys.exit(main())
