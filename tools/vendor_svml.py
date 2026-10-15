"""Writes qtrans/runtime/qtrans_svml_ha.hpp and qtrans_svml_la.hpp, NumPy's own SVML routines (the
float64 maths NumPy runs on Linux x86-64 with AVX-512, carried verbatim as assembly), and
qtrans_npy_own.hpp, NumPy's own C kernels (exp and log by Tang's method, the tanh table).

    python tools/vendor_svml.py [--numpy 2.5.4] [--sdist numpy-2.5.4.tar.gz]

NumPy ships Intel's SVML as assembly (numpy/_core/src/umath/svml/linux/avx512/*.s, under NumPy's
BSD-3 licence, numpy/_core/src/umath/svml/LICENSE). qtrans does not rewrite it: it puts the same
assembly in a C++ header, so a translation calls the very machine code NumPy calls, and gets
NumPy's results to the bit. The only changes are mechanical, so several copies can be linked:

  - every symbol a file defines gets a per-function name (two files may both define a table
    called `_imldExpHATab`); the entry point becomes `__qt_svml_<name>`, weak and hidden;
  - each function's code and data go in their own COMDAT group, so the linker keeps one copy
    however many C++ files include the header;
  - the CET property notes and the stack note are left to the compiler, which writes its own.

The code is the same in NumPy 1.22 to 2.5 (checked by --check-versions); what NumPy changes
between versions is which routine it calls for which function (qtrans_npmath.hpp has that table).
"""
import argparse
import hashlib
import io
import json
import re
import tarfile
import urllib.request
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "qtrans" / "runtime"
SVML = "numpy/_core/src/umath/svml"

# the float64 functions NumPy's loops call SVML for (high accuracy: NumPy >= 2.0; low accuracy:
# NumPy 1.22 to 1.26, where sin, cos and tanh also went through it in some versions)
HA = ["exp", "log", "exp2", "log2", "log10", "expm1", "log1p", "cbrt", "tan", "asin", "acos", "atan",
      "sinh", "cosh", "asinh", "acosh", "atanh", "pow", "atan2"]
LA = HA + ["sin", "cos", "tanh"]
BINARY = {"pow", "atan2"}


def sdist(version: str, path: str | None) -> bytes:
    """NumPy's source distribution, checked against the SHA-256 PyPI publishes for it."""
    meta = json.load(urllib.request.urlopen(f"https://pypi.org/pypi/numpy/{version}/json"))
    entry = next(u for u in meta["urls"] if u["packagetype"] == "sdist")
    data = Path(path).read_bytes() if path else urllib.request.urlopen(entry["url"]).read()
    digest = hashlib.sha256(data).hexdigest()
    if digest != entry["digests"]["sha256"]:
        raise SystemExit(f"numpy {version} sdist: sha256 {digest} is not PyPI's {entry['digests']['sha256']}")
    return data


def members(data: bytes) -> dict:
    """{path inside numpy/: text} for the SVML files and their licence."""
    out = {}
    if data[:2] == b"PK":
        z = zipfile.ZipFile(io.BytesIO(data))
        items = ((n, lambda n=n: z.read(n)) for n in z.namelist())
    else:
        t = tarfile.open(fileobj=io.BytesIO(data))
        items = ((m.name, lambda m=m: t.extractfile(m).read()) for m in t.getmembers() if m.isfile())
    for name, read in items:
        i = name.find("numpy/")
        rel = name[i:] if i >= 0 else name
        rel = rel.replace("numpy/core/", "numpy/_core/")  # NumPy 1.x layout
        if rel.startswith(SVML + "/") or rel in OWN_FILES:
            out[rel] = read().decode("utf-8")
    return out


def wrap(src: str, fn: str, variant: str) -> tuple[str, str]:
    """One .s file as a C++ file-scope asm block, and the name of its entry point."""
    lines, skip = [], False
    for line in src.splitlines():
        s = line.strip()
        if s.startswith(".file") or s == "endbr64" or ".note.GNU-stack" in s:
            continue
        if s.startswith('.section ".note.gnu.property"'):
            skip = True
            continue
        if skip and not (s.startswith(".section") or s.startswith(".text")):
            continue
        skip = False
        lines.append(line.rstrip())
    body = re.sub(r"/\*.*?\*/", "", "\n".join(lines), flags=re.S)  # Intel's banner and comments
    body = re.sub(r"\n\s*\n+", "\n", body)
    entries = re.findall(r"\.globl\s+(\S+)", body)
    if len(entries) != 1:
        raise SystemExit(f"{fn}_{variant}: expected one entry point, found {entries}")
    entry = entries[0]
    defined = set(re.findall(r"^\s*([A-Za-z_.][\w.$]*):", body, flags=re.M))
    ren = {n: f"{n}__qt_{fn}_{variant}" for n in defined if n != entry}
    ren[entry] = "__qt" + entry[1:]
    pat = re.compile(r"(?<![\w.$])(" + "|".join(re.escape(n) for n in sorted(ren, key=len, reverse=True)) + r")(?![\w.$])")
    body = pat.sub(lambda m: ren[m.group(1)], body)
    group = f"__qt_svml_{fn}_{variant}"
    body = re.sub(r"^\s*\.text\s*$", f'\t.section .text.{group},"axG",@progbits,{group},comdat', body, flags=re.M)
    body = re.sub(r'^\s*\.section\s+\.rodata,\s*"a"\s*$', f'\t.section .rodata.{group},"aG",@progbits,{group},comdat',
                  body, flags=re.M)
    if re.search(r"^\s*\.(text|data|bss)\b|^\s*\.section\s+(?!\.(text|rodata)\.__qt_svml_)", body, flags=re.M):
        raise SystemExit(f"{fn}_{variant}: a section qtrans does not know how to group")
    body = re.sub(r"\.globl\s+(\S+)", r".weak \1\n\t.hidden \1", body)
    return f'__asm__(R"QTSVML(\n\t.pushsection .text\n{body}\n\t.popsection\n)QTSVML");', ren[entry]


def header(files: dict, variant: str, version: str, sdist_sha: str) -> str:
    fns = HA if variant == "ha" else LA
    lic = files[f"{SVML}/LICENSE"].strip()
    blocks, decls, wraps, prov = [], [], [], []
    for fn in fns:
        rel = f"{SVML}/linux/avx512/svml_z0_{fn}_d_{variant}.s"
        text = files[rel]
        prov.append(f"//   {rel.split('/')[-1]:28s} sha256 {hashlib.sha256(text.encode()).hexdigest()}")
        asm, entry = wrap(text, fn, variant)
        blocks += [f"// {fn}: {rel}", asm, ""]
        if fn in BINARY:
            decls.append(f'extern "C" QT_SVML_TARGET __m512d {entry}(__m512d, __m512d);')
            wraps.append(f"QT_SVML_TARGET inline __m512d {fn}(__m512d x, __m512d y) {{ return {entry}(x, y); }}")
        else:
            decls.append(f'extern "C" QT_SVML_TARGET __m512d {entry}(__m512d);')
            wraps.append(f"QT_SVML_TARGET inline __m512d {fn}(__m512d x) {{ return {entry}(x); }}")
    acc = "high accuracy (the _ha routines): what NumPy 2.0 and later call" if variant == "ha" else \
        "low accuracy (the _la routines): what NumPy 1.22 to 1.26 call"
    out = [f"// qtrans_svml_{variant}.hpp: NumPy's SVML float64 routines, {acc}.",
           f"// Written by tools/vendor_svml.py from numpy {version} (sdist sha256 {sdist_sha}); do not edit.",
           "//",
           "// The assembly is NumPy's, unchanged but for symbol names, COMDAT grouping and the notes the",
           "// compiler writes itself, so these compute exactly what NumPy computes. Source files:",
           *prov,
           "//",
           "// NumPy distributes these files under the following licence (numpy/_core/src/umath/svml/LICENSE):",
           "//",
           *[("// " + l).rstrip() for l in lic.splitlines()],
           "#pragma once",
           "",
           "#if defined(__x86_64__) && defined(__linux__) && (defined(__GNUC__) || defined(__clang__))",
           "#include <immintrin.h>",
           "",
           '#define QT_SVML_TARGET __attribute__((target("avx512f,avx512cd,avx512bw,avx512dq,avx512vl")))',
           "",
           *blocks,
           *decls,
           "",
           "namespace qt {",
           f"namespace svml_{variant} {{",
           *wraps,
           f"}}  // namespace svml_{variant}",
           "}  // namespace qt",
           "",
           "#endif",
           ""]
    return "\n".join(out)


UMATH = "numpy/_core/src/umath"
OWN_FILES = (f"{UMATH}/loops_exponent_log.dispatch.c.src", f"{UMATH}/npy_simd_data.h",
             f"{UMATH}/loops_hyperbolic.dispatch.cpp.src")


def _between(text: str, start: str, end: str) -> str:
    i = text.index(start)
    return text[i:text.index(end, i) + len(end)]


def own_header(files: dict, version: str, sdist_sha: str) -> str:
    """NumPy's AVX512F exp / log (Tang's method) and its tanh coefficients, taken from its sources."""
    exl = files[f"{UMATH}/loops_exponent_log.dispatch.c.src"]
    data = files[f"{UMATH}/npy_simd_data.h"]
    hyp = files[f"{UMATH}/loops_hyperbolic.dispatch.cpp.src"]
    # the helpers, then the two kernels, as NumPy writes them
    helpers = _between(exl, "#ifdef SIMD_AVX512F\n\nNPY_FINLINE __mmask16", "#endif // SIMD_AVX512F")
    helpers = helpers.split("\n", 1)[1].rsplit("#endif", 1)[0]
    exp_k = _between(exl, "static void\nAVX512F_exp_DOUBLE(", "\n}\n")
    log_k = _between(exl, "static void\nAVX512F_log_DOUBLE(", "\n}\n")
    rep = _between(log_k, "    /**begin repeat", "/**end repeat**/")
    line = [l for l in rep.splitlines() if "@i@" in l]
    log_k = log_k.replace(rep, "\n".join(l.replace("@i@", str(i)) for i in range(8) for l in line))
    exp_k = exp_k.replace("static void\nAVX512F_", "static QT_NPY_TANG_TARGET void\nAVX512F_")
    log_k = log_k.replace("static void\nAVX512F_", "static QT_NPY_TANG_TARGET void\nAVX512F_")
    workaround = _between(exl, "#if defined(__clang__)\n    #if defined(__apple_build_version__)", "#endif\n#endif")
    tables = data.split("#define __NPY_SIMD_DATA_H_", 1)[1].rsplit("#endif", 1)[0]
    tables = tables.replace("static npy_uint64", "alignas(64) static const npy_uint64").replace(
        "static npy_uint32", "alignas(64) static const npy_uint32")
    # NumPy's tanh: 18 coefficient rows (b, c0 ... c16) of 16 intervals each
    lut = _between(hyp, "lut16x18[] = {", "};")
    vals = re.findall(r"0x[0-9a-fA-F]+ull", lut)
    if len(vals) != 18 * 16:
        raise SystemExit(f"tanh table: {len(vals)} values, expected 288")
    lut_t = re.findall(r"0x[0-9a-fA-F]+ull", _between(hyp, "lut18x16[] = {", "};"))
    if [lut_t[i * 18 + r] for r in range(18) for i in range(16)] != vals:
        raise SystemExit("tanh tables: the two layouts disagree")
    rows = ["    " + ", ".join(vals[r * 16:(r + 1) * 16]) + "," for r in range(18)]
    prov = [f"//   {rel.split('/')[-1]:36s} sha256 {hashlib.sha256(files[rel].encode()).hexdigest()}" for rel in OWN_FILES]
    lic = files[f"{SVML}/LICENSE"].strip()
    return "\n".join([
        "// qtrans_npy_own.hpp: NumPy's own float64 kernels: exp and log by Tang's method (AVX512F), as",
        "// NumPy computes them where it doesn't call SVML, and the coefficients of its tanh.",
        f"// Written by tools/vendor_svml.py from numpy {version} (sdist sha256 {sdist_sha}); do not edit.",
        "// The kernels are NumPy's code as written, with NumPy's types and macros defined for C++. Sources:",
        *prov, "//", "// NumPy's licence (BSD-3), as for qtrans_svml_ha.hpp:", "//",
        *[("// " + l).rstrip() for l in lic.splitlines()],
        "#pragma once", "",
        "#include <cstddef>", "#include <cstdint>", "#include <cmath>", "",
        "namespace qt {", "namespace npy_own {", "",
        "using npy_double = double;", "using npy_float = float;", "using npy_int = int;", "using npy_int32 = std::int32_t;",
        "using npy_intp = std::ptrdiff_t;", "using npy_uint64 = std::uint64_t;", "using npy_uint32 = std::uint32_t;", "",
        "// NumPy's tanh (numpy/_core/src/umath/loops_hyperbolic.dispatch.cpp.src): row k holds coefficient k",
        "// (b, c0, ..., c16) of each of the 16 intervals",
        "alignas(64) static const std::uint64_t tanh_lut16x18[18 * 16] = {", *rows, "};", "",
        "}  // namespace npy_own", "}  // namespace qt", "",
        "#if defined(__x86_64__) && (defined(__GNUC__) || defined(__clang__)) && !defined(_WIN32)",
        "#include <immintrin.h>",
        "#define QT_NPY_OWN_TANG 1",
        "#if defined(__GNUC__)",
        "#pragma GCC diagnostic push",
        '#pragma GCC diagnostic ignored "-Wconversion"',
        '#pragma GCC diagnostic ignored "-Wsign-conversion"',
        '#pragma GCC diagnostic ignored "-Wunused-parameter"',
        '#pragma GCC diagnostic ignored "-Wunused-function"',
        '#pragma GCC diagnostic ignored "-Wold-style-cast"',
        '#pragma GCC diagnostic ignored "-Wshadow"',
        '#pragma GCC diagnostic ignored "-Wcast-align"',
        '#pragma GCC diagnostic ignored "-Wdouble-promotion"',
        "#endif",
        "namespace qt {", "namespace npy_own {", "",
        '#define QT_NPY_TANG_TARGET __attribute__((target("avx512f")))',
        "#define NPY_FINLINE static inline QT_NPY_TANG_TARGET",
        "#define NPY_INFINITY __builtin_inf()",
        "#define NPY_NAN (static_cast<double>(__builtin_nanf(\"\")))",
        "#define NPY_DECL_ALIGNED(x) __attribute__((aligned(x)))",
        "#define npyv_tobits_b64(m) (static_cast<unsigned>(m))",
        "#define npy_log std::log",
        "inline void npy_set_floatstatus_overflow() {}", "inline void npy_set_floatstatus_underflow() {}",
        "inline void npy_set_floatstatus_invalid() {}", "inline void npy_set_floatstatus_divbyzero() {}", "",
        "// numpy/_core/src/umath/npy_simd_data.h (its tables are for builds with AVX512F and without SVML)",
        "#define NPY_HAVE_AVX512F 1", tables, "#undef NPY_HAVE_AVX512F", "",
        "// numpy/_core/src/umath/loops_exponent_log.dispatch.c.src", helpers, exp_k, "", workaround, "", log_k,
        "#undef WORKAROUND_LLVM__mm512_mask_mul_pd",
        "#undef NPY_FINLINE", "#undef NPY_INFINITY", "#undef NPY_NAN", "#undef NPY_DECL_ALIGNED", "#undef npyv_tobits_b64",
        "#undef npy_log", "",
        "}  // namespace npy_own", "}  // namespace qt",
        "#if defined(__GNUC__)", "#pragma GCC diagnostic pop", "#endif",
        "#endif", ""])


def normalised(text: str) -> str:
    return "\n".join(l for l in text.splitlines() if "endbr64" not in l and not l.strip().startswith(".file")
                     and "note.GNU-stack" not in l).split('.section ".note.gnu.property"')[0]


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--numpy", default="2.5.4", help="the NumPy release to take SVML from")
    ap.add_argument("--sdist", help="its source distribution, if already downloaded (checked against PyPI)")
    ap.add_argument("--check-versions", nargs="*", default=[], metavar="VERSION",
                    help="also confirm these NumPy releases ship the same code")
    a = ap.parse_args(argv)
    data = sdist(a.numpy, a.sdist)
    files = members(data)
    sha = hashlib.sha256(data).hexdigest()
    for v in a.check_versions:
        other = members(sdist(v, None))
        for rel, text in files.items():
            if rel.endswith("_d_ha.s") or rel.endswith("_d_la.s"):
                if rel in other and normalised(other[rel]) != normalised(text):
                    raise SystemExit(f"numpy {v}: {rel} differs from numpy {a.numpy}")
        print(f"numpy {v}: the same SVML code as {a.numpy}")
    for variant in ("ha", "la"):
        path = OUT / f"qtrans_svml_{variant}.hpp"
        path.write_text(header(files, variant, a.numpy, sha), encoding="utf-8")
        print(f"wrote {path} ({path.stat().st_size:,} bytes)")
    path = OUT / "qtrans_npy_own.hpp"
    path.write_text(own_header(files, a.numpy, sha), encoding="utf-8")
    print(f"wrote {path} ({path.stat().st_size:,} bytes)")


if __name__ == "__main__":
    main()
