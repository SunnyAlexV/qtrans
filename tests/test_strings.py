"""Python's str methods in C++: indexes count characters, and case, character classes, digits
and repr() follow the Unicode tables of the very Python running (qtrans_unicode.hpp is written
from it). Checked for every code point, against that Python, and by qtrans verify on text made
to break things (spaces, case, Unicode digits, sigma at the end of a word)."""
import shutil
import subprocess
import sys
import textwrap
import unicodedata

import pytest

from qtrans import unicode_tables
from qtrans.translator import LIST_HEADER, PRINT_HEADER, RUNTIME_HEADER, UNICODE_HEADER, translate_source
from qtrans.verify import verify_file
from tests.test_semantics import HAVE_CXX

FUNCTIONS = '''def clean(s: str) -> str:
    return s.strip().lower()


def words(s: str) -> list[str]:
    return s.split()


def fields(s: str) -> list[str]:
    return s.split(",", 2)


def first_word(s: str) -> str:
    parts = s.split()
    if not parts:
        return ""
    return parts[0]


def where(s: str, sub: str) -> int:
    return s.find(sub) + 10 * s.count(sub)


def tag(s: str) -> str:
    if s.startswith(("A", "a")):
        return s.upper().center(9, "*")
    if s.endswith("x"):
        return s.replace("x", "y", 1).rjust(6, ".")
    return s.zfill(5) + s.ljust(4) + "|"


def kind(s: str) -> int:
    k = 0
    if s.isdigit():
        k += 1
    if s.isalpha():
        k += 2
    if s.isspace():
        k += 4
    if s.isupper():
        k += 8
    if s.islower():
        k += 16
    if "a" in s:
        k += 32
    return k


def chars(s: str) -> str:
    out = ""
    for ch in s:
        if ch != " ":
            out += ch * 2
    return out + s[::-1] + s[1:3]


def at(s: str, i: int) -> str:
    return s[i]


def vowels(s: str) -> int:
    return len([c for c in s if c in "aeiou"])


def number(s: str) -> float:
    return float(s) * 2.0


def whole(s: str) -> int:
    return int(s) + 1


def total(s: str) -> float:
    t = 0.0
    for p in s.split(","):
        t += float(p)
    return t

'''

PROGRAM = """
line = "  AAPL, 190.5 ,MSFT,410.25  "
parts = [p.strip() for p in line.split(",")]
print(parts, len(parts))
names = parts[0::2]
prices = [float(p) for p in parts[1::2]]
for n, p in zip(names, prices):
    print(n.lower().capitalize().ljust(6, "."), f"{p:8.2f}", n[0], n[-1], n[1:3], "A" in n)
word = "ΟΔΟΣ"
print(word.lower(), "hello wORLD ǆ".title(), word.upper(), word.swapcase(), "Straße".casefold(), "Ab Cd".istitle(), "straße".upper(), repr("tab\\there\\u00a0"))
print("x" * 3, "-".join(["a", "b"]), "a-b-c".replace("-", "+", 1), "abc".find("c"), "  ".isspace())
print(int(" ١٢ ") + 1, float("1_000.5"), "007".zfill(5), "ab".center(7, "*"))
for ch in "héllo":
    print(ch, end="|")
print()
"""


@pytest.mark.skipif(not HAVE_CXX, reason="no C++ compiler")
def test_every_code_point_agrees_with_python(tmp_path):
    for h in (RUNTIME_HEADER, PRINT_HEADER, LIST_HEADER, UNICODE_HEADER):
        shutil.copy(h, tmp_path / h.name)
    (tmp_path / "all.cpp").write_text(encoding="utf-8", data=textwrap.dedent("""
        #include "qtrans_list.hpp"
        #include <cstdio>
        #ifdef _WIN32
        #include <fcntl.h>
        #include <io.h>
        #endif
        static std::string F(const std::string& s) {
            try { return qt::py::repr(qt::py::parse_float(s)); } catch (const std::invalid_argument& e) { return e.what(); }
        }
        static std::string I(const std::string& s) {
            try { return std::to_string(qt::py::parse_int(s)); } catch (const std::invalid_argument& e) { return e.what(); }
        }
        int main() {
        #ifdef _WIN32
            _setmode(_fileno(stdout), _O_BINARY);
        #endif
            const std::string sigma = "\\316\\243";  // U+03A3, in UTF-8
            std::string out;
            for (char32_t c = 0; c < 0x110000; ++c) {
                if (c >= 0xD800 && c <= 0xDFFF) continue;
                std::string s;
                qt::uni::encode(s, c);
                int f = qt::py::isalpha(s) | qt::py::isdigit(s) << 1 | qt::py::isdecimal(s) << 2 |
                        qt::py::isnumeric(s) << 3 | qt::py::isalnum(s) << 4 | qt::py::isspace(s) << 5 |
                        qt::py::isupper(s) << 6 | qt::py::islower(s) << 7 | qt::py::isprintable(s) << 8 |
                        qt::py::istitle(s) << 9 | qt::py::istitle(s + "a") << 10 | qt::py::istitle("A" + s) << 11;
                out += qt::py::upper(s) + "\\t" + qt::py::lower(s) + "\\t" + qt::py::capitalize(s + "A" + sigma) + "\\t" +
                       std::to_string(f) + "\\t" + qt::py::lower("A" + s + sigma) + qt::py::lower(s + sigma) +
                       qt::py::lower("A" + sigma + s) + "\\t" + qt::py::title("a" + s + "b" + sigma + " " + s) + "\\t" +
                       qt::py::swapcase(s + sigma) + "\\t" + qt::py::casefold(s) + "\\t" + qt::py::repr(s) + "\\t" + I(s) + "\\t" +
                       F("1" + s + "5") + "\\t" + I(s + "_0" + s) + "\\n";
            }
            std::fwrite(out.data(), 1, out.size(), stdout);
        }
    """))
    from qtrans.verify import find_compiler
    exe = tmp_path / ("all.exe" if sys.platform == "win32" else "all")
    cmd = find_compiler().command(tmp_path / "all.cpp", exe, tmp_path)
    subprocess.run(cmd, check=True, capture_output=True, cwd=tmp_path)
    cpp = subprocess.run([str(exe)], capture_output=True, check=True).stdout

    def F(s):
        try:
            return repr(float(s))
        except ValueError as e:
            return str(e)

    def I(s):
        try:
            return str(int(s))
        except ValueError as e:
            return str(e)
    rows = []
    for c in range(0x110000):
        if 0xD800 <= c <= 0xDFFF:
            continue
        s = chr(c)
        f = (s.isalpha() | s.isdigit() << 1 | s.isdecimal() << 2 | s.isnumeric() << 3 | s.isalnum() << 4
             | s.isspace() << 5 | s.isupper() << 6 | s.islower() << 7 | s.isprintable() << 8
             | s.istitle() << 9 | (s + "a").istitle() << 10 | ("A" + s).istitle() << 11)
        sig = ("A" + s + "\u03a3").lower() + (s + "\u03a3").lower() + ("A\u03a3" + s).lower()
        titled, swapped = ("a" + s + "b\u03a3 " + s).title(), (s + "\u03a3").swapcase()
        rows.append(f"{s.upper()}\t{s.lower()}\t{(s + 'A' + chr(0x3a3)).capitalize()}\t{f}\t{sig}\t{titled}\t{swapped}\t{s.casefold()}\t{s!r}\t{I(s)}\t"
                    f"{F('1' + s + '5')}\t{I(s + '_0' + s)}\n")
    py = "".join(rows).encode("utf-8")
    if py != cpp:
        pl, cl = py.split(b"\n"), cpp.split(b"\n")
        bad = [i for i in range(min(len(pl), len(cl))) if pl[i] != cl[i]]
        pytest.fail(f"{len(bad)} code points differ, first: {pl[bad[0]]!r} vs {cl[bad[0]]!r}")


def test_the_tables_are_this_pythons():
    """The header the C++ includes holds this Python's Unicode: the shipped one when the versions
    agree, else the one written for this Python (3.10 has Unicode 13, 3.14 has 16)."""
    text = UNICODE_HEADER.read_text(encoding="utf-8")
    assert f'#define QTRANS_UNICODE_VERSION "{unicodedata.unidata_version}"' in text
    assert (UNICODE_HEADER == unicode_tables.SHIPPED) == (unicode_tables.shipped_version() == unicodedata.unidata_version)
    assert text == unicode_tables.generate()


@pytest.mark.skipif(not HAVE_CXX, reason="no C++ compiler")
def test_string_functions_agree_with_python(tmp_path):
    f = tmp_path / "strs.py"
    f.write_text(FUNCTIONS, encoding="utf-8")
    rep = verify_file(f, n_random=150, build_dir=tmp_path / "build")
    assert rep.ok and rep.warnings == [], rep.to_text()
    assert len(rep.functions) == 13 and all(fc.count("mismatch") == 0 for fc in rep.functions)


@pytest.mark.skipif(not HAVE_CXX, reason="no C++ compiler")
def test_string_program_prints_what_python_prints(tmp_path):
    f = tmp_path / "text.py"
    f.write_text(textwrap.dedent(PROGRAM).lstrip(), encoding="utf-8")
    rep = verify_file(f, n_random=0, build_dir=tmp_path / "build")
    assert rep.program is not None, rep.translation.report.to_text()
    assert rep.program.identical and rep.warnings == [], rep.to_text() + rep.program.py_out + rep.program.cpp_out


def test_strings_read_like_cpp():
    h = translate_source(FUNCTIONS, "m").header
    assert "return qt::py::lower(qt::py::strip(s));" in h
    assert 'return qt::py::split(s, ",", 2);' in h
    assert 'if (qt::py::startswith(s, {"A", "a"})) {' in h
    assert "for (const std::string& ch : qt::py::chars(s)) {" in h
    assert "return qt::py::char_at(s, i);" in h
    assert "if (parts.empty()) {" in h
    h = translate_source('def f() -> str:\n    return "abc".upper() + "x".center(5, "*")\n', "m").header
    assert 'return "ABC**x**";' in h  # all literals: Python's own answer, worked out when translating


@pytest.mark.parametrize("src, words", [
    ("def f(s: str) -> str:\n    return s.expandtabs()\n", "str.expandtabs() is not supported"),
    ("def f(s: str) -> bool:\n    return s.startswith('a', 1)\n", "takes 1 argument(s), got 2"),
    ("def f(s: str) -> list[str]:\n    return s.split(sep=',', limit=2)\n", "no keyword 'limit'"),
    ("def f(s: str) -> str:\n    return s.replace('a', 1)\n", "needs a str here, got a int"),
    ("def f(s: str) -> bool:\n    return 1 in s\n", "needs a str on the left"),
    ("def f(s: str) -> str:\n    return s[1.5]\n", "string indices must be ints"),
    ("def f() -> float:\n    return float('abc')\n", "could not convert string to float: 'abc'"),
    ("def f(s: str) -> list[str]:\n    return s.split('')\n", None),
])
def test_string_refusals(src, words):
    r = translate_source(src, "m", "m.py")
    bad = [f for f in r.report.functions if not f.translated]
    if words is None:  # translated; the C++ raises ValueError("empty separator") as Python does
        assert not bad and 'qt::py::split(s, "")' in r.header
        return
    assert bad and words in bad[0].reason, r.report.to_text()


@pytest.mark.skipif(not HAVE_CXX, reason="no C++ compiler")
def test_utf8_to_utf16_for_every_code_point(tmp_path):
    """What qt::py::windows_console hands the Windows console (UTF-16), for every character,
    against Python's own encoder; a character cut off at the end of the bytes waits."""
    for h in (RUNTIME_HEADER, PRINT_HEADER, UNICODE_HEADER):
        shutil.copy(h, tmp_path / h.name)
    (tmp_path / "u16.cpp").write_text(encoding="utf-8", data=textwrap.dedent("""
        #include "qtrans_print.hpp"
        #include <cstdio>
        #ifdef _WIN32
        #include <fcntl.h>
        #include <io.h>
        #endif
        int main() {
        #ifdef _WIN32
            _setmode(_fileno(stdout), _O_BINARY);
        #endif
            std::string all;
            for (char32_t c = 0; c < 0x110000; ++c)
                if (c < 0xD800 || c > 0xDFFF) qt::uni::encode(all, c);
            std::u16string out;
            const std::size_t used = qt::py::utf8_to_utf16(all.data(), all.size() - 2, out);  // the last one cut
            std::printf("%zu %zu\\n", used, all.size());
            for (char16_t u : out) std::printf("%04x", static_cast<unsigned>(u));
        }
    """))
    from qtrans.verify import find_compiler
    exe = tmp_path / ("u16.exe" if sys.platform == "win32" else "u16")
    subprocess.run(find_compiler().command(tmp_path / "u16.cpp", exe, tmp_path), check=True, capture_output=True,
                   cwd=tmp_path)
    head, units = subprocess.run([str(exe)], capture_output=True, check=True, text=True).stdout.split("\n", 1)
    text = "".join(chr(c) for c in range(0x110000) if not 0xD800 <= c <= 0xDFFF)
    data = text.encode("utf-8")
    used, size = map(int, head.split())
    assert size == len(data) and used == len(data) - 4  # U+10FFFF (4 bytes) waits for its last bytes
    assert units == text[:-1].encode("utf-16-be").hex()


def test_programs_with_text_beyond_ascii_write_to_a_windows_console_as_python_does():
    r = translate_source('print("straße", "ΟΔΟΣ".lower())\n', "m", "m.py")
    assert "    qt::py::windows_console console;  // print() to a Windows console as Python does" in r.program
    assert "windows_console" not in translate_source('print("plain", 1)\n', "m", "m.py").program
    clash = translate_source('console = "é"\nprint(console)\n', "m", "m.py").program
    assert "qt::py::windows_console console_;" in clash
