"""The translation report: what converted, what didn't and why, and every place
where C++ behaviour can differ from Python's."""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class FunctionReport:
    name: str
    lineno: int
    translated: bool
    reason: str | None = None
    reason_line: int | None = None
    hint: str | None = None
    notes: list = field(default_factory=list)
    override: str | None = None  # your own C++ is used: "up to date", "out of date", "not yet accepted"


@dataclass
class ModuleReport:
    module: str
    filename: str
    functions: list = field(default_factory=list)
    module_notes: list = field(default_factory=list)
    constants: list = field(default_factory=list)
    general_notes: list = field(default_factory=list)

    @property
    def translated(self):
        return [f for f in self.functions if f.translated]

    @property
    def rejected(self):
        return [f for f in self.functions if not f.translated]

    def summary(self) -> str:
        funcs = [f for f in self.functions if f.name != "(program)"]
        prog = next((f for f in self.functions if f.name == "(program)"), None)
        text = f"{sum(f.translated for f in funcs)} of {len(funcs)} functions translated" if funcs or not prog else ""
        if prog is not None:
            text += ("; " if text else "") + ("the program (main) translated" if prog.translated
                                              else "the program (main) NOT translated")
        return text

    def to_markdown(self) -> str:
        out = [f"# qtrans report: `{self.filename}`", "", f"**{self.summary()}.**", ""]
        if self.translated:
            out += ["## Translated", ""]
            for f in self.translated:
                out.append(f"- `{f.name}` (line {f.lineno})" + (f": your own C++, {f.override}" if f.override else ""))
                for n in f.notes:
                    out.append(f"  - {n}")
            out.append("")
        if self.general_notes:
            out += ["## Differences on overflow and error paths", "",
                    "These only matter on inputs where Python would raise an exception or overflow.", ""]
            out += [f"- {n}" for n in self.general_notes]
            out.append("")
        if self.rejected:
            out += ["## Not translated", ""]
            for f in self.rejected:
                where = f"line {f.reason_line}" if f.reason_line else f"line {f.lineno}"
                out.append(f"- `{f.name}` (line {f.lineno}). {where}: {f.reason}")
                if f.hint:
                    out.append(f"  - Hint: {f.hint}")
            out.append("")
        if self.constants:
            out += ["## Module constants", ""]
            out += [f"- `{c}`" for c in self.constants]
            out.append("")
        if self.module_notes:
            out += ["## Module-level notes", ""]
            out += [f"- {n}" for n in self.module_notes]
            out.append("")
        return "\n".join(out)

    def to_text(self) -> str:
        lines = [f"qtrans: {self.filename}: {self.summary()}"]
        for f in self.functions:
            if f.translated and f.override:
                lines.append(f"  [your C++] {f.name}  ({f.override})")
            elif f.translated:
                lines.append(f"  [ok]       {f.name}")
            else:
                where = f.reason_line or f.lineno
                lines.append(f"  [rejected] {f.name}  (line {where}: {f.reason})")
                if f.hint:
                    lines.append(f"             hint: {f.hint}")
        return "\n".join(lines)
