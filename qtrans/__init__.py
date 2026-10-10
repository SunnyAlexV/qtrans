"""qtrans: translate typed, scalar quant-finance Python into readable, standalone C++,
and validate it against the Python by differential testing.

    from qtrans import translate_file, verify_file
    result = translate_file("black_scholes.py", "cpp_out")   # writes black_scholes.hpp
    report = verify_file("black_scholes.py", bench=True)      # compiles and compares
    print(report.to_text())
"""
from .errors import Unsupported
from .translator import VERSION as __version__
from .project import ProjectError, translate_project
from .translator import translate_file, translate_source
from .verify import verify_file, verify_project

__all__ = ["translate_file", "translate_source", "translate_project", "verify_file", "verify_project",
           "Unsupported", "ProjectError", "__version__"]
