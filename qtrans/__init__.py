"""qtrans: translate typed, scalar quant-finance Python into readable, standalone C++,
and prove the translation matches.

    from qtrans import translate_file, verify_file
    result = translate_file("black_scholes.py", "cpp_out")   # writes black_scholes.hpp
    report = verify_file("black_scholes.py", bench=True)      # compiles and compares
    print(report.to_text())
"""
from .errors import Unsupported
from .translator import VERSION as __version__
from .translator import translate_file, translate_source
from .verify import verify_file

__all__ = ["translate_file", "translate_source", "verify_file", "Unsupported", "__version__"]
