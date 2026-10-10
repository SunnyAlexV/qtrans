// How a C++ developer uses qtrans output: include the generated headers and call the
// functions. No Python, no qtrans and no third-party libraries are needed at this point.
//
// Build (from this folder):
//   g++ -std=c++17 -O2 -I../cpp_out price_book.cpp -o price_book
// MSVC:
//   cl /std:c++17 /O2 /EHsc /I..\cpp_out price_book.cpp

#include <cstdio>
#include <stdexcept>

#include "black_scholes.hpp"
#include "implied_vol.hpp"

struct Quote {
    double strike;
    double expiry;
    bool is_call;
    double market_price;
};

int main() {
    const double spot = 100.0;
    const double rate = 0.03;
    const Quote book[] = {
        {90.0, 0.25, true, 11.40}, {100.0, 0.25, true, 4.35}, {110.0, 0.25, true, 1.10},
        {90.0, 0.25, false, 0.75}, {100.0, 0.25, false, 3.60}, {110.0, 0.25, false, 10.50},
        {100.0, 1.00, true, 9.40}, {100.0, 1.00, false, 6.50}, {100.0, 1.00, true, 150.0},
    };

    std::printf("%8s %6s %5s %9s %8s %8s %8s\n", "strike", "T", "type", "mkt", "iv", "delta", "vega");
    for (const Quote& q : book) {
        try {
            const double iv = implied_vol::implied_vol(q.market_price, spot, q.strike, rate, q.expiry, q.is_call);
            const double d = black_scholes::delta(spot, q.strike, rate, iv, q.expiry, 0.0, q.is_call);
            const double v = black_scholes::vega(spot, q.strike, rate, iv, q.expiry);
            std::printf("%8.1f %6.2f %5s %9.2f %8.4f %8.4f %8.4f\n", q.strike, q.expiry,
                        q.is_call ? "call" : "put", q.market_price, iv, d, v);
        } catch (const std::invalid_argument& e) {
            // the Python `raise ValueError(...)` arrives here as std::invalid_argument
            std::printf("%8.1f %6.2f %5s %9.2f   rejected: %s\n", q.strike, q.expiry,
                        q.is_call ? "call" : "put", q.market_price, e.what());
        }
    }
    return 0;
}
