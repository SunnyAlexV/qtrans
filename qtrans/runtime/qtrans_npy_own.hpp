// qtrans_npy_own.hpp: NumPy's own float64 kernels: exp and log by Tang's method (AVX512F), as
// NumPy computes them where it doesn't call SVML, and the coefficients of its tanh.
// Written by tools/vendor_svml.py from numpy 2.5.4 (sdist sha256 9a94cf751c9ad8ebaa835bcd3d40dacf8534ad086b88c38029b65123c7999d2a); do not edit.
// The kernels are NumPy's code as written, with NumPy's types and macros defined for C++. Sources:
//   loops_exponent_log.dispatch.c.src    sha256 01320349e7e80c9bfd8e0e4fd0f24599b95da0bcf94f991e9989b97b2e7e7059
//   npy_simd_data.h                      sha256 befa536863dadeddec0dff116dea5412e685000e2842a9a9907e8aee35bfea7c
//   loops_hyperbolic.dispatch.cpp.src    sha256 e0154f5f0a19e0b8630a73b68838f43a29545e879b913c2f4894551bcb131104
//
// NumPy's licence (BSD-3), as for qtrans_svml_ha.hpp:
//
// Copyright (c) 2005-2021, NumPy Developers.
// All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are
// met:
//
//     * Redistributions of source code must retain the above copyright
//        notice, this list of conditions and the following disclaimer.
//
//     * Redistributions in binary form must reproduce the above
//        copyright notice, this list of conditions and the following
//        disclaimer in the documentation and/or other materials provided
//        with the distribution.
//
//     * Neither the name of the NumPy Developers nor the names of any
//        contributors may be used to endorse or promote products derived
//        from this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
// "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
// LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
// A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
// OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
// SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
// LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
// DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
// THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
// (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
// OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#pragma once

#include <cstddef>
#include <cstdint>
#include <cmath>

namespace qt {
namespace npy_own {

using npy_double = double;
using npy_float = float;
using npy_int = int;
using npy_int32 = std::int32_t;
using npy_intp = std::ptrdiff_t;
using npy_uint64 = std::uint64_t;
using npy_uint32 = std::uint32_t;

// NumPy's tanh (numpy/_core/src/umath/loops_hyperbolic.dispatch.cpp.src): row k holds coefficient k
// (b, c0, ..., c16) of each of the 16 intervals
alignas(64) static const std::uint64_t tanh_lut16x18[18 * 16] = {
    0x0ull, 0x3fcc000000000000ull, 0x3fd4000000000000ull, 0x3fdc000000000000ull, 0x3fe4000000000000ull, 0x3fec000000000000ull, 0x3ff4000000000000ull, 0x3ffc000000000000ull, 0x4004000000000000ull, 0x400c000000000000ull, 0x4014000000000000ull, 0x401c000000000000ull, 0x4024000000000000ull, 0x402c000000000000ull, 0x4034000000000000ull, 0x0ull,
    0x0ull, 0x3fcb8fd0416a7c92ull, 0x3fd35f98a0ea650eull, 0x3fda5729ee488037ull, 0x3fe1bf47eabb8f95ull, 0x3fe686650b8c2015ull, 0x3feb2523bb6b2deeull, 0x3fee1fbf97e33527ull, 0x3fef9258260a71c2ull, 0x3feff112c63a9077ull, 0x3fefff419668df11ull, 0x3feffffc832750f2ull, 0x3feffffffdc96f35ull, 0x3fefffffffffcf58ull, 0x3ff0000000000000ull, 0x3ff0000000000000ull,
    0x3ff0000000000000ull, 0x3fee842ca3f08532ull, 0x3fed11574af58f1bull, 0x3fea945b9c24e4f9ull, 0x3fe6284c3374f815ull, 0x3fe02500a09f8d6eull, 0x3fd1f25131e3a8c0ull, 0x3fbd22ca1c24a139ull, 0x3f9b3afe1fba5c76ull, 0x3f6dd37d19b22b21ull, 0x3f27ccec13a9ef96ull, 0x3ecbe6c3f33250aeull, 0x3e41b4865394f75full, 0x3d8853f01bda5f28ull, 0x3c73953c0197ef58ull, 0x0ull,
    0xbbf0b3ea3fdfaa19ull, 0xbfca48aaeb53bc21ull, 0xbfd19921f4329916ull, 0xbfd5e0f09bef8011ull, 0xbfd893b59c35c882ull, 0xbfd6ba7cb7576538ull, 0xbfce7291743d7555ull, 0xbfbb6d85a01efb80ull, 0xbf9addae58c7141aull, 0xbf6dc59376c7aa19ull, 0xbf27cc5e74677410ull, 0xbecbe6c0e8b4cc87ull, 0xbe41b486526b0565ull, 0xbd8853f01bef63a4ull, 0xbc73955be519be31ull, 0x0ull,
    0xbfd5555555555555ull, 0xbfd183afc292ba11ull, 0xbfcc1a4b039c9bfaull, 0xbfc16e1e6d8d0be6ull, 0xbf92426c751e48a2ull, 0x3fb4f152b2bad124ull, 0x3fbbba40cbef72beull, 0x3fb01ba038be6a3dull, 0x3f916df44871efc8ull, 0x3f63c6869dfc8870ull, 0x3f1fb9aef915d828ull, 0x3ec299d1e27c6e11ull, 0x3e379b5ddcca334cull, 0x3d8037f57bc62c9aull, 0x3c6a2d4b50a2cff7ull, 0x0ull,
    0xbce6863ee44ed636ull, 0x3fc04dcd0476c75eull, 0x3fc43d3449a80f08ull, 0x3fc5c26f3699b7e7ull, 0x3fc1a686f6ab2533ull, 0x3faf203c316ce730ull, 0xbf89c7a02788557cull, 0xbf98157e26e0d541ull, 0xbf807b55c1c7d278ull, 0xbf53a18d5843190full, 0xbf0fb6bbc89b1a5bull, 0xbeb299c9c684a963ull, 0xbe279b5dd4fb3d01ull, 0xbd7037f57ae72aa6ull, 0xbc5a2ca2bba78e86ull, 0x0ull,
    0x3fc1111111112ab5ull, 0x3fb5c19efdfc08adull, 0x3fa74c98dc34fbacull, 0xbf790d6a8eff0a77ull, 0xbfac3c021789a786ull, 0xbfae2196b7326859ull, 0xbf93a7a011ff8c2aull, 0x3f6e4709c7e8430eull, 0x3f67682afa611151ull, 0x3f3ef2ee77717cbfull, 0x3ef95a4482f180b7ull, 0x3e9dc2c27da3b603ull, 0x3e12e2afd9f7433eull, 0x3d59f320348679baull, 0x3c44b61d9bbcc940ull, 0x0ull,
    0xbda1ea19ddddb3b4ull, 0xbfb0b8df995ce4dfull, 0xbfb2955cf41e8164ull, 0xbfaf9d05c309f7c6ull, 0xbf987d27ccff4291ull, 0x3f8b2ca62572b098ull, 0x3f8f1cf6c7f5b00aull, 0x3f60379811e43dd5ull, 0xbf4793826f78537eull, 0xbf2405695e36240full, 0xbee0e08de39ce756ull, 0xbe83d709ba5f714eull, 0xbdf92e3fc5ee63e0ull, 0xbd414cc030f2110eull, 0xbc2ba022e8d82a87ull, 0x0ull,
    0xbfaba1ba1990520bull, 0xbf96e37bba52f6fcull, 0x3ecff7df18455399ull, 0x3f97362834d33a4eull, 0x3f9e7f8380184b45ull, 0x3f869543e7c420d4ull, 0xbf7326bd4914222aull, 0xbf5fc15b0a9d98faull, 0x3f14cffcfa69fbb6ull, 0x3f057e48e5b79d10ull, 0x3ec33b66d7d77264ull, 0x3e66ac4e578b9b10ull, 0x3ddcc74b8d3d5c42ull, 0x3d23c589137f92b4ull, 0x3c107f8e2c8707a1ull, 0x0ull,
    0xbe351ca7f096011full, 0x3f9eaaf3320c3851ull, 0x3f9cf823fe761fc1ull, 0x3f9022271754ff1full, 0xbf731fe77c9c60afull, 0xbf84a6046865ec7dull, 0xbf4ca3f1f2b9192bull, 0x3f4c77dee0afd227ull, 0x3f04055bce68597aull, 0xbee2bf0cb4a71647ull, 0xbea31eaafe73efd5ull, 0xbe46abb02c4368edull, 0xbdbcc749ca8079ddull, 0xbd03c5883836b9d2ull, 0xbbf07a5416264aecull, 0x0ull,
    0x3f9664f94e6ac14eull, 0xbf94d3343bae39ddull, 0xbf7bc748e60df843ull, 0xbf8c89372b43ba85ull, 0xbf8129a092de747aull, 0x3f60c85b4d538746ull, 0x3f5be9392199ec18ull, 0xbf2a0c68a4489f10ull, 0xbf00462601dc2faaull, 0x3eb7b6a219dea9f4ull, 0x3e80cbcc8d4c5c8aull, 0x3e2425bb231a5e29ull, 0x3d9992a4beac8662ull, 0x3ce191ba5ed3fb67ull, 0x3bc892450bad44c4ull, 0x0ull,
    0xbea8c4c1fd7852feull, 0xbfccce16b1046f13ull, 0xbf81a16f224bb7b6ull, 0xbf62cbf00406bc09ull, 0x3f75b29bb02cf69bull, 0x3f607df0f9f90c17ull, 0xbf4b852a6e0758d5ull, 0xbf0078c63d1b8445ull, 0x3eec12eadd55be7aull, 0xbe6fa600f593181bull, 0xbe5a3c935dce3f7dull, 0xbe001c6d95e3ae96ull, 0xbd74755a00ea1fd3ull, 0xbcbc1c6c063bb7acull, 0xbba3be9a4460fe00ull, 0x0ull,
    0xbf822404577aa9ddull, 0x403d8b07f7a82aa3ull, 0xbf9f44ab92fbab0aull, 0x3fb2eac604473d6aull, 0x3f45f87d903aaac8ull, 0xbf5e104671036300ull, 0x3f19bc98ddf0f340ull, 0x3f0d4304bc9246e8ull, 0xbed13c415f7b9d41ull, 0xbe722b8d9720cdb0ull, 0x3e322666d739bec0ull, 0x3dd76a553d7e7918ull, 0x3d4de0fa59416a39ull, 0x3c948716cf3681b4ull, 0x3b873f9f2d2fda99ull, 0x0ull,
    0xbefdd99a221ed573ull, 0x4070593a3735bab4ull, 0xbfccab654e44835eull, 0x3fd13ed80037dbacull, 0xbf6045b9076cc487ull, 0x3f2085ee7e8ac170ull, 0x3f23524622610430ull, 0xbeff12a6626911b4ull, 0x3eab9008bca408afull, 0x3e634df71865f620ull, 0xbe05bb1bcf83ca73ull, 0xbdaf2ac143fb6762ull, 0xbd23eae52a3dbf57ull, 0xbc6b5e3e9ca0955eull, 0xbb5eca68e2c1ba2eull, 0x0ull,
    0x3f6e3be689423841ull, 0xc0d263511f5baac1ull, 0x40169f73b15ebe5cull, 0xc025c1dd41cd6cb5ull, 0xbf58fd89fe05e0d1ull, 0x3f73f7af01d5af7aull, 0xbf1e40bdead17e6bull, 0x3ee224cd6c4513e5ull, 0xbe24b645e68eeaa3ull, 0xbe4abfebfb72bc83ull, 0x3dd51c38f8695ed3ull, 0x3d8313ac38c6832bull, 0x3cf7787935626685ull, 0x3c401ffc49c6bc29ull, 0xbabf0b21acfa52abull, 0x0ull,
    0xbf2a1306713a4f3aull, 0xc1045e509116b066ull, 0x4041fab9250984ceull, 0xc0458d090ec3de95ull, 0xbf74949d60113d63ull, 0x3f7c9fd6200d0adeull, 0x3f02cd40e0ad0a9full, 0xbe858ab8e019f311ull, 0xbe792fa6323b7cf8ull, 0x3e2df04d67876402ull, 0xbd95c72be95e4d2cull, 0xbd55a89c30203106ull, 0xbccad6b3bb9eff65ull, 0xbc12705ccd3dd884ull, 0xba8e0a4c47ae75f5ull, 0x0ull,
    0xbf55d7e76dc56871ull, 0x41528c38809c90c7ull, 0xc076d57fb5190b02ull, 0x4085f09f888f8adaull, 0x3fa246332a2fcba5ull, 0xbfb29d851a896fcdull, 0x3ed9065ae369b212ull, 0xbeb8e1ba4c98a030ull, 0x3e6ffd0766ad4016ull, 0xbe0c63c29f505f5bull, 0xbd7fab216b9e0e49ull, 0x3d2826b62056aa27ull, 0x3ca313e31762f523ull, 0x3bea37aa21895319ull, 0x3ae5c7f1fd871496ull, 0x0ull,
    0x3f35e67ab76a26e7ull, 0x41848ee0627d8206ull, 0xc0a216d618b489ecull, 0x40a5b89107c8af4full, 0x3fb69d8374520edaull, 0xbfbded519f981716ull, 0xbef02d288b5b3371ull, 0x3eb290981209c1a6ull, 0xbe567e924bf5ff6eull, 0x3de3f7f7de6b0eb6ull, 0x3d69ed18bae3ebbcull, 0xbcf7534c4f3dfa71ull, 0xbc730b73f1eaff20ull, 0xbbba2cff8135d462ull, 0xbab5a71b5f7d9035ull, 0x0ull,
};

}  // namespace npy_own
}  // namespace qt

#if defined(__x86_64__) && (defined(__GNUC__) || defined(__clang__)) && !defined(_WIN32)
#include <immintrin.h>
#define QT_NPY_OWN_TANG 1
#if defined(__GNUC__)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wconversion"
#pragma GCC diagnostic ignored "-Wsign-conversion"
#pragma GCC diagnostic ignored "-Wunused-parameter"
#pragma GCC diagnostic ignored "-Wunused-function"
#pragma GCC diagnostic ignored "-Wold-style-cast"
#pragma GCC diagnostic ignored "-Wshadow"
#pragma GCC diagnostic ignored "-Wcast-align"
#pragma GCC diagnostic ignored "-Wdouble-promotion"
#endif
namespace qt {
namespace npy_own {

#define QT_NPY_TANG_TARGET __attribute__((target("avx512f")))
#define NPY_FINLINE static inline QT_NPY_TANG_TARGET
#define NPY_INFINITY __builtin_inf()
#define NPY_NAN (static_cast<double>(__builtin_nanf("")))
#define NPY_DECL_ALIGNED(x) __attribute__((aligned(x)))
#define npyv_tobits_b64(m) (static_cast<unsigned>(m))
#define npy_log std::log
inline void npy_set_floatstatus_overflow() {}
inline void npy_set_floatstatus_underflow() {}
inline void npy_set_floatstatus_invalid() {}
inline void npy_set_floatstatus_divbyzero() {}

// numpy/_core/src/umath/npy_simd_data.h (its tables are for builds with AVX512F and without SVML)
#define NPY_HAVE_AVX512F 1

#if defined NPY_HAVE_AVX512F
#if !(defined(__clang__) && (__clang_major__ < 10 || (__clang_major__ == 10 && __clang_minor__ < 1)))
/*
 * Constants used in vector implementation of float64 exp(x)
 */
#define NPY_RINT_CVT_MAGIC 0x1.8p52
#define NPY_INV_LN2_MUL_32 0x1.71547652b82fep+5
#define NPY_TANG_NEG_L1 -0x1.62e42fefp-6
#define NPY_TANG_NEG_L2 -0x1.473de6af278edp-39
#define NPY_TANG_A1 0x1p-1
#define NPY_TANG_A2 0x1.5555555548f7cp-3
#define NPY_TANG_A3 0x1.5555555545d4ep-5
#define NPY_TANG_A4 0x1.11115b7aa905ep-7
#define NPY_TANG_A5 0x1.6c1728d739765p-10

#if !defined NPY_HAVE_AVX512_SKX || !defined NPY_CAN_LINK_SVML
/* Lookup table for 2^(j/32) */
alignas(64) static const npy_uint64 EXP_Table_top[32] = {
    0x3FF0000000000000,
    0x3FF059B0D3158540,
    0x3FF0B5586CF98900,
    0x3FF11301D0125B40,
    0x3FF172B83C7D5140,
    0x3FF1D4873168B980,
    0x3FF2387A6E756200,
    0x3FF29E9DF51FDEC0,
    0x3FF306FE0A31B700,
    0x3FF371A7373AA9C0,
    0x3FF3DEA64C123400,
    0x3FF44E0860618900,
    0x3FF4BFDAD5362A00,
    0x3FF5342B569D4F80,
    0x3FF5AB07DD485400,
    0x3FF6247EB03A5580,
    0x3FF6A09E667F3BC0,
    0x3FF71F75E8EC5F40,
    0x3FF7A11473EB0180,
    0x3FF82589994CCE00,
    0x3FF8ACE5422AA0C0,
    0x3FF93737B0CDC5C0,
    0x3FF9C49182A3F080,
    0x3FFA5503B23E2540,
    0x3FFAE89F995AD380,
    0x3FFB7F76F2FB5E40,
    0x3FFC199BDD855280,
    0x3FFCB720DCEF9040,
    0x3FFD5818DCFBA480,
    0x3FFDFC97337B9B40,
    0x3FFEA4AFA2A490C0,
    0x3FFF50765B6E4540,
};

alignas(64) static const npy_uint64 EXP_Table_tail[32] = {
    0x0000000000000000,
    0x3D0A1D73E2A475B4,
    0x3CEEC5317256E308,
    0x3CF0A4EBBF1AED93,
    0x3D0D6E6FBE462876,
    0x3D053C02DC0144C8,
    0x3D0C3360FD6D8E0B,
    0x3D009612E8AFAD12,
    0x3CF52DE8D5A46306,
    0x3CE54E28AA05E8A9,
    0x3D011ADA0911F09F,
    0x3D068189B7A04EF8,
    0x3D038EA1CBD7F621,
    0x3CBDF0A83C49D86A,
    0x3D04AC64980A8C8F,
    0x3CD2C7C3E81BF4B7,
    0x3CE921165F626CDD,
    0x3D09EE91B8797785,
    0x3CDB5F54408FDB37,
    0x3CF28ACF88AFAB35,
    0x3CFB5BA7C55A192D,
    0x3D027A280E1F92A0,
    0x3CF01C7C46B071F3,
    0x3CFC8B424491CAF8,
    0x3D06AF439A68BB99,
    0x3CDBAA9EC206AD4F,
    0x3CFC2220CB12A092,
    0x3D048A81E5E8F4A5,
    0x3CDC976816BAD9B8,
    0x3CFEB968CAC39ED3,
    0x3CF9858F73A18F5E,
    0x3C99D3E12DD8A18B,
};
#endif //#if !defined NPY_HAVE_AVX512_SKX || !defined NPY_CAN_LINK_SVML
#endif
#endif

/*
 * Constants used in vector implementation of exp(x)
 */
#define NPY_RINT_CVT_MAGICf 0x1.800000p+23f
#define NPY_CODY_WAITE_LOGE_2_HIGHf -6.93145752e-1f
#define NPY_CODY_WAITE_LOGE_2_LOWf -1.42860677e-6f
#define NPY_COEFF_P0_EXPf 9.999999999980870924916e-01f
#define NPY_COEFF_P1_EXPf 7.257664613233124478488e-01f
#define NPY_COEFF_P2_EXPf 2.473615434895520810817e-01f
#define NPY_COEFF_P3_EXPf 5.114512081637298353406e-02f
#define NPY_COEFF_P4_EXPf 6.757896990527504603057e-03f
#define NPY_COEFF_P5_EXPf 5.082762527590693718096e-04f
#define NPY_COEFF_Q0_EXPf 1.000000000000000000000e+00f
#define NPY_COEFF_Q1_EXPf -2.742335390411667452936e-01f
#define NPY_COEFF_Q2_EXPf 2.159509375685829852307e-02f

/*
 * Constants used in vector implementation of log(x)
 */
#define NPY_COEFF_P0_LOGf 0.000000000000000000000e+00f
#define NPY_COEFF_P1_LOGf 9.999999999999998702752e-01f
#define NPY_COEFF_P2_LOGf 2.112677543073053063722e+00f
#define NPY_COEFF_P3_LOGf 1.480000633576506585156e+00f
#define NPY_COEFF_P4_LOGf 3.808837741388407920751e-01f
#define NPY_COEFF_P5_LOGf 2.589979117907922693523e-02f
#define NPY_COEFF_Q0_LOGf 1.000000000000000000000e+00f
#define NPY_COEFF_Q1_LOGf 2.612677543073109236779e+00f
#define NPY_COEFF_Q2_LOGf 2.453006071784736363091e+00f
#define NPY_COEFF_Q3_LOGf 9.864942958519418960339e-01f
#define NPY_COEFF_Q4_LOGf 1.546476374983906719538e-01f
#define NPY_COEFF_Q5_LOGf 5.875095403124574342950e-03f

/*
 * Lookup table of log(c_k)
 * Reference form: Tang, Ping-Tak Peter. "Table-driven implementation of the
 *     logarithm function in IEEE floating-point arithmetic." ACM Transactions
 *     on Mathematical Software (TOMS) 16.4 (1990): 378-400.
 */
#if defined NPY_HAVE_AVX512F
#if !(defined(__clang__) && (__clang_major__ < 10 || (__clang_major__ == 10 && __clang_minor__ < 1)))
#if !defined NPY_HAVE_AVX512_SKX || !defined NPY_CAN_LINK_SVML
alignas(64) static const npy_uint64 LOG_TABLE_TOP[64] = {
    0x0000000000000000,
    0x3F8FC0A8B1000000,
    0x3F9F829B0E780000,
    0x3FA77458F6340000,
    0x3FAF0A30C0100000,
    0x3FB341D7961C0000,
    0x3FB6F0D28AE60000,
    0x3FBA926D3A4A0000,
    0x3FBE27076E2A0000,
    0x3FC0D77E7CD10000,
    0x3FC29552F8200000,
    0x3FC44D2B6CCB0000,
    0x3FC5FF3070A80000,
    0x3FC7AB8902110000,
    0x3FC9525A9CF40000,
    0x3FCAF3C94E810000,
    0x3FCC8FF7C79B0000,
    0x3FCE27076E2B0000,
    0x3FCFB9186D5E0000,
    0x3FD0A324E2738000,
    0x3FD1675CABAB8000,
    0x3FD22941FBCF8000,
    0x3FD2E8E2BAE10000,
    0x3FD3A64C55698000,
    0x3FD4618BC21C8000,
    0x3FD51AAD872E0000,
    0x3FD5D1BDBF580000,
    0x3FD686C81E9B0000,
    0x3FD739D7F6BC0000,
    0x3FD7EAF83B828000,
    0x3FD89A3386C18000,
    0x3FD947941C210000,
    0x3FD9F323ECBF8000,
    0x3FDA9CEC9A9A0000,
    0x3FDB44F77BCC8000,
    0x3FDBEB4D9DA70000,
    0x3FDC8FF7C79A8000,
    0x3FDD32FE7E010000,
    0x3FDDD46A04C20000,
    0x3FDE744261D68000,
    0x3FDF128F5FAF0000,
    0x3FDFAF588F790000,
    0x3FE02552A5A5C000,
    0x3FE0723E5C1CC000,
    0x3FE0BE72E4254000,
    0x3FE109F39E2D4000,
    0x3FE154C3D2F4C000,
    0x3FE19EE6B467C000,
    0x3FE1E85F5E704000,
    0x3FE23130D7BEC000,
    0x3FE2795E1289C000,
    0x3FE2C0E9ED448000,
    0x3FE307D7334F0000,
    0x3FE34E289D9D0000,
    0x3FE393E0D3564000,
    0x3FE3D9026A714000,
    0x3FE41D8FE8468000,
    0x3FE4618BC21C4000,
    0x3FE4A4F85DB04000,
    0x3FE4E7D811B74000,
    0x3FE52A2D265BC000,
    0x3FE56BF9D5B40000,
    0x3FE5AD404C358000,
    0x3FE5EE02A9240000,
};

alignas(64) static const npy_uint64 LOG_TABLE_TAIL[64] = {
    0x0000000000000000,
    0xBD5FE0E183092C59,
    0x3D2980267C7E09E4,
    0xBD62303B9CB0D5E1,
    0x3D662A6617CC9717,
    0xBD4717B6B33E44F8,
    0xBD62968C836CC8C2,
    0x3D6AAC6CA17A4554,
    0x3D6E5CBD3D50FFFC,
    0xBD6C69A65A23A170,
    0xBD35B967F4471DFC,
    0x3D6F4799F4F6543E,
    0xBD6B0B0DE3077D7E,
    0xBD537B720E4A694B,
    0x3D65AD1D904C1D4E,
    0xBD600349CC67F9B2,
    0xBD697794F689F843,
    0xBD3A342C2AF0003C,
    0x3D5F1546AAA3361C,
    0x3D50E35F73F7A018,
    0x3D630701CE63EAB9,
    0xBD3A6976F5EB0963,
    0x3D5D309C2CC91A85,
    0xBD6D0B1C68651946,
    0xBD609EC17A426426,
    0xBD3F4BD8DB0A7CC1,
    0x3D4394A11B1C1EE4,
    0x3D54AEC442BE1015,
    0xBD67FCB18ED9D603,
    0x3D67E1B259D2F3DA,
    0xBD6ED2A52C73BF78,
    0x3D56FABA4CDD147D,
    0x3D584BF2B68D766F,
    0x3D40931A909FEA5E,
    0x3D4EC5197DDB55D3,
    0x3D5B7BF7861D37AC,
    0x3D5A21AC25DB1EF3,
    0xBD542A9E21373414,
    0xBD6DAFA08CECADB1,
    0x3D3E1F8DF68DBCF3,
    0x3D3BB2CD720EC44C,
    0xBD49C24CA098362B,
    0x3D60FEC69C695D7F,
    0x3D6F404E57963891,
    0xBD657D49676844CC,
    0x3D592DFBC7D93617,
    0x3D65E9A98F33A396,
    0x3D52DD98B97BAEF0,
    0x3D1A07BD8B34BE7C,
    0xBD17AFA4392F1BA7,
    0xBD5DCA290F818480,
    0x3D5D1772F5386374,
    0x3D60BE1FB590A1F5,
    0xBD6E2CE9146D271A,
    0xBD65E6563BBD9FC9,
    0x3D66FAA404263D0B,
    0xBD5AA33736867A17,
    0x3D6EC27D0B7B37B3,
    0xBD244FDD840B8591,
    0x3D6BB09CB0985646,
    0x3D46ABB9DF22BC57,
    0xBD58CD7DC73BD194,
    0x3D6F2CFB29AAA5F0,
    0x3D66757006095FD2,
};
#endif //#if !defined NPY_HAVE_AVX512_SKX || !defined NPY_CAN_LINK_SVML

#define NPY_TANG_LOG_A1 0x1.55555555554e6p-4
#define NPY_TANG_LOG_A2 0x1.9999999bac6d4p-7
#define NPY_TANG_LOG_A3 0x1.2492307f1519fp-9
#define NPY_TANG_LOG_A4 0x1.c8034c85dfffp-12

#define NPY_TANG_LOG_LN2HI 0x1.62e42fefa4p-1
#define NPY_TANG_LOG_LN2LO -0x1.8432a1b0e2634p-43
#endif
#endif


#undef NPY_HAVE_AVX512F

// numpy/_core/src/umath/loops_exponent_log.dispatch.c.src

NPY_FINLINE __mmask16
avx512_get_full_load_mask_ps(void)
{
    return 0xFFFF;
}

NPY_FINLINE __mmask8
avx512_get_full_load_mask_pd(void)
{
    return 0xFF;
}

NPY_FINLINE __mmask16
avx512_get_partial_load_mask_ps(const npy_int num_elem, const npy_int total_elem)
{
    return (0x0001 << num_elem) - 0x0001;
}

NPY_FINLINE __mmask8
avx512_get_partial_load_mask_pd(const npy_int num_elem, const npy_int total_elem)
{
    return (0x01 << num_elem) - 0x01;
}

NPY_FINLINE __m512
avx512_masked_gather_ps(__m512 src,
                        npy_float* addr,
                        __m512i vindex,
                        __mmask16 kmask)
{
    return _mm512_mask_i32gather_ps(src, kmask, vindex, addr, 4);
}

NPY_FINLINE __m512d
avx512_masked_gather_pd(__m512d src,
                        npy_double* addr,
                        __m256i vindex,
                        __mmask8 kmask)
{
    return _mm512_mask_i32gather_pd(src, kmask, vindex, addr, 8);
}

NPY_FINLINE __m512
avx512_masked_load_ps(__mmask16 mask, npy_float* addr)
{
    return _mm512_maskz_loadu_ps(mask, (__m512 *)addr);
}

NPY_FINLINE __m512d
avx512_masked_load_pd(__mmask8 mask, npy_double* addr)
{
    return _mm512_maskz_loadu_pd(mask, (__m512d *)addr);
}

NPY_FINLINE __m512
avx512_set_masked_lanes_ps(__m512 x, __m512 val, __mmask16 mask)
{
    return _mm512_mask_blend_ps(mask, x, val);
}

NPY_FINLINE __m512d
avx512_set_masked_lanes_pd(__m512d x, __m512d val, __mmask8 mask)
{
    return _mm512_mask_blend_pd(mask, x, val);
}

NPY_FINLINE __m512
avx512_blend(__m512 x, __m512 y, __mmask16 ymask)
{
    return _mm512_mask_mov_ps(x, ymask, y);
}

NPY_FINLINE __m512
avx512_get_exponent(__m512 x)
{
    return _mm512_add_ps(_mm512_getexp_ps(x), _mm512_set1_ps(1.0f));
}

NPY_FINLINE __m512
avx512_get_mantissa(__m512 x)
{
    return _mm512_getmant_ps(x, _MM_MANT_NORM_p5_1, _MM_MANT_SIGN_src);
}

NPY_FINLINE __m512
avx512_scalef_ps(__m512 poly, __m512 quadrant)
{
    return _mm512_scalef_ps(poly, quadrant);
}

NPY_FINLINE __m512d
avx512_permute_x4var_pd(__m512d t0,
                        __m512d t1,
                        __m512d t2,
                        __m512d t3,
                        __m512i index)
{
    __mmask8 lut_mask = _mm512_cmp_epi64_mask(
                          _mm512_and_epi64(_mm512_set1_epi64(0x10ULL), index),
                          _mm512_set1_epi64(0), _MM_CMPINT_GT);
    __m512d res1 = _mm512_permutex2var_pd(t0, index, t1);
    __m512d res2 = _mm512_permutex2var_pd(t2, index, t3);
    return _mm512_mask_blend_pd(lut_mask, res1, res2);
}

NPY_FINLINE __m512d
avx512_permute_x8var_pd(__m512d t0, __m512d t1, __m512d t2, __m512d t3,
                        __m512d t4, __m512d t5, __m512d t6, __m512d t7,
                        __m512i index)
{
    __mmask8 lut_mask = _mm512_cmp_epi64_mask(
                          _mm512_and_epi64(_mm512_set1_epi64(0x20ULL), index),
                          _mm512_set1_epi64(0), _MM_CMPINT_GT);
    __m512d res1 = avx512_permute_x4var_pd(t0, t1, t2, t3, index);
    __m512d res2 = avx512_permute_x4var_pd(t4, t5, t6, t7, index);
    return _mm512_mask_blend_pd(lut_mask, res1, res2);
}


static QT_NPY_TANG_TARGET void
AVX512F_exp_DOUBLE(npy_double * op,
                npy_double * ip,
                const npy_intp array_size,
                const npy_intp steps)
{
    npy_intp num_remaining_elements = array_size;
    const npy_intp stride = steps / (npy_intp)sizeof(npy_double);
    const npy_int num_lanes = 64 / (npy_intp)sizeof(npy_double);
    npy_int32 indexarr[8];
    for (npy_int32 ii = 0; ii < 8; ii++) {
        indexarr[ii] = ii*stride;
    }

    __m512d InvLn2N = _mm512_set1_pd(NPY_INV_LN2_MUL_32);
    __m512d mShift = _mm512_set1_pd(NPY_RINT_CVT_MAGIC);
    __m512d mNegL1 = _mm512_set1_pd(NPY_TANG_NEG_L1);
    __m512d mNegL2 = _mm512_set1_pd(NPY_TANG_NEG_L2);
    __m512i mMod = _mm512_set1_epi64(0x1f);
    __m512d mA1 = _mm512_set1_pd(NPY_TANG_A1);
    __m512d mA2 = _mm512_set1_pd(NPY_TANG_A2);
    __m512d mA3 = _mm512_set1_pd(NPY_TANG_A3);
    __m512d mA4 = _mm512_set1_pd(NPY_TANG_A4);
    __m512d mA5 = _mm512_set1_pd(NPY_TANG_A5);
    __m512d mTH_nearzero = _mm512_set1_pd(0x1p-54);
    __m512d mTH_max = _mm512_set1_pd(0x1.62e42fefa39efp+9);
    __m512d mTH_min = _mm512_set1_pd(-0x1.74910d52d3053p+9);
    __m512d mTH_inf = _mm512_set1_pd(NPY_INFINITY);
    __m512d mTH_ninf = _mm512_set1_pd(-NPY_INFINITY);
    __m512d zeros_d = _mm512_set1_pd(0.0f);
    __m512d ones_d = _mm512_set1_pd(1.0f);
    __m256i vindex = _mm256_loadu_si256((__m256i*)&indexarr[0]);

    __m512d mTable_top_0 = _mm512_loadu_pd(&(EXP_Table_top[8*0]));
    __m512d mTable_top_1 = _mm512_loadu_pd(&(EXP_Table_top[8*1]));
    __m512d mTable_top_2 = _mm512_loadu_pd(&(EXP_Table_top[8*2]));
    __m512d mTable_top_3 = _mm512_loadu_pd(&(EXP_Table_top[8*3]));
    __m512d mTable_tail_0 = _mm512_loadu_pd(&(EXP_Table_tail[8*0]));
    __m512d mTable_tail_1 = _mm512_loadu_pd(&(EXP_Table_tail[8*1]));
    __m512d mTable_tail_2 = _mm512_loadu_pd(&(EXP_Table_tail[8*2]));
    __m512d mTable_tail_3 = _mm512_loadu_pd(&(EXP_Table_tail[8*3]));

    __mmask8 overflow_mask = avx512_get_partial_load_mask_pd(0, num_lanes);
    __mmask8 underflow_mask = avx512_get_partial_load_mask_pd(0, num_lanes);
    __mmask8 load_mask = avx512_get_full_load_mask_pd();
    __mmask8 xmin_mask, xmax_mask, inf_mask, ninf_mask, nan_mask, nearzero_mask;

    while (num_remaining_elements > 0) {
        if (num_remaining_elements < num_lanes) {
            load_mask = avx512_get_partial_load_mask_pd(num_remaining_elements,
                                                      num_lanes);
        }

        __m512d x;
        if (1 == stride) {
            x = avx512_masked_load_pd(load_mask, ip);
        }
        else {
            x = avx512_masked_gather_pd(zeros_d, ip, vindex, load_mask);
        }

        nan_mask = _mm512_cmp_pd_mask(x, x, _CMP_NEQ_UQ);
        x = avx512_set_masked_lanes_pd(x, zeros_d, nan_mask);
        xmax_mask = _mm512_cmp_pd_mask(x, mTH_max, _CMP_GT_OQ);
        xmin_mask = _mm512_cmp_pd_mask(x, mTH_min, _CMP_LT_OQ);
        inf_mask = _mm512_cmp_pd_mask(x, mTH_inf, _CMP_EQ_OQ);
        ninf_mask = _mm512_cmp_pd_mask(x, mTH_ninf, _CMP_EQ_OQ);
        __m512i x_abs = _mm512_and_epi64(_mm512_castpd_si512(x),
                                _mm512_set1_epi64(0x7FFFFFFFFFFFFFFF));
        nearzero_mask = _mm512_cmp_pd_mask(_mm512_castsi512_pd(x_abs),
                                    mTH_nearzero, _CMP_LT_OQ);
        nearzero_mask = _mm512_kxor(nearzero_mask, nan_mask);
        overflow_mask = _mm512_kor(overflow_mask,
                                _mm512_kxor(xmax_mask, inf_mask));
        underflow_mask = _mm512_kor(underflow_mask,
                                _mm512_kxor(xmin_mask, ninf_mask));
        x = avx512_set_masked_lanes_pd(x, zeros_d,
                        _mm512_kor(_mm512_kor(nan_mask, xmin_mask),
                            _mm512_kor(xmax_mask, nearzero_mask)));

        /* z = x * 32/ln2 */
        __m512d z = _mm512_mul_pd(x, InvLn2N);

        /* round to nearest */
        __m512d kd = _mm512_add_pd(z, mShift);
        __m512i ki = _mm512_castpd_si512(kd);
        kd = _mm512_sub_pd(kd, mShift);

        /* r = (x + kd*mNegL1) + kd*mNegL2 */
        __m512d r1 = _mm512_fmadd_pd(kd, mNegL1, x);
        __m512d r2 = _mm512_mul_pd(kd, mNegL2);
        __m512d r = _mm512_add_pd(r1,r2);

        /* Polynomial approximation for exp(r) - 1 */
        __m512d q = _mm512_fmadd_pd(mA5, r, mA4);
        q = _mm512_fmadd_pd(q, r, mA3);
        q = _mm512_fmadd_pd(q, r, mA2);
        q = _mm512_fmadd_pd(q, r, mA1);
        q = _mm512_mul_pd(q, r);
        __m512d p = _mm512_fmadd_pd(r, q, r2);
        p = _mm512_add_pd(r1, p);

        /* Get 2^(j/32) from lookup table */
        __m512i j = _mm512_and_epi64(ki, mMod);
        __m512d top = avx512_permute_x4var_pd(mTable_top_0, mTable_top_1,
                                  mTable_top_2, mTable_top_3, j);
        __m512d tail = avx512_permute_x4var_pd(mTable_tail_0, mTable_tail_1,
                                  mTable_tail_2, mTable_tail_3, j);

        /*
         * s = top + tail;
         * exp(x) = 2^m * (top + (tail + s * p));
         */
        __m512d s = _mm512_add_pd(top, tail);
        __m512d res = _mm512_fmadd_pd(s, p, tail);
        res = _mm512_add_pd(res, top);
        res= _mm512_scalef_pd(res, _mm512_div_pd(kd, _mm512_set1_pd(32)));

        /* return special cases */
        res = avx512_set_masked_lanes_pd(res, _mm512_add_pd(x, ones_d),
                                        nearzero_mask);
        res = avx512_set_masked_lanes_pd(res, _mm512_set1_pd(NPY_NAN),
                                        nan_mask);
        res = avx512_set_masked_lanes_pd(res, mTH_inf, xmax_mask);
        res = avx512_set_masked_lanes_pd(res, zeros_d, xmin_mask);

        _mm512_mask_storeu_pd(op, load_mask, res);

        ip += num_lanes * stride;
        op += num_lanes;
        num_remaining_elements -= num_lanes;
    }
    /*
     * Don't count on the compiler for cast between mask and int registers.
     * On gcc7 with flags -march>=nocona -O3 can cause FP stack overflow
     * which may lead to putting NaN into certain HW/FP calculations.
     *
     * For more details, please check the comments in:
     * - https://github.com/numpy/numpy/issues/20356
     */
    if (npyv_tobits_b64(overflow_mask)) {
        npy_set_floatstatus_overflow();
    }

    if (npyv_tobits_b64(underflow_mask)) {
        npy_set_floatstatus_underflow();
    }
}


#if defined(__clang__)
    #if defined(__apple_build_version__)
    // Apple Clang
        #if __apple_build_version__ > 11000000
        // Apple Clang after v11
        #define WORKAROUND_LLVM__mm512_mask_mul_pd
        #endif
    #else
    // Clang, not Apple Clang
        #if __clang_major__ > 9
        // Clang v9+
        #define WORKAROUND_LLVM__mm512_mask_mul_pd
        #endif
    #endif
#endif

static QT_NPY_TANG_TARGET void
AVX512F_log_DOUBLE(npy_double * op,
                npy_double * ip,
                const npy_intp array_size,
                const npy_intp steps)
{
    npy_intp num_remaining_elements = array_size;
    const npy_intp stride = steps / (npy_intp)sizeof(npy_double);
    const npy_int num_lanes = 64 / (npy_intp)sizeof(npy_double);
    npy_int32 indexarr[8];
    for (npy_int32 ii = 0; ii < 8; ii++) {
        indexarr[ii] = ii*stride;
    }

    __m512d zeros_d = _mm512_set1_pd(0.0f);
    __m512d ones_d = _mm512_set1_pd(1.0f);
    __m512d mInf = _mm512_set1_pd(NPY_INFINITY);
    __m512d mInv64 = _mm512_castsi512_pd(_mm512_set1_epi64(0x3f90000000000000));
    __m512d mNeg_nan = _mm512_set1_pd(-NPY_NAN);
    __m512d mNan = _mm512_set1_pd(NPY_NAN);
    __m512d mNeg_inf = _mm512_set1_pd(-NPY_INFINITY);
    __m512d mA1 = _mm512_set1_pd(NPY_TANG_LOG_A1);
    __m512d mA2 = _mm512_set1_pd(NPY_TANG_LOG_A2);
    __m512d mA3 = _mm512_set1_pd(NPY_TANG_LOG_A3);
    __m512d mA4 = _mm512_set1_pd(NPY_TANG_LOG_A4);
    __m512d mLN2HI = _mm512_set1_pd(NPY_TANG_LOG_LN2HI);
    __m512d mLN2LO = _mm512_set1_pd(NPY_TANG_LOG_LN2LO);

    __m512d mTo_glibc_min = _mm512_set1_pd(1.0 - 0x1p-4);
    __m512d mTo_glibc_max = _mm512_set1_pd(1.0 + 0x1.09p-4);
    __m256i vindex = _mm256_loadu_si256((__m256i*)&indexarr[0]);

    /* Load lookup table data */
    __m512d mLUT_TOP_0 = _mm512_loadu_pd(&(LOG_TABLE_TOP[8*0]));
    __m512d mLUT_TAIL_0 = _mm512_loadu_pd(&(LOG_TABLE_TAIL[8*0]));
    __m512d mLUT_TOP_1 = _mm512_loadu_pd(&(LOG_TABLE_TOP[8*1]));
    __m512d mLUT_TAIL_1 = _mm512_loadu_pd(&(LOG_TABLE_TAIL[8*1]));
    __m512d mLUT_TOP_2 = _mm512_loadu_pd(&(LOG_TABLE_TOP[8*2]));
    __m512d mLUT_TAIL_2 = _mm512_loadu_pd(&(LOG_TABLE_TAIL[8*2]));
    __m512d mLUT_TOP_3 = _mm512_loadu_pd(&(LOG_TABLE_TOP[8*3]));
    __m512d mLUT_TAIL_3 = _mm512_loadu_pd(&(LOG_TABLE_TAIL[8*3]));
    __m512d mLUT_TOP_4 = _mm512_loadu_pd(&(LOG_TABLE_TOP[8*4]));
    __m512d mLUT_TAIL_4 = _mm512_loadu_pd(&(LOG_TABLE_TAIL[8*4]));
    __m512d mLUT_TOP_5 = _mm512_loadu_pd(&(LOG_TABLE_TOP[8*5]));
    __m512d mLUT_TAIL_5 = _mm512_loadu_pd(&(LOG_TABLE_TAIL[8*5]));
    __m512d mLUT_TOP_6 = _mm512_loadu_pd(&(LOG_TABLE_TOP[8*6]));
    __m512d mLUT_TAIL_6 = _mm512_loadu_pd(&(LOG_TABLE_TAIL[8*6]));
    __m512d mLUT_TOP_7 = _mm512_loadu_pd(&(LOG_TABLE_TOP[8*7]));
    __m512d mLUT_TAIL_7 = _mm512_loadu_pd(&(LOG_TABLE_TAIL[8*7]));

    __mmask8 load_mask = avx512_get_full_load_mask_pd();
    __mmask8 invalid_mask = avx512_get_partial_load_mask_pd(0, num_lanes);
    __mmask8 divide_by_zero_mask = invalid_mask;

    __mmask8 inf_mask, nan_mask, zero_mask, negx_mask, denormal_mask,
             glibc_mask;

    __m512d x_in;
    while (num_remaining_elements > 0) {
        if (num_remaining_elements < num_lanes) {
            load_mask = avx512_get_partial_load_mask_pd(num_remaining_elements,
                                                      num_lanes);
        }

        if (1 == stride) {
            x_in = avx512_masked_load_pd(load_mask, ip);
        }
        else {
            x_in = avx512_masked_gather_pd(zeros_d, ip, vindex, load_mask);
        }

        /* call glibc when x on [1.0 - 0x1p-4, 1.0 + 0x1.09p-4] */
        __mmask8 m1 = _mm512_cmp_pd_mask(x_in, mTo_glibc_max, _CMP_LT_OQ);
        __mmask8 m2 = _mm512_cmp_pd_mask(x_in, mTo_glibc_min, _CMP_GT_OQ);
        glibc_mask =  m1 & m2;

        if (glibc_mask != 0xFF) {
            zero_mask = _mm512_cmp_pd_mask(x_in, zeros_d, _CMP_EQ_OQ);
            inf_mask = _mm512_cmp_pd_mask(x_in, mInf, _CMP_EQ_OQ);
            negx_mask = _mm512_cmp_pd_mask(x_in, zeros_d, _CMP_LT_OQ);
            nan_mask = _mm512_cmp_pd_mask(x_in, x_in, _CMP_NEQ_UQ);

            divide_by_zero_mask = divide_by_zero_mask | (zero_mask & load_mask);
            invalid_mask = invalid_mask | negx_mask;

            __m512d x = avx512_set_masked_lanes_pd(x_in, zeros_d, negx_mask);
            __m512i ix = _mm512_castpd_si512(x);

            /* Normalize x when it is denormal */
            __m512i top12 = _mm512_and_epi64(ix,
                                _mm512_set1_epi64(0xfff0000000000000));
            denormal_mask = _mm512_cmp_epi64_mask(top12, _mm512_set1_epi64(0),
                                _CMP_EQ_OQ);
            denormal_mask = (~zero_mask) & denormal_mask;
            __m512d masked_x = x;
            #ifdef WORKAROUND_LLVM__mm512_mask_mul_pd
            masked_x = avx512_set_masked_lanes_pd(masked_x, zeros_d, (~denormal_mask));
            #endif
            ix = _mm512_castpd_si512(_mm512_mask_mul_pd(x, denormal_mask,
                                    masked_x, _mm512_set1_pd(0x1p52)));
            ix = _mm512_mask_sub_epi64(ix, denormal_mask,
                                    ix, _mm512_set1_epi64(52ULL << 52));

            /*
             * x = 2^k * z; where z in range [1,2]
             */
            __m512i tmp = _mm512_sub_epi64(ix,
                              _mm512_set1_epi64(0x3ff0000000000000));
            __m512i i = _mm512_and_epi64(_mm512_srai_epi64(tmp, 52 - 6),
                            _mm512_set1_epi64(0x3fULL));
            __m512i ik = _mm512_srai_epi64(tmp, 52);
            __m512d z = _mm512_castsi512_pd(_mm512_sub_epi64(ix, _mm512_and_epi64(tmp,
                            _mm512_set1_epi64(0xfff0000000000000))));
            /* c = i/64 + 1 */
            __m256i i_32 = _mm512_cvtepi64_epi32(i);
            __m512d c = _mm512_fmadd_pd(_mm512_cvtepi32_pd(i_32), mInv64, ones_d);

            /* u = 2 * (z - c) / (z + c) */
            __m512d u = _mm512_div_pd(_mm512_sub_pd(z, c), _mm512_add_pd(z, c));
            u = _mm512_mul_pd(_mm512_set1_pd(2.0), u);

            /* v = u * u */
            __m512d v = _mm512_mul_pd(u,u);

            /* log(z/c) = u + u*v*(A1 + v*(A2 + v*(A3 + v*A4))) */
            __m512d res = _mm512_fmadd_pd(v, mA4, mA3);
            res = _mm512_fmadd_pd(v, res, mA2);
            res = _mm512_fmadd_pd(v, res, mA1);
            res = _mm512_mul_pd(v, res);
            res = _mm512_fmadd_pd(u, res, u);

            /* Load lookup table data */
            __m512d c_hi = avx512_permute_x8var_pd(mLUT_TOP_0, mLUT_TOP_1,
                            mLUT_TOP_2, mLUT_TOP_3, mLUT_TOP_4, mLUT_TOP_5,
                            mLUT_TOP_6, mLUT_TOP_7, i);
            __m512d c_lo = avx512_permute_x8var_pd(mLUT_TAIL_0, mLUT_TAIL_1,
                              mLUT_TAIL_2, mLUT_TAIL_3, mLUT_TAIL_4, mLUT_TAIL_5,
                              mLUT_TAIL_6, mLUT_TAIL_7, i);

            /*
             * log(x) = k * ln2_hi + c_hi +
             *          k * ln2_lo + c_lo +
             *          log(z/c)
             */
            __m256i ik_32 = _mm512_cvtepi64_epi32(ik);
            __m512d k = _mm512_cvtepi32_pd(ik_32);
            __m512d tt = _mm512_fmadd_pd(k, mLN2HI, c_hi);
            __m512d tt2 = _mm512_fmadd_pd(k, mLN2LO, c_lo);
            tt = _mm512_add_pd(tt, tt2);
            res = _mm512_add_pd(tt, res);

            /* return special cases */
            res = avx512_set_masked_lanes_pd(res, mNan, nan_mask);
            res = avx512_set_masked_lanes_pd(res, mNeg_nan, negx_mask);
            res = avx512_set_masked_lanes_pd(res, mNeg_inf, zero_mask);
            res = avx512_set_masked_lanes_pd(res, mInf, inf_mask);

            _mm512_mask_storeu_pd(op, load_mask, res);
        }

        /* call glibc's log func when x around 1.0f. */
        if (glibc_mask != 0) {
            double NPY_DECL_ALIGNED(64) ip_fback[8];
            /* Using a mask_store_pd instead of store_pd to prevent a fatal
             * compiler optimization bug. See
             * https://github.com/numpy/numpy/issues/27745#issuecomment-2498684564
             * for details.*/
            _mm512_mask_store_pd(ip_fback, avx512_get_full_load_mask_pd(), x_in);

            for (int ii = 0; ii < 8; ++ii, glibc_mask >>= 1) {
                if (glibc_mask & 0x01) {
                    op[ii] = npy_log(ip_fback[ii]);
                }
            }
        }
        ip += num_lanes * stride;
        op += num_lanes;
        num_remaining_elements -= num_lanes;
    }

    if (npyv_tobits_b64(invalid_mask)) {
        npy_set_floatstatus_invalid();
    }
    if (npyv_tobits_b64(divide_by_zero_mask)) {
        npy_set_floatstatus_divbyzero();
    }
}

#undef WORKAROUND_LLVM__mm512_mask_mul_pd
#undef NPY_FINLINE
#undef NPY_INFINITY
#undef NPY_NAN
#undef NPY_DECL_ALIGNED
#undef npyv_tobits_b64
#undef npy_log

}  // namespace npy_own
}  // namespace qt
#if defined(__GNUC__)
#pragma GCC diagnostic pop
#endif
#endif
