// qtrans_python.hpp: the error messages of Python 3.13.16, and when its math functions raise.
// Written by qtrans.pyfacts from that Python, by making each error happen; do not edit.
#pragma once

#include <cmath>
#include <string>

#define QTRANS_PYTHON_VERSION "3.13.16"
#define QTRANS_PYTHON_FACTS 2

namespace qt {
namespace msg {

inline constexpr const char* int_floordiv_zero = "integer division or modulo by zero";
inline constexpr const char* int_mod_zero = "integer modulo by zero";
inline constexpr const char* int_truediv_zero = "division by zero";
inline constexpr const char* float_truediv_zero = "float division by zero";
inline constexpr const char* float_floordiv_zero = "float floor division by zero";
inline constexpr const char* float_mod_zero = "float modulo by zero";
inline constexpr const char* int_divmod_zero = "integer division or modulo by zero";
inline constexpr const char* float_divmod_zero = "float divmod()";
inline constexpr const char* min_empty = "min() iterable argument is empty";
inline constexpr const char* max_empty = "max() iterable argument is empty";
inline constexpr const char* zero_negative_power = "0.0 cannot be raised to a negative power";
inline constexpr const char* int_zero_negative_power = "0.0 cannot be raised to a negative power";
inline constexpr const char* float_pow_overflow = "(34, 'Numerical result out of range')";
inline constexpr const char* int_of_nan = "cannot convert float NaN to integer";
inline constexpr const char* int_of_inf = "cannot convert float infinity to integer";
inline constexpr const char* list_index = "list index out of range";
inline constexpr const char* str_index = "string index out of range";
inline constexpr const char* pop_empty = "pop from empty list";
inline constexpr const char* pop_index = "pop index out of range";
inline constexpr const char* math_range = "math range error";
inline constexpr const char* math_domain_pow = "math domain error";
inline constexpr const char* math_domain_fmod = "math domain error";
inline constexpr const char* fsum_overflow = "intermediate overflow in fsum";
inline constexpr const char* fsum_inf_minus_inf = "-inf + inf in fsum";
inline constexpr bool math_pow_zero_to_minus_inf_raises = false;

// list.index(x) without x: the message holds repr(x) or not
inline constexpr const char* list_index_prefix = "";
inline constexpr const char* list_index_suffix = " is not in list";
inline constexpr bool list_index_has_value = true;

// a read of a local no assignment has set (UnboundLocalError), or of a name at a program's top
// level (NameError): prefix, the name, suffix
inline constexpr const char* unbound_local_prefix = "cannot access local variable '";
inline constexpr const char* unbound_local_suffix = "' where it is not associated with a value";
inline constexpr const char* name_error_prefix = "name '";
inline constexpr const char* name_error_suffix = "' is not defined";

// math.<f>(x) outside its domain: prefix, repr(x) (when has_value), suffix; by the class of x
struct DomainText { const char* prefix; const char* suffix; bool has_value; bool as_int; };
inline constexpr DomainText sqrt_float[] = {{"math domain error", "", false, false}, {"math domain error", "", false, false}, {"math domain error", "", false, false}, {"math domain error", "", false, false}};
inline constexpr DomainText sqrt_int[] = {{"math domain error", "", false, false}, {"math domain error", "", false, false}, {"math domain error", "", false, false}, {"math domain error", "", false, false}};
inline constexpr DomainText exp_float[] = {{"math domain error", "", false, false}, {"math domain error", "", false, false}, {"math domain error", "", false, false}, {"math domain error", "", false, false}};
inline constexpr DomainText exp_int[] = {{"math domain error", "", false, false}, {"math domain error", "", false, false}, {"math domain error", "", false, false}, {"math domain error", "", false, false}};
inline constexpr DomainText log_float[] = {{"math domain error", "", false, false}, {"math domain error", "", false, false}, {"math domain error", "", false, false}, {"math domain error", "", false, false}};
inline constexpr DomainText log_int[] = {{"math domain error", "", false, false}, {"math domain error", "", false, false}, {"math domain error", "", false, false}, {"math domain error", "", false, false}};
inline constexpr DomainText log10_float[] = {{"math domain error", "", false, false}, {"math domain error", "", false, false}, {"math domain error", "", false, false}, {"math domain error", "", false, false}};
inline constexpr DomainText log10_int[] = {{"math domain error", "", false, false}, {"math domain error", "", false, false}, {"math domain error", "", false, false}, {"math domain error", "", false, false}};
inline constexpr DomainText log2_float[] = {{"math domain error", "", false, false}, {"math domain error", "", false, false}, {"math domain error", "", false, false}, {"math domain error", "", false, false}};
inline constexpr DomainText log2_int[] = {{"math domain error", "", false, false}, {"math domain error", "", false, false}, {"math domain error", "", false, false}, {"math domain error", "", false, false}};
inline constexpr DomainText log1p_float[] = {{"math domain error", "", false, false}, {"math domain error", "", false, false}, {"math domain error", "", false, false}, {"math domain error", "", false, false}};
inline constexpr DomainText log1p_int[] = {{"math domain error", "", false, false}, {"math domain error", "", false, false}, {"math domain error", "", false, false}, {"math domain error", "", false, false}};
inline constexpr DomainText expm1_float[] = {{"math domain error", "", false, false}, {"math domain error", "", false, false}, {"math domain error", "", false, false}, {"math domain error", "", false, false}};
inline constexpr DomainText expm1_int[] = {{"math domain error", "", false, false}, {"math domain error", "", false, false}, {"math domain error", "", false, false}, {"math domain error", "", false, false}};
inline constexpr DomainText sin_float[] = {{"math domain error", "", false, false}, {"math domain error", "", false, false}, {"math domain error", "", false, false}, {"math domain error", "", false, false}};
inline constexpr DomainText sin_int[] = {{"math domain error", "", false, false}, {"math domain error", "", false, false}, {"math domain error", "", false, false}, {"math domain error", "", false, false}};
inline constexpr DomainText cos_float[] = {{"math domain error", "", false, false}, {"math domain error", "", false, false}, {"math domain error", "", false, false}, {"math domain error", "", false, false}};
inline constexpr DomainText cos_int[] = {{"math domain error", "", false, false}, {"math domain error", "", false, false}, {"math domain error", "", false, false}, {"math domain error", "", false, false}};
inline constexpr DomainText tan_float[] = {{"math domain error", "", false, false}, {"math domain error", "", false, false}, {"math domain error", "", false, false}, {"math domain error", "", false, false}};
inline constexpr DomainText tan_int[] = {{"math domain error", "", false, false}, {"math domain error", "", false, false}, {"math domain error", "", false, false}, {"math domain error", "", false, false}};
inline constexpr DomainText asin_float[] = {{"math domain error", "", false, false}, {"math domain error", "", false, false}, {"math domain error", "", false, false}, {"math domain error", "", false, false}};
inline constexpr DomainText asin_int[] = {{"math domain error", "", false, false}, {"math domain error", "", false, false}, {"math domain error", "", false, false}, {"math domain error", "", false, false}};
inline constexpr DomainText acos_float[] = {{"math domain error", "", false, false}, {"math domain error", "", false, false}, {"math domain error", "", false, false}, {"math domain error", "", false, false}};
inline constexpr DomainText acos_int[] = {{"math domain error", "", false, false}, {"math domain error", "", false, false}, {"math domain error", "", false, false}, {"math domain error", "", false, false}};
inline constexpr DomainText atan_float[] = {{"math domain error", "", false, false}, {"math domain error", "", false, false}, {"math domain error", "", false, false}, {"math domain error", "", false, false}};
inline constexpr DomainText atan_int[] = {{"math domain error", "", false, false}, {"math domain error", "", false, false}, {"math domain error", "", false, false}, {"math domain error", "", false, false}};
inline constexpr DomainText sinh_float[] = {{"math domain error", "", false, false}, {"math domain error", "", false, false}, {"math domain error", "", false, false}, {"math domain error", "", false, false}};
inline constexpr DomainText sinh_int[] = {{"math domain error", "", false, false}, {"math domain error", "", false, false}, {"math domain error", "", false, false}, {"math domain error", "", false, false}};
inline constexpr DomainText cosh_float[] = {{"math domain error", "", false, false}, {"math domain error", "", false, false}, {"math domain error", "", false, false}, {"math domain error", "", false, false}};
inline constexpr DomainText cosh_int[] = {{"math domain error", "", false, false}, {"math domain error", "", false, false}, {"math domain error", "", false, false}, {"math domain error", "", false, false}};
inline constexpr DomainText tanh_float[] = {{"math domain error", "", false, false}, {"math domain error", "", false, false}, {"math domain error", "", false, false}, {"math domain error", "", false, false}};
inline constexpr DomainText tanh_int[] = {{"math domain error", "", false, false}, {"math domain error", "", false, false}, {"math domain error", "", false, false}, {"math domain error", "", false, false}};
inline constexpr DomainText asinh_float[] = {{"math domain error", "", false, false}, {"math domain error", "", false, false}, {"math domain error", "", false, false}, {"math domain error", "", false, false}};
inline constexpr DomainText asinh_int[] = {{"math domain error", "", false, false}, {"math domain error", "", false, false}, {"math domain error", "", false, false}, {"math domain error", "", false, false}};
inline constexpr DomainText acosh_float[] = {{"math domain error", "", false, false}, {"math domain error", "", false, false}, {"math domain error", "", false, false}, {"math domain error", "", false, false}};
inline constexpr DomainText acosh_int[] = {{"math domain error", "", false, false}, {"math domain error", "", false, false}, {"math domain error", "", false, false}, {"math domain error", "", false, false}};
inline constexpr DomainText atanh_float[] = {{"math domain error", "", false, false}, {"math domain error", "", false, false}, {"math domain error", "", false, false}, {"math domain error", "", false, false}};
inline constexpr DomainText atanh_int[] = {{"math domain error", "", false, false}, {"math domain error", "", false, false}, {"math domain error", "", false, false}, {"math domain error", "", false, false}};
inline constexpr DomainText erf_float[] = {{"math domain error", "", false, false}, {"math domain error", "", false, false}, {"math domain error", "", false, false}, {"math domain error", "", false, false}};
inline constexpr DomainText erf_int[] = {{"math domain error", "", false, false}, {"math domain error", "", false, false}, {"math domain error", "", false, false}, {"math domain error", "", false, false}};
inline constexpr DomainText erfc_float[] = {{"math domain error", "", false, false}, {"math domain error", "", false, false}, {"math domain error", "", false, false}, {"math domain error", "", false, false}};
inline constexpr DomainText erfc_int[] = {{"math domain error", "", false, false}, {"math domain error", "", false, false}, {"math domain error", "", false, false}, {"math domain error", "", false, false}};
inline constexpr DomainText cbrt_float[] = {{"math domain error", "", false, false}, {"math domain error", "", false, false}, {"math domain error", "", false, false}, {"math domain error", "", false, false}};
inline constexpr DomainText cbrt_int[] = {{"math domain error", "", false, false}, {"math domain error", "", false, false}, {"math domain error", "", false, false}, {"math domain error", "", false, false}};
inline constexpr DomainText exp2_float[] = {{"math domain error", "", false, false}, {"math domain error", "", false, false}, {"math domain error", "", false, false}, {"math domain error", "", false, false}};
inline constexpr DomainText exp2_int[] = {{"math domain error", "", false, false}, {"math domain error", "", false, false}, {"math domain error", "", false, false}, {"math domain error", "", false, false}};
inline constexpr DomainText fabs_float[] = {{"math domain error", "", false, false}, {"math domain error", "", false, false}, {"math domain error", "", false, false}, {"math domain error", "", false, false}};
inline constexpr DomainText fabs_int[] = {{"math domain error", "", false, false}, {"math domain error", "", false, false}, {"math domain error", "", false, false}, {"math domain error", "", false, false}};
inline constexpr DomainText gamma_float[] = {{"math domain error", "", false, false}, {"math domain error", "", false, false}, {"math domain error", "", false, false}, {"math domain error", "", false, false}};
inline constexpr DomainText gamma_int[] = {{"math domain error", "", false, false}, {"math domain error", "", false, false}, {"math domain error", "", false, false}, {"math domain error", "", false, false}};
inline constexpr DomainText lgamma_float[] = {{"math domain error", "", false, false}, {"math domain error", "", false, false}, {"math domain error", "", false, false}, {"math domain error", "", false, false}};
inline constexpr DomainText lgamma_int[] = {{"math domain error", "", false, false}, {"math domain error", "", false, false}, {"math domain error", "", false, false}, {"math domain error", "", false, false}};

// the random module: the words of its errors (randrange's changed in 3.12, randint's in 3.14)
inline constexpr const char* random_choice_empty = "Cannot choose from an empty sequence";  // IndexError
inline constexpr const char* random_randrange_empty = "empty range for randrange()";
inline constexpr const char* random_randrange_two = "empty range in randrange";
inline constexpr int random_randrange_two_form = 1;  // 0: (start, stop, width), 1: (start, stop)
inline constexpr const char* random_randrange_three = "empty range in randrange";
inline constexpr int random_randrange_three_form = 1;  // 0: no values, 1: (start, stop, step)
inline constexpr const char* random_randint = "empty range in randrange";
inline constexpr int random_randint_form = 1;  // 0, 1: randrange(a, b + 1)'s; 2: (a, b)
inline constexpr const char* random_zero_step = "zero step for randrange()";
inline constexpr const char* random_sample_size = "Sample larger than population or is negative";
inline constexpr const char* random_weights_total = "Total of weights must be greater than zero";
inline constexpr const char* random_weights_finite = "Total of weights must be finite";
inline constexpr const char* random_weights_count = "The number of weights does not match the population";
inline constexpr bool random_choice_empty_is_index_error = true;

}  // namespace msg
}  // namespace qt
