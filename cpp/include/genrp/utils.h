#ifndef _GENRP_UTILS_
#define _GENRP_UTILS_

#include <complex>

// Helper for getting real values generally - a no-op for doubles.
inline double get_real (double value) { return value; }
inline double get_real (std::complex<double> value) { return value.real(); }

inline double get_conj (double value) { return value; }
inline std::complex<double> get_conj (std::complex<double> value) { return std::conj(value); }

#endif
