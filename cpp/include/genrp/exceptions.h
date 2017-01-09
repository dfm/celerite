#ifndef _GENRP_EXCEPTIONS_H_
#define _GENRP_EXCEPTIONS_H_

#include <exception>

namespace genrp {

struct carma_exception : public std::exception {
  const char * what () const throw () {
    return "CARMA model encountered an instability";
  }
};

struct compute_exception : public std::exception {
  const char * what () const throw () {
    return "you must call 'compute' first";
  }
};

struct dimension_mismatch : public std::exception {
  const char * what () const throw () {
    return "dimension mismatch";
  }
};

}; // namespace genrp

#endif // _GENRP_EXCEPTIONS_H_
