#ifndef _CELERITE_EXCEPTIONS_H_
#define _CELERITE_EXCEPTIONS_H_

#include <exception>

namespace celerite {

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

struct no_lapack : public std::exception {
  const char * what () const throw () {
    return "celerite was not compiled with LAPACK support";
  }
};

struct linalg_exception : public std::exception {
  const char * what () const throw () {
    return "failed to factorize or solve matrix";
  }
};

}; // namespace celerite

#endif // _CELERITE_EXCEPTIONS_H_
