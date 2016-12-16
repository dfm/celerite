#ifndef _GENRP_SOLVER_SOLVER_H_
#define _GENRP_SOLVER_SOLVER_H_

#include <Eigen/Core>
#include "genrp/poly.h"

namespace genrp {

int SOLVER_DIMENSION_MISMATCH = 1;
int SOLVER_NOT_COMPUTED = 2;

template <typename T>
class Solver {
protected:
  bool computed_;
  size_t n_, p_real_, p_complex_;
  T log_det_;

public:
  Solver () : computed_(false) {};
  virtual ~Solver () {};

  virtual int compute (
    const Eigen::Matrix<T, Eigen::Dynamic, 1>& alpha_real,
    const Eigen::Matrix<T, Eigen::Dynamic, 1>& beta_real,
    const Eigen::Matrix<T, Eigen::Dynamic, 1>& alpha_complex_real,
    const Eigen::Matrix<T, Eigen::Dynamic, 1>& alpha_complex_imag,
    const Eigen::Matrix<T, Eigen::Dynamic, 1>& beta_complex_real,
    const Eigen::Matrix<T, Eigen::Dynamic, 1>& beta_complex_imag,
    const Eigen::VectorXd& x,
    const Eigen::Matrix<T, Eigen::Dynamic, 1>& diag
  ) = 0;
  virtual void solve (const Eigen::MatrixXd& b, T* x) const = 0;

  T dot_solve (const Eigen::VectorXd& b) const;
  T log_determinant () const;

  // Helpers
  int compute (
    const Eigen::Matrix<T, Eigen::Dynamic, 1>& alpha_real,
    const Eigen::Matrix<T, Eigen::Dynamic, 1>& beta_real,
    const Eigen::Matrix<T, Eigen::Dynamic, 1>& alpha_complex_real,
    const Eigen::Matrix<T, Eigen::Dynamic, 1>& beta_complex_real,
    const Eigen::Matrix<T, Eigen::Dynamic, 1>& beta_complex_imag,
    const Eigen::VectorXd& x,
    const Eigen::Matrix<T, Eigen::Dynamic, 1>& diag
  );
  int compute (
    const Eigen::Matrix<T, Eigen::Dynamic, 1>& alpha_real,
    const Eigen::Matrix<T, Eigen::Dynamic, 1>& beta_real,
    const Eigen::VectorXd& x,
    const Eigen::Matrix<T, Eigen::Dynamic, 1>& diag
  );
  int compute (
    const Eigen::Matrix<T, Eigen::Dynamic, 1>& alpha_complex_real,
    const Eigen::Matrix<T, Eigen::Dynamic, 1>& beta_complex_real,
    const Eigen::Matrix<T, Eigen::Dynamic, 1>& beta_complex_imag,
    const Eigen::VectorXd& x,
    const Eigen::Matrix<T, Eigen::Dynamic, 1>& diag
  );
  int compute (
    const Eigen::Matrix<T, Eigen::Dynamic, 1>& alpha_complex_real,
    const Eigen::Matrix<T, Eigen::Dynamic, 1>& alpha_complex_imag,
    const Eigen::Matrix<T, Eigen::Dynamic, 1>& beta_complex_real,
    const Eigen::Matrix<T, Eigen::Dynamic, 1>& beta_complex_imag,
    const Eigen::VectorXd& x,
    const Eigen::Matrix<T, Eigen::Dynamic, 1>& diag
  );

  // Eigen-free interface.
  int compute (size_t p_real,
                const T* const alpha_real, const T* const beta_real,
                size_t p_complex,
                const T* const alpha_complex_real,
                const T* const beta_complex_real, const T* const beta_complex_imag,
                size_t n, const double* t, const T* const diag);
  int compute (size_t p_real,
                const T* const alpha_real, const T* const beta_real,
                size_t p_complex,
                const T* const alpha_complex_real, const T* const alpha_complex_imag,
                const T* const beta_complex_real, const T* const beta_complex_imag,
                size_t n, const double* t, const T* const diag);
  void solve (const double* const b, T* x) const;
  void solve (size_t nrhs, const double* const b, T* x) const;
  T dot_solve (const double* const b) const;

};

template <typename T>
T Solver<T>::dot_solve (const Eigen::VectorXd& b) const {
  Eigen::Matrix<T, Eigen::Dynamic, 1> out(n_);
  solve(b, out.data());
  return b.transpose().cast<T>() * out;
}

template <typename T>
T Solver<T>::log_determinant () const {
  return log_det_;
}

// Helpers
template <typename T>
int Solver<T>::compute (
  const Eigen::Matrix<T, Eigen::Dynamic, 1>& alpha_real,
  const Eigen::Matrix<T, Eigen::Dynamic, 1>& beta_real,
  const Eigen::Matrix<T, Eigen::Dynamic, 1>& alpha_complex_real,
  const Eigen::Matrix<T, Eigen::Dynamic, 1>& beta_complex_real,
  const Eigen::Matrix<T, Eigen::Dynamic, 1>& beta_complex_imag,
  const Eigen::VectorXd& x,
  const Eigen::Matrix<T, Eigen::Dynamic, 1>& diag
)
{
  Eigen::Matrix<T, Eigen::Dynamic, 1> alpha_complex_imag = Eigen::Matrix<T, Eigen::Dynamic, 1>::Zero(alpha_complex_real.rows());
  return this->compute(alpha_real, beta_real, alpha_complex_real, alpha_complex_imag, beta_complex_real, beta_complex_imag, x, diag);
}

template <typename T>
int Solver<T>::compute (
  const Eigen::Matrix<T, Eigen::Dynamic, 1>& alpha_real,
  const Eigen::Matrix<T, Eigen::Dynamic, 1>& beta_real,
  const Eigen::VectorXd& x,
  const Eigen::Matrix<T, Eigen::Dynamic, 1>& diag
)
{
  Eigen::Matrix<T, Eigen::Dynamic, 1> nothing;
  return this->compute(alpha_real, beta_real, nothing, nothing, nothing, nothing, x, diag);
}

template <typename T>
int Solver<T>::compute (
  const Eigen::Matrix<T, Eigen::Dynamic, 1>& alpha_complex_real,
  const Eigen::Matrix<T, Eigen::Dynamic, 1>& beta_complex_real,
  const Eigen::Matrix<T, Eigen::Dynamic, 1>& beta_complex_imag,
  const Eigen::VectorXd& x,
  const Eigen::Matrix<T, Eigen::Dynamic, 1>& diag
)
{
  Eigen::Matrix<T, Eigen::Dynamic, 1> nothing;
  Eigen::Matrix<T, Eigen::Dynamic, 1> alpha_complex_imag = Eigen::Matrix<T, Eigen::Dynamic, 1>::Zero(alpha_complex_real.rows());
  return this->compute(nothing, nothing, alpha_complex_real, alpha_complex_imag, beta_complex_real, beta_complex_imag, x, diag);
}

template <typename T>
int Solver<T>::compute (
  const Eigen::Matrix<T, Eigen::Dynamic, 1>& alpha_complex_real,
  const Eigen::Matrix<T, Eigen::Dynamic, 1>& alpha_complex_imag,
  const Eigen::Matrix<T, Eigen::Dynamic, 1>& beta_complex_real,
  const Eigen::Matrix<T, Eigen::Dynamic, 1>& beta_complex_imag,
  const Eigen::VectorXd& x,
  const Eigen::Matrix<T, Eigen::Dynamic, 1>& diag
)
{
  Eigen::Matrix<T, Eigen::Dynamic, 1> nothing;
  return this->compute(nothing, nothing, alpha_complex_real, alpha_complex_imag, beta_complex_real, beta_complex_imag, x, diag);
}

// Eigen-free interface:
template <typename T>
int Solver<T>::compute (
    size_t p_real, const T* const alpha_real, const T* const beta_real,
    size_t p_complex, const T* const alpha_complex_real, const T* const beta_complex_real, const T* const beta_complex_imag,
    size_t n, const double* t, const T* const diag
)
{
  typedef Eigen::Map<const Eigen::Matrix<T, Eigen::Dynamic, 1> > dbl_vector_t;
  typedef Eigen::Map<const Eigen::Matrix<T, Eigen::Dynamic, 1> > vector_t;
  return compute(
    vector_t(alpha_real, p_real, 1),
    vector_t(beta_real, p_real, 1),
    vector_t(alpha_complex_real, p_complex, 1),
    vector_t(beta_complex_real, p_complex, 1),
    vector_t(beta_complex_imag, p_complex, 1),
    dbl_vector_t(t, n, 1), vector_t(diag, n, 1)
  );
}

template <typename T>
int Solver<T>::compute (
    size_t p_real, const T* const alpha_real, const T* const beta_real,
    size_t p_complex, const T* const alpha_complex_real, const T* const alpha_complex_imag, const T* const beta_complex_real, const T* const beta_complex_imag,
    size_t n, const double* t, const T* const diag
)
{
  typedef Eigen::Map<const Eigen::Matrix<T, Eigen::Dynamic, 1> > dbl_vector_t;
  typedef Eigen::Map<const Eigen::Matrix<T, Eigen::Dynamic, 1> > vector_t;
  return compute(
    vector_t(alpha_real, p_real, 1),
    vector_t(beta_real, p_real, 1),
    vector_t(alpha_complex_real, p_complex, 1),
    vector_t(alpha_complex_imag, p_complex, 1),
    vector_t(beta_complex_real, p_complex, 1),
    vector_t(beta_complex_imag, p_complex, 1),
    dbl_vector_t(t, n, 1), vector_t(diag, n, 1)
  );
}

template <typename T>
void Solver<T>::solve (const double* b, T* x) const {
  Eigen::Map<const Eigen::MatrixXd> bin(b, n_, 1);
  solve(bin, x);
}

template <typename T>
void Solver<T>::solve (size_t nrhs, const double* b, T* x) const {
  Eigen::Map<const Eigen::MatrixXd> bin(b, n_, nrhs);
  solve(bin, x);
}

template <typename T>
T Solver<T>::dot_solve (const double* b) const {
  Eigen::Map<const Eigen::VectorXd> bin(b, n_);
  return dot_solve(bin);
}

};

#endif
