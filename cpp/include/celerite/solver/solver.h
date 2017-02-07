#ifndef _CELERITE_SOLVER_SOLVER_H_
#define _CELERITE_SOLVER_SOLVER_H_

#include <Eigen/Core>

#include "celerite/exceptions.h"

namespace celerite {
namespace solver {

int SOLVER_DIMENSION_MISMATCH = 1;
int SOLVER_NOT_COMPUTED = 2;

template <typename T>
class Solver {
protected:
  bool computed_;
  int n_, p_real_, p_complex_;
  T log_det_;

public:
  Solver () : computed_(false) {};
  virtual ~Solver () {};

  /// Compute the matrix and factorize
  ///
  /// This method is overloaded by subclasses to provide the specific implementation.
  ///
  /// @param alpha_real The coefficients of the real terms.
  /// @param beta_real The exponents of the real terms.
  /// @param alpha_complex_real The real part of the coefficients of the complex terms.
  /// @param alpha_complex_imag The imaginary part of the of the complex terms.
  /// @param beta_complex_real The real part of the exponents of the complex terms.
  /// @param beta_complex_imag The imaginary part of the exponents of the complex terms.
  /// @param x The _sorted_ array of input coordinates.
  /// @param diag An array that should be added to the diagonal of the matrix.
  ///             This often corresponds to measurement uncertainties and in that case,
  ///             ``diag`` should be the measurement _variance_ (i.e. sigma^2).
  ///
  /// @return ``0`` on success. ``1`` for mismatched dimensions.
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

  /// Solve a linear system for the matrix defined in ``compute``
  ///
  /// A previous call to `solver::Solver::compute` defines a matrix ``A``
  /// and this method solves for ``x`` in the matrix equation ``A.x = b``.
  ///
  /// @param b The right hand side of the linear system.
  /// @param x A pointer that will be overwritten with the result.
  Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> solve (const Eigen::MatrixXd& b) const;

  /// Solve the system ``b^T . A^-1 . b``
  ///
  /// A previous call to `solver::Solver::compute` defines a matrix ``A``
  /// and this method solves ``b^T . A^-1 . b`` for a vector ``b``.
  ///
  /// @param b The right hand side of the linear system.
  T dot_solve (const Eigen::VectorXd& b) const;

  /// Get the log determinant of the matrix
  T log_determinant () const {
    if (!(this->computed_)) throw compute_exception();
    return log_det_;
  };

  /// Flag indicating if ``compute`` was successfully executed
  bool computed () const { return computed_; };

  /// Compute the matrix and factorize for purely real alphas
  ///
  /// @param alpha_real The coefficients of the real terms.
  /// @param beta_real The exponents of the real terms.
  /// @param alpha_complex_real The real part of the coefficients of the complex terms.
  /// @param beta_complex_real The real part of the exponents of the complex terms.
  /// @param beta_complex_imag The imaginary part of the exponents of the complex terms.
  /// @param x The _sorted_ array of input coordinates.
  /// @param diag An array that should be added to the diagonal of the matrix.
  ///             This often corresponds to measurement uncertainties and in that case,
  ///             ``diag`` should be the measurement _variance_ (i.e. sigma^2).
  ///
  /// @return ``0`` on success. ``1`` for mismatched dimensions.
  int compute (
    const Eigen::Matrix<T, Eigen::Dynamic, 1>& alpha_real,
    const Eigen::Matrix<T, Eigen::Dynamic, 1>& beta_real,
    const Eigen::Matrix<T, Eigen::Dynamic, 1>& alpha_complex_real,
    const Eigen::Matrix<T, Eigen::Dynamic, 1>& beta_complex_real,
    const Eigen::Matrix<T, Eigen::Dynamic, 1>& beta_complex_imag,
    const Eigen::VectorXd& x,
    const Eigen::Matrix<T, Eigen::Dynamic, 1>& diag
  );

  /// Compute the matrix and factorize for a set of purely real terms
  ///
  /// @param alpha_real The coefficients of the real terms.
  /// @param beta_real The exponents of the real terms.
  /// @param x The _sorted_ array of input coordinates.
  /// @param diag An array that should be added to the diagonal of the matrix.
  ///             This often corresponds to measurement uncertainties and in that case,
  ///             ``diag`` should be the measurement _variance_ (i.e. sigma^2).
  ///
  /// @return ``0`` on success. ``1`` for mismatched dimensions.
  int compute (
    const Eigen::Matrix<T, Eigen::Dynamic, 1>& alpha_real,
    const Eigen::Matrix<T, Eigen::Dynamic, 1>& beta_real,
    const Eigen::VectorXd& x,
    const Eigen::Matrix<T, Eigen::Dynamic, 1>& diag
  );

  /// Compute the matrix and factorize for purely complex terms with real alphas
  ///
  /// @param alpha_complex_real The real part of the coefficients of the complex terms.
  /// @param beta_complex_real The real part of the exponents of the complex terms.
  /// @param beta_complex_imag The imaginary part of the exponents of the complex terms.
  /// @param x The _sorted_ array of input coordinates.
  /// @param diag An array that should be added to the diagonal of the matrix.
  ///             This often corresponds to measurement uncertainties and in that case,
  ///             ``diag`` should be the measurement _variance_ (i.e. sigma^2).
  ///
  /// @return ``0`` on success. ``1`` for mismatched dimensions.
  int compute (
    const Eigen::Matrix<T, Eigen::Dynamic, 1>& alpha_complex_real,
    const Eigen::Matrix<T, Eigen::Dynamic, 1>& beta_complex_real,
    const Eigen::Matrix<T, Eigen::Dynamic, 1>& beta_complex_imag,
    const Eigen::VectorXd& x,
    const Eigen::Matrix<T, Eigen::Dynamic, 1>& diag
  );

  /// Compute the matrix and factorize for purely complex terms
  ///
  /// @param alpha_complex_real The real part of the coefficients of the complex terms.
  /// @param alpha_complex_imag The imaginary part of the coefficients of the complex terms.
  /// @param beta_complex_real The real part of the exponents of the complex terms.
  /// @param beta_complex_imag The imaginary part of the exponents of the complex terms.
  /// @param x The _sorted_ array of input coordinates.
  /// @param diag An array that should be added to the diagonal of the matrix.
  ///             This often corresponds to measurement uncertainties and in that case,
  ///             ``diag`` should be the measurement _variance_ (i.e. sigma^2).
  ///
  /// @return ``0`` on success. ``1`` for mismatched dimensions.
  int compute (
    const Eigen::Matrix<T, Eigen::Dynamic, 1>& alpha_complex_real,
    const Eigen::Matrix<T, Eigen::Dynamic, 1>& alpha_complex_imag,
    const Eigen::Matrix<T, Eigen::Dynamic, 1>& beta_complex_real,
    const Eigen::Matrix<T, Eigen::Dynamic, 1>& beta_complex_imag,
    const Eigen::VectorXd& x,
    const Eigen::Matrix<T, Eigen::Dynamic, 1>& diag
  );

};

template <typename T>
T Solver<T>::dot_solve (const Eigen::VectorXd& b) const {
  if (!(this->computed_)) throw compute_exception();
  Eigen::Matrix<T, Eigen::Dynamic, 1> out(n_);
  solve(b, out.data());
  return b.transpose().cast<T>() * out;
}

template <typename T>
Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> Solver<T>::solve (const Eigen::MatrixXd& b) const {
  Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> x(b.rows(), b.cols());
  solve(b, x.data());
  return x;
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

};
};

#endif
