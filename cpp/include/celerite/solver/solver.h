#ifndef _CELERITE_SOLVER_SOLVER_H_
#define _CELERITE_SOLVER_SOLVER_H_

#include <Eigen/Core>
#include "celerite/exceptions.h"

namespace celerite {
namespace solver {

template <typename T>
class Solver {
protected:

bool computed_;
int N_;
T log_det_;

public:

Solver () : computed_(false) {};
virtual ~Solver () {};

/// Compute the matrix and factorize
///
/// This method is overloaded by subclasses to provide the specific implementation.
///
/// @param a_real The coefficients of the real terms.
/// @param c_real The exponents of the real terms.
/// @param a_comp The real part of the coefficients of the complex terms.
/// @param b_comp The imaginary part of the of the complex terms.
/// @param c_comp The real part of the exponents of the complex terms.
/// @param d_comp The imaginary part of the exponents of the complex terms.
/// @param x The _sorted_ array of input coordinates.
/// @param diag An array that should be added to the diagonal of the matrix.
///             This often corresponds to measurement uncertainties and in that case,
///             ``diag`` should be the measurement _variance_ (i.e. sigma^2).
///
/// @return ``0`` on success. ``1`` for mismatched dimensions.
virtual void compute (
  const Eigen::Matrix<T, Eigen::Dynamic, 1>& a_real,
  const Eigen::Matrix<T, Eigen::Dynamic, 1>& c_real,
  const Eigen::Matrix<T, Eigen::Dynamic, 1>& a_comp,
  const Eigen::Matrix<T, Eigen::Dynamic, 1>& b_comp,
  const Eigen::Matrix<T, Eigen::Dynamic, 1>& c_comp,
  const Eigen::Matrix<T, Eigen::Dynamic, 1>& d_comp,
  const Eigen::VectorXd& x,
  const Eigen::Matrix<T, Eigen::Dynamic, 1>& diag
) = 0;
virtual Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> solve (const Eigen::MatrixXd& b) const = 0;
virtual Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> dot_L (const Eigen::MatrixXd& z) const = 0;


/// Solve the system ``b^T . A^-1 . b``
///
/// A previous call to `solver::Solver::compute` defines a matrix ``A``
/// and this method solves ``b^T . A^-1 . b`` for a vector ``b``.
///
/// @param b The right hand side of the linear system.
T dot_solve (const Eigen::VectorXd& b) const {
  if (!(this->computed_)) throw compute_exception();
  Eigen::Matrix<T, Eigen::Dynamic, 1> out = solve(b);
  return b.transpose().cast<T>() * out;
};


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
void compute (
  const Eigen::Matrix<T, Eigen::Dynamic, 1>& a_real,
  const Eigen::Matrix<T, Eigen::Dynamic, 1>& c_real,
  const Eigen::Matrix<T, Eigen::Dynamic, 1>& a_comp,
  const Eigen::Matrix<T, Eigen::Dynamic, 1>& c_comp,
  const Eigen::Matrix<T, Eigen::Dynamic, 1>& d_comp,
  const Eigen::VectorXd& x,
  const Eigen::Matrix<T, Eigen::Dynamic, 1>& diag
) {
  Eigen::Matrix<T, Eigen::Dynamic, 1> b_comp(a_comp.rows());
  b_comp.setZero();
  return this->compute(a_real, c_real, a_comp, b_comp, c_comp, d_comp, x, diag);
};


/// Compute the matrix and factorize for a set of purely real terms
///
/// @param a_real The coefficients of the real terms.
/// @param c_real The exponents of the real terms.
/// @param x The _sorted_ array of input coordinates.
/// @param diag An array that should be added to the diagonal of the matrix.
///             This often corresponds to measurement uncertainties and in that case,
///             ``diag`` should be the measurement _variance_ (i.e. sigma^2).
///
/// @return ``0`` on success. ``1`` for mismatched dimensions.
void compute (
  const Eigen::Matrix<T, Eigen::Dynamic, 1>& a_real,
  const Eigen::Matrix<T, Eigen::Dynamic, 1>& c_real,
  const Eigen::VectorXd& x,
  const Eigen::Matrix<T, Eigen::Dynamic, 1>& diag
) {
  Eigen::Matrix<T, Eigen::Dynamic, 1> nothing;
  return this->compute(a_real, c_real, nothing, nothing, nothing, nothing, x, diag);
};

/// Compute the matrix and factorize for purely complex terms with real alphas
///
/// @param a_comp The real part of the coefficients of the complex terms.
/// @param c_comp The real part of the exponents of the complex terms.
/// @param d_comp The imaginary part of the exponents of the complex terms.
/// @param x The _sorted_ array of input coordinates.
/// @param diag An array that should be added to the diagonal of the matrix.
///             This often corresponds to measurement uncertainties and in that case,
///             ``diag`` should be the measurement _variance_ (i.e. sigma^2).
///
/// @return ``0`` on success. ``1`` for mismatched dimensions.
void compute (
  const Eigen::Matrix<T, Eigen::Dynamic, 1>& a_comp,
  const Eigen::Matrix<T, Eigen::Dynamic, 1>& c_comp,
  const Eigen::Matrix<T, Eigen::Dynamic, 1>& d_comp,
  const Eigen::VectorXd& x,
  const Eigen::Matrix<T, Eigen::Dynamic, 1>& diag
) {
  Eigen::Matrix<T, Eigen::Dynamic, 1> nothing;
  Eigen::Matrix<T, Eigen::Dynamic, 1> b_comp(a_comp.rows());
  b_comp.setZero();
  return this->compute(nothing, nothing, a_comp, b_comp, c_comp, d_comp, x, diag);
};


/// Compute the matrix and factorize for purely complex terms
///
/// @param a_comp The real part of the coefficients of the complex terms.
/// @param b_comp The imaginary part of the coefficients of the complex terms.
/// @param c_comp The real part of the exponents of the complex terms.
/// @param d_comp The imaginary part of the exponents of the complex terms.
/// @param x The _sorted_ array of input coordinates.
/// @param diag An array that should be added to the diagonal of the matrix.
///             This often corresponds to measurement uncertainties and in that case,
///             ``diag`` should be the measurement _variance_ (i.e. sigma^2).
///
/// @return ``0`` on success. ``1`` for mismatched dimensions.
void compute (
  const Eigen::Matrix<T, Eigen::Dynamic, 1>& a_comp,
  const Eigen::Matrix<T, Eigen::Dynamic, 1>& b_comp,
  const Eigen::Matrix<T, Eigen::Dynamic, 1>& c_comp,
  const Eigen::Matrix<T, Eigen::Dynamic, 1>& d_comp,
  const Eigen::VectorXd& x,
  const Eigen::Matrix<T, Eigen::Dynamic, 1>& diag
) {
  Eigen::Matrix<T, Eigen::Dynamic, 1> nothing;
  return this->compute(nothing, nothing, a_comp, b_comp, c_comp, d_comp, x, diag);
};

}; // class Solver
}; // namespace solver
}; // namespace celerite

#endif
