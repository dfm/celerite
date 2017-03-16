#ifndef _CELERITE_SOLVER_SINGLE_H_
#define _CELERITE_SOLVER_SINGLE_H_

#include <iostream>

#include <cmath>
#include <algorithm>
#include <Eigen/Core>

#include "celerite/utils.h"
#include "celerite/extended.h"
#include "celerite/exceptions.h"

#include "celerite/lapack.h"
#include "celerite/solver/solver.h"

namespace celerite {
namespace solver {

/// The class implements the original R&P solver
template <typename T>
class SingleSolver : public Solver<T> {
public:
  /// You can decide to use LAPACK for solving if it is available
  ///
  /// @param use_lapack If true, LAPACK will be used for solving the band
  //                    system.
  SingleSolver () : Solver<T>() {
#ifndef WITH_LAPACK
    throw no_lapack();
#endif
  };
  ~SingleSolver () {};

  void get_diags (
    const T& a, const T& c, const Eigen::VectorXd& x,
    Eigen::Array<T, Eigen::Dynamic, 1>& d, Eigen::Array<T, Eigen::Dynamic, 1>& dl
  );

  /// Compute the extended matrix and perform the banded LU decomposition
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
  int compute (
    const Eigen::Matrix<T, Eigen::Dynamic, 1>& alpha_real,
    const Eigen::Matrix<T, Eigen::Dynamic, 1>& beta_real,
    const Eigen::Matrix<T, Eigen::Dynamic, 1>& alpha_complex_real,
    const Eigen::Matrix<T, Eigen::Dynamic, 1>& alpha_complex_imag,
    const Eigen::Matrix<T, Eigen::Dynamic, 1>& beta_complex_real,
    const Eigen::Matrix<T, Eigen::Dynamic, 1>& beta_complex_imag,
    const Eigen::VectorXd& x,
    const Eigen::Matrix<T, Eigen::Dynamic, 1>& diag
  );

  /// Solve a linear system for the matrix defined in ``compute``
  ///
  /// A previous call to `solver.Solver.compute` defines a matrix ``A``
  /// and this method solves for ``x`` in the matrix equation ``A.x = b``.
  ///
  /// @param b The right hand side of the linear system.
  /// @param x A pointer that will be overwritten with the result.
  void solve (const Eigen::MatrixXd& b, T* x) const;

  /// Compute the dot product of a ``celerite`` matrix and another arbitrary matrix
  ///
  /// This method computes ``A.b`` where ``A`` is defined by the parameters and
  /// ``b`` is an arbitrary matrix of the correct shape.
  ///
  /// @param alpha_real The coefficients of the real terms.
  /// @param beta_real The exponents of the real terms.
  /// @param alpha_complex_real The real part of the coefficients of the complex terms.
  /// @param alpha_complex_imag The imaginary part of the of the complex terms.
  /// @param beta_complex_real The real part of the exponents of the complex terms.
  /// @param beta_complex_imag The imaginary part of the exponents of the complex terms.
  /// @param x The _sorted_ array of input coordinates.
  /// @param b_in The matrix ``b`` described above.
  ///
  /// @return The matrix ``A.b``.
  Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> dot (
    const Eigen::Matrix<T, Eigen::Dynamic, 1>& alpha_real,
    const Eigen::Matrix<T, Eigen::Dynamic, 1>& beta_real,
    const Eigen::Matrix<T, Eigen::Dynamic, 1>& alpha_complex_real,
    const Eigen::Matrix<T, Eigen::Dynamic, 1>& alpha_complex_imag,
    const Eigen::Matrix<T, Eigen::Dynamic, 1>& beta_complex_real,
    const Eigen::Matrix<T, Eigen::Dynamic, 1>& beta_complex_imag,
    const Eigen::VectorXd& x,
    const Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>& b_in
  );

  // Needed for the Eigen-free interface.
  using Solver<T>::compute;
  using Solver<T>::solve;

protected:
  Eigen::Array<T, Eigen::Dynamic, 1> dl_, d_, dl_noise_, d_noise_, du_noise_;

};

template <typename T>
int sgn(T val) {
  return (T(0) < val) - (val < T(0));
}

template <typename T>
T log_minus_exp(int sgn_a, const T& a, int sgn_b, const T& b, int* sgn_fn) {
  using std::max;
  using std::abs;
  T base = max(a, b),
    arg = sgn_a * exp(a - base) - sgn_b * exp(b - base);
  *sgn_fn = sgn(arg);
  return base + log(abs(arg));
}

template <typename Derived>
typename Derived::Scalar tri_slogdet (const Eigen::DenseBase<Derived>& dl,
                                      const Eigen::DenseBase<Derived>& d,
                                      const Eigen::DenseBase<Derived>& du,
                                      int* sfn)
{
  typedef typename Derived::Scalar T;
  int n = d.rows(), sfm1 = 1, sfm2 = 1;
  T fm1 = log(d(0)),
    fm2 = T(0.0),
    fn;

  for (int i = 1; i < n - 1; ++i) {
    fn = log_minus_exp(sfm1, log(d(i)) + fm1, sfm2, log(dl(i-1)*du(i-1)) + fm2, sfn);
    fm2 = fm1;
    sfm2 = sfm1;
    fm1 = fn;
    sfm1 = *sfn;
  }

  return log_minus_exp(sfm1, log(d(n-1)) + fm1, sfm2, log(dl(n-2)*du(n-2)) + fm2, sfn);
}

template <typename T>
void SingleSolver<T>::get_diags (
  const T& a, const T& c, const Eigen::VectorXd& x,
  Eigen::Array<T, Eigen::Dynamic, 1>& d, Eigen::Array<T, Eigen::Dynamic, 1>& dl
)
{
  int n = x.rows();
  T r = exp(-c * (x(1) - x(0))),
    e = 1.0 / (1.0 / r - r);
  dl.resize(n - 1, 1);
  d.resize(n, 1);
  d(0) = (1.0 + r * e) / a;
  for (int i = 1; i < n; ++i) {
    r = exp(-c * (x(i) - x(i-1)));
    e = 1.0 / (1.0 / r - r);
    dl(i - 1) = -e / a;
    d(i) = (1.0 + r * e) / a;
    if (i > 1) d(i-1) += r * e / a;
  }
}

template <typename T>
int SingleSolver<T>::compute (
  const Eigen::Matrix<T, Eigen::Dynamic, 1>& alpha_real,
  const Eigen::Matrix<T, Eigen::Dynamic, 1>& beta_real,
  const Eigen::Matrix<T, Eigen::Dynamic, 1>& alpha_complex_real,
  const Eigen::Matrix<T, Eigen::Dynamic, 1>& alpha_complex_imag,
  const Eigen::Matrix<T, Eigen::Dynamic, 1>& beta_complex_real,
  const Eigen::Matrix<T, Eigen::Dynamic, 1>& beta_complex_imag,
  const Eigen::VectorXd& x,
  const Eigen::Matrix<T, Eigen::Dynamic, 1>& diag
)
{
  using std::abs;

  this->computed_ = false;
  if (x.rows() != diag.rows()) return SOLVER_DIMENSION_MISMATCH;
  if (alpha_real.rows() != 1) return SOLVER_DIMENSION_MISMATCH;
  if (beta_real.rows() != 1) return SOLVER_DIMENSION_MISMATCH;
  if (alpha_complex_real.rows() != 0) return SOLVER_DIMENSION_MISMATCH;
  if (alpha_complex_imag.rows() != 0) return SOLVER_DIMENSION_MISMATCH;
  if (beta_complex_real.rows() != 0) return SOLVER_DIMENSION_MISMATCH;
  if (beta_complex_imag.rows() != 0) return SOLVER_DIMENSION_MISMATCH;

  T a = alpha_real(0), c = beta_real(0);
  int n = this->n_ = x.rows();

  // Build the diagonals
  get_diags(a, c, x, d_, dl_);

  d_noise_ = d_ * diag.array() + T(1.0);
  du_noise_ = dl_ * diag.head(n - 1).array();
  dl_noise_ = dl_ * diag.tail(n - 1).array();

  // Compute the determinant
  int s = 0, flag = 0;
  T logdet_Kinv = tri_slogdet (dl_, d_, dl_, &s);
  if (s != 1) flag += -1;
  T logdet_noise = tri_slogdet (dl_noise_, d_noise_, du_noise_, &s);
  if (s != 1) flag += -2;

  this->log_det_ = logdet_noise - logdet_Kinv;
  this->computed_ = true;

  return flag;
}

template <typename T>
void SingleSolver<T>::solve (const Eigen::MatrixXd& b, T* x) const {
#ifdef WITH_LAPACK
  if (b.rows() != this->n_) throw dimension_mismatch();
  if (!(this->computed_)) throw compute_exception();

  int info, n = this->n_,
      nrhs = b.cols();

  // Copy the data to x
  for (int i = 0; i < nrhs; ++i)
    for (int j = 0; j < n; ++j)
      x[i*n + j] = T(b(j, i));

  // Copy the diagonals so that they don't get overwritten
  Eigen::Array<T, Eigen::Dynamic, 1> dl = dl_noise_, d = d_noise_, du = du_noise_;

  // Do the solve of (1 + K^{-1}.N)^{-1}
  dgtsv_(&n, &nrhs, dl.data(), d.data(), du.data(), x, &n, &info);

  // Do the K^{-1} dot product
  T tmp, swp;
  for (int i = 0; i < nrhs; ++i) {
    swp = x[i*n];
    x[i*n] = x[i*n] * d_(0) + x[i*n + 1] * dl_(0);
    for (int j = 1; j < n - 1; ++j) {
      tmp = x[i*n + j];
      x[i*n+j] = swp*dl_(j-1) + x[i*n+j]*d_(j) + x[i*n+j+1]*dl_(j);
      swp = tmp;
    }
    x[i*n+n-1] = swp*dl_(n-2) + x[i*n+n-1]*d_(n-1);
  }
#else
  throw no_lapack();
#endif
}

template <typename T>
Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> SingleSolver<T>::dot (
  const Eigen::Matrix<T, Eigen::Dynamic, 1>& alpha_real,
  const Eigen::Matrix<T, Eigen::Dynamic, 1>& beta_real,
  const Eigen::Matrix<T, Eigen::Dynamic, 1>& alpha_complex_real,
  const Eigen::Matrix<T, Eigen::Dynamic, 1>& alpha_complex_imag,
  const Eigen::Matrix<T, Eigen::Dynamic, 1>& beta_complex_real,
  const Eigen::Matrix<T, Eigen::Dynamic, 1>& beta_complex_imag,
  const Eigen::VectorXd& t,
  const Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>& b_in
) {
  if (t.rows() != b_in.rows()) throw dimension_mismatch();

  int n = t.rows(), nrhs = b_in.cols(), info = 0;
  Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> b_out = b_in;

  Eigen::Array<T, Eigen::Dynamic, 1> d, dl, du;
  T a = alpha_real(0), c = beta_real(0);
  get_diags(a, c, t, d, dl);
  du = dl;

#ifdef WITH_LAPACK
  // Do the dot product using a tridiagonal solve
  dgtsv_(&n, &nrhs, dl.data(), d.data(), du.data(), b_out.data(), &n, &info);
#else
  throw no_lapack();
#endif

  return b_out;
}

};
};

#endif
