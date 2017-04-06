#ifndef _CELERITE_SOLVER_CHOLESKY_H_
#define _CELERITE_SOLVER_CHOLESKY_H_

#include <cmath>
#include <complex>
#include <Eigen/Core>

#include "celerite/utils.h"
#include "celerite/exceptions.h"

#include "celerite/solver/solver.h"

namespace celerite {
namespace solver {

template <typename T>
class CholeskySolver : public Solver<T> {
public:
  CholeskySolver () : Solver<T>() {};
  ~CholeskySolver () {};
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

  void solve (const Eigen::MatrixXd& b, T* x) const;

  //Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> dot (
  //  const Eigen::Matrix<T, Eigen::Dynamic, 1>& alpha_real,
  //  const Eigen::Matrix<T, Eigen::Dynamic, 1>& beta_real,
  //  const Eigen::Matrix<T, Eigen::Dynamic, 1>& alpha_complex_real,
  //  const Eigen::Matrix<T, Eigen::Dynamic, 1>& alpha_complex_imag,
  //  const Eigen::Matrix<T, Eigen::Dynamic, 1>& beta_complex_real,
  //  const Eigen::Matrix<T, Eigen::Dynamic, 1>& beta_complex_imag,
  //  const Eigen::VectorXd& x,
  //  const Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>& b_in
  //);

  using Solver<T>::compute;
  using Solver<T>::solve;

protected:
  int j_;
  Eigen::Matrix<std::complex<T>, Eigen::Dynamic, Eigen::Dynamic> phi_, X_;
  Eigen::Array<std::complex<T>, Eigen::Dynamic, 1> alpha_;
  Eigen::Array<T, Eigen::Dynamic, 1> D_;

};

template <typename T>
int CholeskySolver<T>::compute (
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
  this->computed_ = false;
  if (x.rows() != diag.rows()) return SOLVER_DIMENSION_MISMATCH;
  if (alpha_real.rows() != beta_real.rows()) return SOLVER_DIMENSION_MISMATCH;
  if (alpha_complex_real.rows() != alpha_complex_imag.rows()) return SOLVER_DIMENSION_MISMATCH;
  if (alpha_complex_real.rows() != beta_complex_real.rows()) return SOLVER_DIMENSION_MISMATCH;
  if (alpha_complex_real.rows() != beta_complex_imag.rows()) return SOLVER_DIMENSION_MISMATCH;

  int N = x.rows(), J = alpha_real.rows() + 2 * alpha_complex_real.rows();
  phi_.resize(J, N-1);
  D_.resize(N);
  X_.resize(J, N);
  alpha_.resize(J);
  Eigen::Array<std::complex<T>, Eigen::Dynamic, Eigen::Dynamic> S(J, J);
  Eigen::Array<std::complex<T>, Eigen::Dynamic, 1> beta(J), tmp(J);

  int i = 0;
  for (int j = 0; j < alpha_real.rows(); ++j) {
    alpha_(i) = alpha_real(j);
    beta(i++) = beta_real(j);
  }
  for (int j = 0; j < alpha_complex_real.rows(); ++j) {
    alpha_(i) = 0.5*std::complex<T>(alpha_complex_real(j), alpha_complex_imag(j));
    beta(i++) = std::complex<T>(beta_complex_real(j), beta_complex_imag(j));
    alpha_(i) = 0.5*std::complex<T>(alpha_complex_real(j), -alpha_complex_imag(j));
    beta(i++) = std::complex<T>(beta_complex_real(j), -beta_complex_imag(j));
  }

  // First pass
  T alpha_sum = alpha_.sum().real();
  D_(0) = diag(0) + alpha_sum;
  S.setConstant(1.0 / D_(0));
  X_.col(0).setConstant(1.0 / D_(0));

  for (int n = 1; n < N; ++n) {
    phi_.col(n-1) = exp(-beta*(x(n) - x(n-1)));
    S *= (phi_.col(n-1) * phi_.col(n-1).transpose()).array();
    for (int k = 0; k < J; ++k)
      tmp(k) = (alpha_ * S.col(k)).sum();
    D_(n) = (diag(n) + alpha_sum - (alpha_ * tmp).sum()).real();
    if (D_(n) < T(0.0)) return -n;
    //D_(n) = sqrt(D_(n));
    X_.col(n).array() = (1.0 - tmp) / D_(n);
    S += D_(n) * (X_.col(n) * X_.col(n).transpose()).array();
  }

  this->j_ = J;
  this->n_ = N;
  this->log_det_ = log(D_).sum();
  this->computed_ = true;

  return 0;
}

template <typename T>
void CholeskySolver<T>::solve (const Eigen::MatrixXd& b, T* x) const {
  if (b.rows() != this->n_) throw dimension_mismatch();
  if (!(this->computed_)) throw compute_exception();
  int J = this->j_, N = this->n_, nrhs = b.cols();
  Eigen::Array<std::complex<T>, Eigen::Dynamic, 1> f(J);
  Eigen::Map<Eigen::Array<T, Eigen::Dynamic, Eigen::Dynamic> > xout(x, N, nrhs);

  for (int k = 0; k < nrhs; ++k) {
    // Forward pass
    f.setConstant(T(0.0));
    xout(0, k) = b(0, k);
    for (int n = 1; n < N; ++n) {
      f = phi_.col(n-1).array() * (f + alpha_ * X_.col(n-1).array() * xout(n-1, k));
      xout(n, k) = b(n, k) - f.sum().real();
    }

    xout.col(k) /= D_;

    // Backwards pass
    f.setConstant(T(0.0));
    for (int n = N-2; n >= 0; --n) {
      f = phi_.col(n).array() * (f + alpha_ * xout(n+1, k));
      xout(n, k) = xout(n, k) - (f * X_.col(n).array()).sum().real();
    }
  }
}

//template <typename T>
//Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> BandSolver<T>::dot (
//  const Eigen::Matrix<T, Eigen::Dynamic, 1>& alpha_real,
//  const Eigen::Matrix<T, Eigen::Dynamic, 1>& beta_real,
//  const Eigen::Matrix<T, Eigen::Dynamic, 1>& alpha_complex_real,
//  const Eigen::Matrix<T, Eigen::Dynamic, 1>& alpha_complex_imag,
//  const Eigen::Matrix<T, Eigen::Dynamic, 1>& beta_complex_real,
//  const Eigen::Matrix<T, Eigen::Dynamic, 1>& beta_complex_imag,
//  const Eigen::VectorXd& t,
//  const Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>& b_in
//) {
//  if (t.rows() != b_in.rows()) throw dimension_mismatch();
//  Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> bex =
//    build_b_ext(alpha_real, beta_real, alpha_complex_real, alpha_complex_imag,
//                beta_complex_real, beta_complex_imag, t, b_in);

//  int p_real = alpha_real.rows(),
//      p_complex = alpha_complex_real.rows(),
//      n = t.rows(),
//      nrhs = b_in.cols();
//  BLOCKSIZE_BASE
//  WIDTH

//  // Build the extended matrix
//  Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> A;
//  build_matrix(alpha_real, beta_real, alpha_complex_real, alpha_complex_imag,
//               beta_complex_real, beta_complex_imag, 1, t, A);

//  Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> b_out(b_in.rows(), b_in.cols());
//  // Do the dot product - WARNING: this assumes symmetry!
//  for (int j = 0; j < nrhs; ++j) {
//    for (int i = 0; i < n; ++i) {
//      int k = block_size * i;
//      b_out(i, j) = 0.0;
//      for (int kp = std::max(0, width - k); kp < std::min(2*width+1, dim_ext + width - k); ++kp)
//        b_out(i, j) += A(kp, k) * bex(k + kp - width, j);
//    }
//  }

//  return b_out;
//}

};
};

#endif
