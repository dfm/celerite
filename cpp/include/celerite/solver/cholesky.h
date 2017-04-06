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
  const Eigen::Matrix<T, Eigen::Dynamic, 1>& a_real,
  const Eigen::Matrix<T, Eigen::Dynamic, 1>& c_real,
  const Eigen::Matrix<T, Eigen::Dynamic, 1>& a_comp,
  const Eigen::Matrix<T, Eigen::Dynamic, 1>& b_comp,
  const Eigen::Matrix<T, Eigen::Dynamic, 1>& c_comp,
  const Eigen::Matrix<T, Eigen::Dynamic, 1>& d_comp,
  const Eigen::VectorXd& x,
  const Eigen::Matrix<T, Eigen::Dynamic, 1>& diag
)
{
  this->computed_ = false;
  if (x.rows() != diag.rows()) return SOLVER_DIMENSION_MISMATCH;
  if (a_real.rows() != c_real.rows()) return SOLVER_DIMENSION_MISMATCH;
  if (a_comp.rows() != b_comp.rows()) return SOLVER_DIMENSION_MISMATCH;
  if (a_comp.rows() != c_comp.rows()) return SOLVER_DIMENSION_MISMATCH;
  if (a_comp.rows() != d_comp.rows()) return SOLVER_DIMENSION_MISMATCH;

  this->n_ = N_ = x.rows();
  J_real_ = a_real.rows();
  J_comp_ = a_comp.rows();
  //Eigen::Array<T, Eigen::Dynamic, 1> a(J_comp_), b(J_comp_), c(J_comp_), d(J_comp_);
  //a << a_comp, a_real;
  //b.setConstant(T(0.0));
  //b.head(b_comp.rows()) << b_comp;
  //c << c_comp, c_real;
  //d.setConstant(T(0.0));
  //d.head(b_comp.rows()) << d_comp;

  T a_sum = a_real.sum() + a_comp.sum();

  phi_real_.resize(J_real_, N_-1);
  u_real_ = a_real.array();
  X_real_.resize(J_real_, N_);

  phi_comp_.resize(J_comp_, N_-1);
  u1_comp_.resize(J_comp_, N_-1);
  u2_comp_.resize(J_comp_, N_-1);
  X1_comp_.resize(J_comp_, N_);
  X2_comp_.resize(J_comp_, N_);

  D_.resize(N_);

  // Work arrays.
  Eigen::Matrix<T, Eigen::Dynamic, 1> tmp00, tmp01, tmp10, tmp02, tmp20, tmp11, tmp12, tmp21, tmp22;
  Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> S00(J_real_, J_real_), S01, S02, S11, S12, S22;

  // First row
  D_(0) = diag(0) + a_sum;
  X_real_.col(0).setConstant(1.0 / D_(0));
  X1_comp_.col(0) = cos(d_comp.array() * x(0)) / D_(0);
  X2_comp_.col(0) = sin(d_comp.array() * x(0)) / D_(0);

  S00.setConstant(1.0 / D_(0));
  S01 = D_(0) * X_real_.col(0) * X1_comp_.col(0).transpose();
  S02 = D_(0) * X_real_.col(0) * X2_comp_.col(0).transpose();
  S11 = D_(0) * X1_comp_.col(0) * X1_comp_.col(0).transpose();
  S12 = D_(0) * X1_comp_.col(0) * X2_comp_.col(0).transpose();
  S22 = D_(0) * X2_comp_.col(0) * X2_comp_.col(0).transpose();

  for (int n = 1; n < N_; ++n) {
    u1_comp_.col(n-1) = a_comp.array() * cos(d_comp.array() * x(n)) + b_comp.array() * sin(d_comp.array() * x(n));
    u2_comp_.col(n-1) = a_comp.array() * sin(d_comp.array() * x(n)) - b_comp.array() * cos(d_comp.array() * x(n));

    phi_real_.col(n-1) = exp(-c_real.array() * (x(n) - x(n-1)));
    phi_comp_.col(n-1) = exp(-c_comp.array() * (x(n) - x(n-1)));

    S00 = phi_real_.col(n-1).asDiagonal() * S00 * phi_real_.col(n-1).asDiagonal();
    S01 = phi_real_.col(n-1).asDiagonal() * S01 * phi_comp_.col(n-1).asDiagonal();
    S02 = phi_real_.col(n-1).asDiagonal() * S02 * phi_comp_.col(n-1).asDiagonal();
    S11 = phi_comp_.col(n-1).asDiagonal() * S11 * phi_comp_.col(n-1).asDiagonal();
    S12 = phi_comp_.col(n-1).asDiagonal() * S12 * phi_comp_.col(n-1).asDiagonal();
    S22 = phi_comp_.col(n-1).asDiagonal() * S22 * phi_comp_.col(n-1).asDiagonal();

    tmp00 = u_real_.transpose() * S00;
    tmp01 = u_real_.transpose() * S01;
    tmp10 = (S01 * u1_comp_.col(n-1)).transpose();
    tmp02 = u_real_.transpose() * S02;
    tmp20 = (S02 * u2_comp_.col(n-1)).transpose();
    tmp11 = u1_comp_.col(n-1).transpose() * S11;
    tmp12 = u1_comp_.col(n-1).transpose() * S12;
    tmp21 = (S12 * u2_comp_.col(n-1)).transpose();
    tmp22 = u2_comp_.col(n-1).transpose() * S22;

    D_(n) = diag(n) + a_sum
      - (tmp00 + 2.0 * (tmp10 + tmp20)).transpose() * u_real_
      - tmp11.transpose() * u1_comp_.col(n-1)
      - (2.0 * tmp12 + tmp22).transpose() * u2_comp_.col(n-1);

    X_real_.col(n)  = (1.0 - (tmp00 + tmp10 + tmp20).array()) / D_(n);
    X1_comp_.col(n) = (cos(d_comp.array()*x(n)) - (tmp01 + tmp11 + tmp21).array()) / D_(n);
    X2_comp_.col(n) = (sin(d_comp.array()*x(n)) - (tmp02 + tmp12 + tmp22).array()) / D_(n);

    S00 = S00 + D_(n) * X_real_.col(n) * X_real_.col(n).transpose();
    S01 = S01 + D_(n) * X_real_.col(n) * X1_comp_.col(n).transpose();
    S02 = S02 + D_(n) * X_real_.col(n) * X2_comp_.col(n).transpose();
    S11 = S11 + D_(n) * X1_comp_.col(n) * X1_comp_.col(n).transpose();
    S12 = S12 + D_(n) * X1_comp_.col(n) * X2_comp_.col(n).transpose();
    S22 = S22 + D_(n) * X2_comp_.col(n) * X2_comp_.col(n).transpose();
  }

  this->log_det_ = log(D_.array()).sum();
  this->computed_ = true;

  return 0;
};

void solve (const Eigen::MatrixXd& b, T* x) const {
  if (b.rows() != this->n_) throw dimension_mismatch();
  if (!(this->computed_)) throw compute_exception();

  int nrhs = b.cols();
  Eigen::Matrix<T, Eigen::Dynamic, 1> f0(J_real_), f1(J_comp_), f2(J_comp_);
  Eigen::Map<Eigen::Array<T, Eigen::Dynamic, Eigen::Dynamic> > xout(x, N_, nrhs);

  for (int k = 0; k < nrhs; ++k) {
    // Forward
    f0.setConstant(T(0.0));
    f1.setConstant(T(0.0));
    f2.setConstant(T(0.0));
    xout(0, k) = b(0, k);
    for (int n = 1; n < N_; ++n) {
      f0 = phi_real_.col(n-1).asDiagonal() * (f0 + X_real_.col(n-1) * xout(n-1, k));
      f1 = phi_comp_.col(n-1).asDiagonal() * (f1 + X1_comp_.col(n-1) * xout(n-1, k));
      f2 = phi_comp_.col(n-1).asDiagonal() * (f2 + X2_comp_.col(n-1) * xout(n-1, k));
      xout(n, k) = b(n, k) - u_real_.transpose() * f0 - u1_comp_.col(n-1).transpose() * f1 - u2_comp_.col(n-1).transpose() * f2;
    }
    xout.col(k) /= D_.array();

    // Backward
    f0.setConstant(T(0.0));
    f1.setConstant(T(0.0));
    f2.setConstant(T(0.0));
    for (int n = N_-2; n >= 0; --n) {
      f0 = phi_real_.col(n).asDiagonal() * (f0 + u_real_ * xout(n+1, k));
      f1 = phi_comp_.col(n).asDiagonal() * (f1 + u1_comp_.col(n) * xout(n+1, k));
      f2 = phi_comp_.col(n).asDiagonal() * (f2 + u2_comp_.col(n) * xout(n+1, k));
      xout(n, k) = xout(n, k) - X_real_.col(n).transpose() * f0 - X1_comp_.col(n).transpose() * f1 - X2_comp_.col(n).transpose() * f2;
    }
  }
};

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
  int N_, J_real_, J_comp_;
  Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>
    phi_real_, X_real_,
    u1_comp_, u2_comp_, phi_comp_, X1_comp_, X2_comp_;
  Eigen::Matrix<T, Eigen::Dynamic, 1> D_, u_real_;

};

//template <typename T>
//int CholeskySolver<T>::compute (
//    const Eigen::Matrix<T, Eigen::Dynamic, 1>& a_real,
//    const Eigen::Matrix<T, Eigen::Dynamic, 1>& c_real,
//    const Eigen::Matrix<T, Eigen::Dynamic, 1>& a_comp,
//    const Eigen::Matrix<T, Eigen::Dynamic, 1>& b_comp,
//    const Eigen::Matrix<T, Eigen::Dynamic, 1>& c_comp,
//    const Eigen::Matrix<T, Eigen::Dynamic, 1>& d_comp,
//    const Eigen::VectorXd& x,
//    const Eigen::Matrix<T, Eigen::Dynamic, 1>& diag
//)
//{
//  this->computed_ = false;
//  if (x.rows() != diag.rows()) return SOLVER_DIMENSION_MISMATCH;
//  if (alpha_real.rows() != beta_real.rows()) return SOLVER_DIMENSION_MISMATCH;
//  if (alpha_complex_real.rows() != alpha_complex_imag.rows()) return SOLVER_DIMENSION_MISMATCH;
//  if (alpha_complex_real.rows() != beta_complex_real.rows()) return SOLVER_DIMENSION_MISMATCH;
//  if (alpha_complex_real.rows() != beta_complex_imag.rows()) return SOLVER_DIMENSION_MISMATCH;

//  int N = x.rows(), J = alpha_real.rows() + 2 * alpha_complex_real.rows();
//  phi_.resize(J, N-1);
//  D_.resize(N);
//  X_.resize(J, N);
//  alpha_.resize(J);
//  Eigen::Array<std::complex<T>, Eigen::Dynamic, Eigen::Dynamic> S(J, J);
//  Eigen::Array<std::complex<T>, Eigen::Dynamic, 1> beta(J), tmp(J);

//  int i = 0;
//  for (int j = 0; j < alpha_real.rows(); ++j) {
//    alpha_(i) = alpha_real(j);
//    beta(i++) = beta_real(j);
//  }
//  for (int j = 0; j < alpha_complex_real.rows(); ++j) {
//    alpha_(i) = 0.5*std::complex<T>(alpha_complex_real(j), alpha_complex_imag(j));
//    beta(i++) = std::complex<T>(beta_complex_real(j), beta_complex_imag(j));
//    alpha_(i) = 0.5*std::complex<T>(alpha_complex_real(j), -alpha_complex_imag(j));
//    beta(i++) = std::complex<T>(beta_complex_real(j), -beta_complex_imag(j));
//  }

//  // First pass
//  T alpha_sum = alpha_.sum().real();
//  D_(0) = diag(0) + alpha_sum;
//  S.setConstant(1.0 / D_(0));
//  X_.col(0).setConstant(1.0 / D_(0));

//  for (int n = 1; n < N; ++n) {
//    phi_.col(n-1) = exp(-beta*(x(n) - x(n-1)));
//    S *= (phi_.col(n-1) * phi_.col(n-1).transpose()).array();
//    for (int k = 0; k < J; ++k)
//      tmp(k) = (alpha_ * S.col(k)).sum();
//    D_(n) = (diag(n) + alpha_sum - (alpha_ * tmp).sum()).real();
//    if (D_(n) < T(0.0)) return -n;
//    //D_(n) = sqrt(D_(n));
//    X_.col(n).array() = (1.0 - tmp) / D_(n);
//    S += D_(n) * (X_.col(n) * X_.col(n).transpose()).array();
//  }

//  this->j_ = J;
//  this->n_ = N;
//  this->log_det_ = log(D_).sum();
//  this->computed_ = true;

//  return 0;
//}


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
