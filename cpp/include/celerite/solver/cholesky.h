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

  int N = this->n_ = x.rows(), J0 = a_real.rows();
  J1_ = a_real.rows() + a_comp.rows();
  J2_ = a_comp.rows();
  Eigen::Array<T, Eigen::Dynamic, 1> a1(J1_), a2(J2_), b1(J1_), b2(J2_), c1(J1_), c2(J2_), d1(J1_), d2(J2_), cd1(J1_), sd2;
  a1 << a_comp, a_real;
  a2 << a_comp;
  b1.setConstant(T(0.0));
  b1.head(J2_) << b_comp;
  b2 << b_comp;
  c1 << c_comp, c_real;
  c2 << c_comp;
  d1.setConstant(T(0.0));
  d1.head(J2_) << d_comp;
  d2 << d_comp;

  T a_sum = a1.sum();
  phi1_.resize(J1_, N-1);
  phi2_.resize(J2_, N-1);
  u1_.resize(J1_, N-1);
  u2_.resize(J2_, N-1);
  X1_.resize(J1_, N);
  X2_.resize(J2_, N);
  D_.resize(N);

  // Work arrays.
  Eigen::Matrix<T, Eigen::Dynamic, 1> tmp11, tmp12, tmp21, tmp22;
  Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> S11, S12, S22;

  // First row
  D_(0) = diag(0) + a_sum;
  X1_.col(0) = cos(d1 * x(0)) / D_(0);
  X2_.col(0) = sin(d2 * x(0)) / D_(0);

  S11 = D_(0) * X1_.col(0) * X1_.col(0).transpose();
  S12 = D_(0) * X1_.col(0) * X2_.col(0).transpose();
  S22 = D_(0) * X2_.col(0) * X2_.col(0).transpose();

  cd1.setConstant(T(1.0));
  for (int n = 1; n < N; ++n) {
    cd1.head(J2_) = cos(d2*x(n));
    sd2 = sin(d2*x(n));

    u1_.col(n-1).head(J2_) = a2 * cd1.head(J2_) + b2 * sd2;
    u1_.col(n-1).tail(J0) = a1.tail(J0);
    u2_.col(n-1) = a2 * sd2 - b2 * cd1.head(J2_);
    phi2_.col(n-1) = exp(-c2 * (x(n) - x(n-1)));
    phi1_.col(n-1).head(J2_) = phi2_.col(n-1);
    phi1_.col(n-1).tail(J0) = exp(-c1.tail(J0) * (x(n) - x(n-1)));

    S11 = phi1_.col(n-1).asDiagonal() * S11 * phi1_.col(n-1).asDiagonal();
    S12 = phi1_.col(n-1).asDiagonal() * S12 * phi2_.col(n-1).asDiagonal();
    S22 = phi2_.col(n-1).asDiagonal() * S22 * phi2_.col(n-1).asDiagonal();

    tmp11 = u1_.col(n-1).transpose() * S11;
    tmp12 = u1_.col(n-1).transpose() * S12;
    tmp21 = (S12 * u2_.col(n-1)).transpose();
    tmp22 = u2_.col(n-1).transpose() * S22;

    D_(n) = diag(n) + a_sum
      - tmp11.transpose() * u1_.col(n-1)
      - (2.0 * tmp12 + tmp22).transpose() * u2_.col(n-1);

    X1_.col(n) = (cd1 - (tmp11 + tmp21).array()) / D_(n);
    X2_.col(n) = (sd2 - (tmp12 + tmp22).array()) / D_(n);

    S11 = S11 + D_(n) * X1_.col(n) * X1_.col(n).transpose();
    S12 = S12 + D_(n) * X1_.col(n) * X2_.col(n).transpose();
    S22 = S22 + D_(n) * X2_.col(n) * X2_.col(n).transpose();
  }

  this->log_det_ = log(D_.array()).sum();
  this->computed_ = true;

  return 0;
};

void solve (const Eigen::MatrixXd& b, T* x) const {
  if (b.rows() != this->n_) throw dimension_mismatch();
  if (!(this->computed_)) throw compute_exception();

  int nrhs = b.cols();
  Eigen::Matrix<T, Eigen::Dynamic, 1> f1(J1_), f2(J2_);
  Eigen::Map<Eigen::Array<T, Eigen::Dynamic, Eigen::Dynamic> > xout(x, this->n_, nrhs);

  for (int k = 0; k < nrhs; ++k) {
    // Forward
    f1.setConstant(T(0.0));
    f2.setConstant(T(0.0));
    xout(0, k) = b(0, k);
    for (int n = 1; n < this->n_; ++n) {
      f1 = phi1_.col(n-1).asDiagonal() * (f1 + X1_.col(n-1) * xout(n-1, k));
      f2 = phi2_.col(n-1).asDiagonal() * (f2 + X2_.col(n-1) * xout(n-1, k));
      xout(n, k) = b(n, k) - u1_.col(n-1).transpose() * f1 - u2_.col(n-1).transpose() * f2;
    }
    xout.col(k) /= D_.array();

    // Backward
    f1.setConstant(T(0.0));
    f2.setConstant(T(0.0));
    for (int n = this->n_-2; n >= 0; --n) {
      f1 = phi1_.col(n).asDiagonal() * (f1 + u1_.col(n) * xout(n+1, k));
      f2 = phi2_.col(n).asDiagonal() * (f2 + u2_.col(n) * xout(n+1, k));
      xout(n, k) = xout(n, k) - X1_.col(n).transpose() * f1 - X2_.col(n).transpose() * f2;
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
  int J1_, J2_;
  Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> u1_, u2_, phi1_, phi2_, X1_, X2_;
  Eigen::Matrix<T, Eigen::Dynamic, 1> D_;

};

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
