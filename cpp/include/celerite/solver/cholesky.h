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

  int N = this->n_ = x.rows();
  int J_real = a_real.rows(), J_comp = a_comp.rows();

  J_ = J_real + 2*J_comp;
  Eigen::Array<T, Eigen::Dynamic, 1> a1(J_real), a2(J_comp), b2(J_comp),
                                      c1(J_real), c2(J_comp), d2(J_comp),
                                      cd, sd;
  a1 << a_real;
  a2 << a_comp;
  b2 << b_comp;
  c1 << c_real;
  c2 << c_comp;
  d2 << d_comp;

  T a_sum = a1.sum() + a2.sum();

  phi_.resize(J_, N-1);
  u_.resize(J_, N-1);
  X_.resize(J_, N);
  D_.resize(N);

  // Work arrays.
  Eigen::Matrix<T, Eigen::Dynamic, 1> tmp;
  Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> S;

  // First row
  D_(0) = diag(0) + a_sum;
  X_.col(0).head(J_real).setConstant(T(1.0) / D_(0));
  X_.col(0).segment(J_real, J_comp) = cos(d2*x(0)) / D_(0);
  X_.col(0).segment(J_real+J_comp, J_comp) = sin(d2*x(0)) / D_(0);
  S.noalias() = D_(0) * X_.col(0) * X_.col(0).transpose();

  for (int n = 1; n < N; ++n) {
    cd = cos(d2*x(n));
    sd = sin(d2*x(n));

    u_.col(n-1).head(J_real) = a1;
    u_.col(n-1).segment(J_real, J_comp) = a2 * cd + b2 * sd;
    u_.col(n-1).segment(J_real+J_comp, J_comp) = a2 * sd - b2 * cd;

    X_.col(n).head(J_real).setOnes();
    X_.col(n).segment(J_real, J_comp) = cd;
    X_.col(n).segment(J_real+J_comp, J_comp) = sd;

    T dx = x(n) - x(n-1);
    phi_.col(n-1).head(J_real) = exp(-c1*dx);
    phi_.col(n-1).segment(J_real, J_comp) = exp(-c2*dx);
    phi_.col(n-1).segment(J_real+J_comp, J_comp) = phi_.col(n-1).segment(J_real, J_comp);

    S.array() *= (phi_.col(n-1) * phi_.col(n-1).transpose()).array();
    tmp = u_.col(n-1).transpose() * S;
    D_(n) = diag(n) + a_sum - tmp.transpose() * u_.col(n-1);
    X_.col(n) = (T(1.0) / D_(n)) * (X_.col(n) - tmp);
    S.noalias() += D_(n) * X_.col(n) * X_.col(n).transpose();
  }

  this->log_det_ = log(D_.array()).sum();
  this->computed_ = true;

  return 0;
};

void solve (const Eigen::MatrixXd& b, T* x) const {
  if (b.rows() != this->n_) throw dimension_mismatch();
  if (!(this->computed_)) throw compute_exception();

  int nrhs = b.cols();
  Eigen::Matrix<T, Eigen::Dynamic, 1> f(J_);
  Eigen::Map<Eigen::Array<T, Eigen::Dynamic, Eigen::Dynamic> > xout(x, this->n_, nrhs);

  for (int k = 0; k < nrhs; ++k) {
    // Forward
    f.setConstant(T(0.0));
    xout(0, k) = b(0, k);
    for (int n = 1; n < this->n_; ++n) {
      f = phi_.col(n-1).asDiagonal() * (f + X_.col(n-1) * xout(n-1, k));
      xout(n, k) = b(n, k) - u_.col(n-1).transpose() * f;
    }
    xout.col(k) /= D_.array();

    // Backward
    f.setConstant(T(0.0));
    for (int n = this->n_-2; n >= 0; --n) {
      f = phi_.col(n).asDiagonal() * (f + u_.col(n) * xout(n+1, k));
      xout(n, k) = xout(n, k) - X_.col(n).transpose() * f;
    }
  }
};

Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> dot_L (const Eigen::MatrixXd& z) const {
  if (z.rows() != this->n_) throw dimension_mismatch();
  if (!(this->computed_)) throw compute_exception();

  T tmp;
  int N = z.rows(), nrhs = z.cols();
  Eigen::Array<T, Eigen::Dynamic, 1> D = sqrt(D_);
  Eigen::Matrix<T, Eigen::Dynamic, 1> f(J_);
  Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> y(N, nrhs);

  for (int k = 0; k < nrhs; ++k) {
    f.setZero();
    tmp = z(0, k) * D(0);
    y(0, k) = tmp;
    for (int n = 1; n < N; ++n) {
      f = phi_.col(n-1).asDiagonal() * (f + X_.col(n-1) * tmp);
      tmp = D(n) * z(n, k);
      y(n, k) = tmp + u_.col(n-1).transpose() * f;
    }
  }

  return y;

  //z = np.array(y) * D
  //y = np.empty(N)
  //y[0] = z[0]
  //f1 = 0.0
  //f2 = 0.0
  //for n in range(1, N):
  //    f1 = phi[n-1] * (f1 + X1[n-1] * z[n-1])
  //    f2 = phi[n-1] * (f2 + X2[n-1] * z[n-1])
  //    y[n] = (z[n] + np.dot(ut1[n], f1) + np.dot(ut2[n], f2))
  //print("Forward dot error: {0}".format(np.max(np.abs(y - np.dot(L, z)))))
};

Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> dot (
  const Eigen::Matrix<T, Eigen::Dynamic, 1>& a_real,
  const Eigen::Matrix<T, Eigen::Dynamic, 1>& c_real,
  const Eigen::Matrix<T, Eigen::Dynamic, 1>& a_comp,
  const Eigen::Matrix<T, Eigen::Dynamic, 1>& b_comp,
  const Eigen::Matrix<T, Eigen::Dynamic, 1>& c_comp,
  const Eigen::Matrix<T, Eigen::Dynamic, 1>& d_comp,
  const Eigen::VectorXd& x,
  const Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>& z
) {
  if (x.rows() != z.rows()) throw dimension_mismatch();

  int N = z.rows(), nrhs = z.cols();
  int J_real = a_real.rows(), J_comp = a_comp.rows(), J = J_real + 2*J_comp;
  Eigen::Array<T, Eigen::Dynamic, 1> a1(J_real), a2(J_comp), b2(J_comp),
                                     c1(J_real), c2(J_comp), d2(J_comp),
                                     cd, sd;
  a1 << a_real;
  a2 << a_comp;
  b2 << b_comp;
  c1 << c_real;
  c2 << c_comp;
  d2 << d_comp;

  T a_sum = a1.sum() + a2.sum();

  Eigen::Matrix<T, Eigen::Dynamic, 1> f(J);
  Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> y(N, nrhs), phi(J, N-1), u(J, N-1), v(J, N-1);

  cd = cos(d2*x(0));
  sd = sin(d2*x(0));
  for (int n = 0; n < N-1; ++n) {
    v.col(n).head(J_real).setOnes();
    v.col(n).segment(J_real, J_comp) = cd;
    v.col(n).segment(J_real+J_comp, J_comp) = sd;

    cd = cos(d2*x(n+1));
    sd = sin(d2*x(n+1));
    u.col(n).head(J_real) = a1;
    u.col(n).segment(J_real, J_comp) = a2 * cd + b2 * sd;
    u.col(n).segment(J_real+J_comp, J_comp) = a2 * sd - b2 * cd;

    T dx = x(n+1) - x(n);
    phi.col(n).head(J_real) = exp(-c1*dx);
    phi.col(n).segment(J_real, J_comp) = exp(-c2*dx);
    phi.col(n).segment(J_real+J_comp, J_comp) = phi.col(n).segment(J_real, J_comp);
  }

  for (int k = 0; k < nrhs; ++k) {
    y(N-1, k) = a_sum * z(N-1, k);
    f.setZero();
    for (int n = N-2; n >= 0; --n) {
      f = phi.col(n).asDiagonal() * (f + u.col(n) * z(n+1, k));
      y(n, k) = a_sum * z(n, k) + v.col(n).transpose() * f;
    }

    f.setZero();
    for (int n = 1; n < N; ++n) {
      f = phi.col(n-1).asDiagonal() * (f + v.col(n-1) * z(n-1, k));
      y(n, k) += u.col(n-1).transpose() * f;
    }
  }

  return y;
};

  using Solver<T>::compute;
  using Solver<T>::solve;

protected:
  int J_;
  Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> u_, phi_, X_;
  Eigen::Matrix<T, Eigen::Dynamic, 1> D_;

};

};
};

#endif
