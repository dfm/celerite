#ifndef _CELERITE_SOLVER_CHOLESKY_H_
#define _CELERITE_SOLVER_CHOLESKY_H_

#include <cmath>
#include <complex>
#include <Eigen/Core>

#include "celerite/utils.h"
#include "celerite/exceptions.h"

#include "celerite/solver/solver.h"

#ifndef CHOLTURN
#define CHOLTURN 32
#endif

namespace celerite {
namespace solver {

template <typename T>
class CholeskySolver : public Solver<T> {
private:
typedef Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> matrix_t;
typedef Eigen::Matrix<T, Eigen::Dynamic, 1> vector_t;

public:
CholeskySolver () : Solver<T>() {};
~CholeskySolver () {};

void compute (
  const T& jitter,
  const vector_t& a_real,
  const vector_t& c_real,
  const vector_t& a_comp,
  const vector_t& b_comp,
  const vector_t& c_comp,
  const vector_t& d_comp,
  const Eigen::VectorXd& x,
  const Eigen::VectorXd& diag
)
{
  this->computed_ = false;
  if (x.rows() != diag.rows()) throw dimension_mismatch();
  if (a_real.rows() != c_real.rows()) throw dimension_mismatch();
  if (a_comp.rows() != b_comp.rows()) throw dimension_mismatch();
  if (a_comp.rows() != c_comp.rows()) throw dimension_mismatch();
  if (a_comp.rows() != d_comp.rows()) throw dimension_mismatch();

  int N = this->N_ = x.rows();
  int J_real = a_real.rows(), J_comp = a_comp.rows();
  int J = J_ = J_real + 2*J_comp;
  phi_.resize(J, N-1);
  u_.resize(J, N-1);
  X_.resize(J, N);

  D_ = diag.array() + a_real.sum() + a_comp.sum() + jitter;

  if (J <= CHOLTURN) {

    // We unroll the loops for the smallest models for speed.
    int j, k;
    T value, dx, tmp, cd, sd, phij, uj, xj;
    T *Xnm1, *Xn, *phin;
    Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> S(J, J);

    value = T(1.0) / D_(0);
    for (int j = 0; j < J_real; ++j) {
      X_(j, 0) = value;
    }
    for (int j = 0, k = J_real; j < J_comp; ++j, k += 2) {
      X_(k,   0) = cos(d_comp(j) * x(0)) * value;
      X_(k+1, 0) = sin(d_comp(j) * x(0)) * value;
    }

    S.setZero();
    for (int n = 1; n < N; ++n) {
      Xnm1 = X_.data() + J * (n-1);
      Xn = X_.data() + J * n;
      phin = phi_.data() + J * (n-1);

      dx = x(n) - x(n-1);
      for (j = 0; j < J_real; ++j) {
        phin[j] = exp(-c_real(j)*dx);
        u_(j, n-1) = a_real(j);
        Xn[j] = T(1.0);
      }
      for (j = 0, k = J_real; j < J_comp; ++j, k += 2) {
        value = exp(-c_comp(j)*dx);
        phin[k]   = value;
        phin[k+1] = value;
        cd = cos(d_comp(j)*x(n));
        sd = sin(d_comp(j)*x(n));
        u_(k,   n-1) = a_comp(j)*cd + b_comp(j)*sd;
        u_(k+1, n-1) = a_comp(j)*sd - b_comp(j)*cd;
        Xn[k]   = cd;
        Xn[k+1] = sd;
      }

      for (j = 0; j < J; ++j) {
        phij = phin[j];
        xj = D_(n-1)*Xnm1[j];
        for (k = 0; k <= j; ++k) {
          S(k, j) = phij*phin[k]*(S(k, j) + xj*Xnm1[k]);
        }
      }

      for (j = 0; j < J; ++j) {
        uj = u_(j, n-1);
        xj = Xn[j];
        for (k = 0; k < j; ++k) {
          tmp = u_(k, n-1) * S(k, j);
          D_(n) -= 2.0*uj*tmp;
          xj -= tmp;
          Xn[k] -= uj*S(k, j);
        }
        tmp = uj*S(j, j);
        D_(n) -= uj*tmp;
        Xn[j] = xj - tmp;
      }
      if (D_(n) < 0) throw linalg_exception();
      X_.col(n) /= D_(n);
    }

  } else if (J_comp == 0) {

    // Real only.
    Eigen::Matrix<T, Eigen::Dynamic, 1> tmp(J);
    Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> S(J, J);

    X_.col(0).setConstant(T(1.0) / D_(0));
    S.setZero();
    for (int n = 1; n < N; ++n) {
      phi_.col(n-1) = exp(-c_real.array()*(x(n) - x(n-1)));
      S.noalias() += D_(n-1) * X_.col(n-1) * X_.col(n-1).transpose();
      S.array() *= (phi_.col(n-1) * phi_.col(n-1).transpose()).array();
      u_.col(n-1) = a_real;
      X_.col(n).setOnes();
      tmp = u_.col(n-1).transpose() * S;
      D_(n) -= tmp.transpose().dot(u_.col(n-1));
      if (D_(n) < 0) throw linalg_exception();
      X_.col(n) = (T(1.0) / D_(n)) * (X_.col(n) - tmp);
    }

  } else {

    // General case.
    Eigen::Array<T, Eigen::Dynamic, 1> cd, sd;
    Eigen::Matrix<T, Eigen::Dynamic, 1> tmp;
    Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> S(J, J);

    X_.col(0).head(J_real).setConstant(T(1.0) / D_(0));
    X_.col(0).segment(J_real, J_comp) = cos(d_comp.array()*x(0)) / D_(0);
    X_.col(0).segment(J_real+J_comp, J_comp) = sin(d_comp.array()*x(0)) / D_(0);
    S.setZero();
    for (int n = 1; n < N; ++n) {
      cd = cos(d_comp.array()*x(n));
      sd = sin(d_comp.array()*x(n));

      T dx = x(n) - x(n-1);
      phi_.col(n-1).head(J_real) = exp(-c_real.array()*dx);
      phi_.col(n-1).segment(J_real, J_comp) = exp(-c_comp.array()*dx);
      phi_.col(n-1).segment(J_real+J_comp, J_comp) = phi_.col(n-1).segment(J_real, J_comp);
      S.noalias() += D_(n-1) * X_.col(n-1) * X_.col(n-1).transpose();
      S.array() *= (phi_.col(n-1) * phi_.col(n-1).transpose()).array();

      u_.col(n-1).head(J_real) = a_real;
      u_.col(n-1).segment(J_real, J_comp) = a_comp.array() * cd + b_comp.array() * sd;
      u_.col(n-1).segment(J_real+J_comp, J_comp) = a_comp.array() * sd - b_comp.array() * cd;

      X_.col(n).head(J_real).setOnes();
      X_.col(n).segment(J_real, J_comp) = cd;
      X_.col(n).segment(J_real+J_comp, J_comp) = sd;

      tmp = u_.col(n-1).transpose() * S;
      D_(n) -= tmp.transpose().dot(u_.col(n-1));
      if (D_(n) < 0) throw linalg_exception();
      X_.col(n) = (T(1.0) / D_(n)) * (X_.col(n) - tmp);
    }

  }

  this->log_det_ = log(D_.array()).sum();
  this->computed_ = true;
};

matrix_t solve (const Eigen::MatrixXd& b) const {
  if (b.rows() != this->N_) throw dimension_mismatch();
  if (!(this->computed_)) throw compute_exception();

  int j, k, n, nrhs = b.cols(), N = this->N_, J = J_;
  vector_t f(J);
  matrix_t x(N, nrhs);

  if (J < 16) {

    // Unroll smaller loops.
    for (k = 0; k < nrhs; ++k) {
      // Forward
      f.setConstant(T(0.0));
      x(0, k) = b(0, k);
      for (n = 1; n < N; ++n) {
        x(n, k) = b(n, k);
        for (j = 0; j < J; ++j) {
          f(j) = phi_(j, n-1) * (f(j) + X_(j, n-1) * x(n-1, k));
          x(n, k) -= u_(j, n-1) * f(j);
        }
      }
      x.col(k).array() /= D_.array();

      // Backward
      f.setConstant(T(0.0));
      for (n = N-2; n >= 0; --n) {
        for (j = 0; j < J; ++j) {
          f(j) = phi_(j, n) * (f(j) + u_(j, n) * x(n+1, k));
          x(n, k) -= X_(j, n) * f(j);
        }
      }
    }

  } else {

    for (k = 0; k < nrhs; ++k) {
      // Forward
      f.setConstant(T(0.0));
      x(0, k) = b(0, k);
      for (n = 1; n < N; ++n) {
        f = phi_.col(n-1).asDiagonal() * (f + X_.col(n-1) * x(n-1, k));
        x(n, k) = b(n, k) - u_.col(n-1).transpose().dot(f);
      }
      x.col(k).array() /= D_.array();

      // Backward
      f.setConstant(T(0.0));
      for (n = N-2; n >= 0; --n) {
        f = phi_.col(n).asDiagonal() * (f + u_.col(n) * x(n+1, k));
        x(n, k) = x(n, k) - X_.col(n).transpose().dot(f);
      }
    }

  }

  return x;
};

matrix_t dot_L (const Eigen::MatrixXd& z) const {
  if (z.rows() != this->N_) throw dimension_mismatch();
  if (!(this->computed_)) throw compute_exception();

  T tmp;
  int N = z.rows(), nrhs = z.cols();
  Eigen::Array<T, Eigen::Dynamic, 1> D = sqrt(D_.array());
  vector_t f(J_);
  matrix_t y(N, nrhs);

  for (int k = 0; k < nrhs; ++k) {
    f.setZero();
    tmp = z(0, k) * D(0);
    y(0, k) = tmp;
    for (int n = 1; n < N; ++n) {
      f = phi_.col(n-1).asDiagonal() * (f + X_.col(n-1) * tmp);
      tmp = D(n) * z(n, k);
      y(n, k) = tmp + u_.col(n-1).transpose().dot(f);
    }
  }

  return y;
};

matrix_t dot (
  const T& jitter,
  const vector_t& a_real,
  const vector_t& c_real,
  const vector_t& a_comp,
  const vector_t& b_comp,
  const vector_t& c_comp,
  const vector_t& d_comp,
  const Eigen::VectorXd& x,
  const Eigen::MatrixXd& z
) {
  if (x.rows() != z.rows()) throw dimension_mismatch();
  if (a_real.rows() != c_real.rows()) throw dimension_mismatch();
  if (a_comp.rows() != b_comp.rows()) throw dimension_mismatch();
  if (a_comp.rows() != c_comp.rows()) throw dimension_mismatch();
  if (a_comp.rows() != d_comp.rows()) throw dimension_mismatch();

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

  T a_sum = jitter + a1.sum() + a2.sum();

  vector_t f(J);
  matrix_t y(N, nrhs), phi(J, N-1), u(J, N-1), v(J, N-1);

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
      y(n, k) = a_sum * z(n, k) + v.col(n).transpose().dot(f);
    }

    f.setZero();
    for (int n = 1; n < N; ++n) {
      f = phi.col(n-1).asDiagonal() * (f + v.col(n-1) * z(n-1, k));
      y(n, k) += u.col(n-1).transpose().dot(f);
    }
  }

  return y;
};

using Solver<T>::compute;

protected:
int J_;
matrix_t u_, phi_, X_;
vector_t D_;

};

};
};

#endif
