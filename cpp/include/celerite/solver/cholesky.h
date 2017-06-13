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

template <typename T, int SIZE = Eigen::Dynamic>
class CholeskySolver : public Solver<T> {
private:
typedef Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> matrix_t;
typedef Eigen::Matrix<T, SIZE, Eigen::Dynamic> matrix_j_t;
typedef Eigen::Matrix<T, Eigen::Dynamic, 1> vector_t;
typedef Eigen::Matrix<T, SIZE, 1> vector_j_t;

public:
CholeskySolver () : Solver<T>() {};
~CholeskySolver () {};

/// Compute the Cholesky factorization of the matrix
///
/// @param jitter The jitter of the kernel.
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
  if (SIZE != Eigen::Dynamic && J != SIZE) throw dimension_mismatch();
  phi_.resize(J, N-1);
  u_.resize(J, N-1);
  X_.resize(J, N);

  // Save the inputs. We need these for the 'predict' method.
  a_real_ = a_real;
  c_real_ = c_real;
  a_comp_ = a_comp;
  b_comp_ = b_comp;
  c_comp_ = c_comp;
  d_comp_ = d_comp;
  t_ = x;

  // Initialize the diagonal.
  D_ = diag.array() + a_real.sum() + a_comp.sum() + jitter;
  T Dn = D_(0);

  // Compute the values at x[0]
  {
    T value = 1.0 / Dn,
      t = x(0);
    for (int j = 0; j < J_real; ++j) {
      X_(j, 0) = value;
    }
    for (int j = 0, k = J_real; j < J_comp; ++j, k += 2) {
      T d = d_comp(j) * t;
      X_(k,   0) = cos(d)*value;
      X_(k+1, 0) = sin(d)*value;
    }
  }

  // We'll write the algorithm using a pre-processor macro because that lets
  // us use fixed size matrices for small problems.

#define FIXED_SIZE_HACKZ(SIZE_MACRO)                                          \
  Eigen::Matrix<T, SIZE_MACRO, SIZE_MACRO> S(J, J);                           \
  S.setZero();                                                                \
                                                                              \
  for (int n = 1; n < N; ++n) {                                               \
    T t = x(n),                                                               \
      dx = t - x(n-1);                                                        \
    for (int j = 0; j < J_real; ++j) {                                        \
      phi_(j, n-1) = exp(-c_real(j)*dx);                                      \
      u_(j, n-1) = a_real(j);                                                 \
      X_(j, n) = T(1.0);                                                      \
    }                                                                         \
    for (int j = 0, k = J_real; j < J_comp; ++j, k += 2) {                    \
      T a = a_comp(j),                                                        \
        b = b_comp(j),                                                        \
        d = d_comp(j) * t,                                                    \
        cd = cos(d),                                                          \
        sd = sin(d);                                                          \
      T value = exp(-c_comp(j)*dx);                                           \
      phi_(k,   n-1) = value;                                                 \
      phi_(k+1, n-1) = value;                                                 \
      u_(k,   n-1) = a*cd + b*sd;                                             \
      u_(k+1, n-1) = a*sd - b*cd;                                             \
      X_(k,   n) = cd;                                                        \
      X_(k+1, n) = sd;                                                        \
    }                                                                         \
                                                                              \
    for (int j = 0; j < J; ++j) {                                             \
      T phij = phi_(j, n-1),                                                  \
        xj = Dn*X_(j, n-1);                                                   \
      for (int k = 0; k <= j; ++k) {                                          \
        S(k, j) = phij*phi_(k, n-1)*(S(k, j) + xj*X_(k, n-1));                \
      }                                                                       \
    }                                                                         \
                                                                              \
    Dn = D_(n);                                                               \
    for (int j = 0; j < J; ++j) {                                             \
      T uj = u_(j, n-1),                                                      \
        xj = X_(j, n);                                                        \
      for (int k = 0; k < j; ++k) {                                           \
        T tmp = u_(k, n-1) * S(k, j);                                         \
        Dn -= 2.0*uj*tmp;                                                     \
        xj -= tmp;                                                            \
        X_(k, n) -= uj*S(k, j);                                               \
      }                                                                       \
      T tmp = uj*S(j, j);                                                     \
      Dn -= uj*tmp;                                                           \
      X_(j, n) = xj - tmp;                                                    \
    }                                                                         \
    if (Dn < 0) throw linalg_exception();                                     \
    D_(n) = Dn;                                                               \
    X_.col(n) /= Dn;                                                          \
  }

  if (SIZE == Eigen::Dynamic) {
    switch (J) {
      case 1:  { FIXED_SIZE_HACKZ(1)  break; }
      case 2:  { FIXED_SIZE_HACKZ(2)  break; }
      case 3:  { FIXED_SIZE_HACKZ(3)  break; }
      case 4:  { FIXED_SIZE_HACKZ(4)  break; }
      case 5:  { FIXED_SIZE_HACKZ(5)  break; }
      case 6:  { FIXED_SIZE_HACKZ(6)  break; }
      case 7:  { FIXED_SIZE_HACKZ(7)  break; }
      case 8:  { FIXED_SIZE_HACKZ(8)  break; }
      case 9:  { FIXED_SIZE_HACKZ(9)  break; }
      case 10: { FIXED_SIZE_HACKZ(10) break; }
      case 11: { FIXED_SIZE_HACKZ(11) break; }
      case 12: { FIXED_SIZE_HACKZ(12) break; }
      case 13: { FIXED_SIZE_HACKZ(13) break; }
      case 14: { FIXED_SIZE_HACKZ(14) break; }
      case 15: { FIXED_SIZE_HACKZ(15) break; }
      case 16: { FIXED_SIZE_HACKZ(16) break; }
      default: FIXED_SIZE_HACKZ(Eigen::Dynamic)
    }
  } else {
    // The size was already specified at compile time.
    FIXED_SIZE_HACKZ(SIZE)
  }

#undef FIXED_SIZE_HACKZ

  this->log_det_ = log(D_.array()).sum();
  this->computed_ = true;
};

/// Solve the system ``b^T . A^-1 . b``
///
/// A previous call to `solver::CholeskySolver::compute` defines a matrix
/// ``A`` and this method solves ``b^T . A^-1 . b`` for a vector ``b``.
///
/// @param b The right hand side of the linear system.
matrix_t solve (const Eigen::MatrixXd& b) const {
  if (b.rows() != this->N_) throw dimension_mismatch();
  if (!(this->computed_)) throw compute_exception();

  int nrhs = b.cols(), N = this->N_, J = J_;
  matrix_t x(N, nrhs);

  if (J < 16) {

#define FIXED_SIZE_HACKZ(SIZE_MACRO)                                          \
    Eigen::Matrix<T, SIZE_MACRO, 1> f(J);                                     \
    for (int k = 0; k < nrhs; ++k) {                                          \
      f.setZero();                                                            \
      x(0, k) = b(0, k);                                                      \
      for (int n = 1; n < N; ++n) {                                           \
        T xnm1 = x(n-1, k);                                                   \
        x(n, k) = b(n, k);                                                    \
        for (int j = 0; j < J; ++j) {                                         \
          T value = phi_(j, n-1) * (f(j) + X_(j, n-1) * xnm1);                \
          f(j) = value;                                                       \
          x(n, k) -= u_(j, n-1) * value;                                      \
        }                                                                     \
      }                                                                       \
      x.col(k).array() /= D_.array();                                         \
                                                                              \
      f.setZero();                                                            \
      for (int n = N-2; n >= 0; --n) {                                        \
        T xnp1 = x(n+1, k);                                                   \
        for (int j = 0; j < J; ++j) {                                         \
          T value = phi_(j, n) * (f(j) + u_(j, n) * xnp1);                    \
          f(j) = value;                                                       \
          x(n, k) -= X_(j, n) * value;                                        \
        }                                                                     \
      }                                                                       \
    }

    if (SIZE == Eigen::Dynamic) {
      switch (J) {
        case 1:  { FIXED_SIZE_HACKZ(1)  break; }
        case 2:  { FIXED_SIZE_HACKZ(2)  break; }
        case 3:  { FIXED_SIZE_HACKZ(3)  break; }
        case 4:  { FIXED_SIZE_HACKZ(4)  break; }
        case 5:  { FIXED_SIZE_HACKZ(5)  break; }
        case 6:  { FIXED_SIZE_HACKZ(6)  break; }
        case 7:  { FIXED_SIZE_HACKZ(7)  break; }
        case 8:  { FIXED_SIZE_HACKZ(8)  break; }
        case 9:  { FIXED_SIZE_HACKZ(9)  break; }
        case 10: { FIXED_SIZE_HACKZ(10) break; }
        case 11: { FIXED_SIZE_HACKZ(11) break; }
        case 12: { FIXED_SIZE_HACKZ(12) break; }
        case 13: { FIXED_SIZE_HACKZ(13) break; }
        case 14: { FIXED_SIZE_HACKZ(14) break; }
        case 15: { FIXED_SIZE_HACKZ(15) break; }
        case 16: { FIXED_SIZE_HACKZ(16) break; }
        default: FIXED_SIZE_HACKZ(Eigen::Dynamic)
      }
  } else {
    // The size was already specified at compile time.
    FIXED_SIZE_HACKZ(SIZE)
  }

#undef FIXED_SIZE_HACKZ

  } else {

    vector_j_t f(J);
    for (int k = 0; k < nrhs; ++k) {

      // Forward
      f.setZero();
      T value = b(0, k);
      x(0, k) = value;
      for (int n = 1; n < N; ++n) {
        f = phi_.col(n-1).asDiagonal() * (f + X_.col(n-1) * value);
        value = b(n, k) - u_.col(n-1).transpose().dot(f);
        x(n, k) = value;
      }
      x.col(k).array() /= D_.array();

      // Backward
      f.setZero();
      value = x(N-1, k);
      for (int n = N-2; n >= 0; --n) {
        f = phi_.col(n).asDiagonal() * (f + u_.col(n) * value);
        value = x(n, k) - X_.col(n).transpose().dot(f);
        x(n, k) = value;
      }
    }

  }

  return x;
};

/// Solve the system ``b^T . A^-1 . b``
///
/// A previous call to `solver::CholeskySolver::compute` defines a matrix ``A``
/// and this method solves ``b^T . A^-1 . b`` for a vector ``b``.
///
/// @param b The right hand side of the linear system.
T dot_solve (const Eigen::VectorXd& b) const {
  if (b.rows() != this->N_) throw dimension_mismatch();
  if (!(this->computed_)) throw compute_exception();

  int N = this->N_, J = J_;
  T result;

  if (J < 16) {

#define FIXED_SIZE_HACKZ(SIZE_MACRO)                                        \
    Eigen::Matrix<T, SIZE_MACRO, 1> f(J);                                   \
    f.setZero();                                                            \
    T x, xm1 = b(0);                                                        \
    result = xm1 * (xm1 / D_(0));                                           \
    for (int n = 1; n < N; ++n) {                                           \
      x = b(n);                                                             \
      for (int j = 0; j < J; ++j) {                                         \
        T value = phi_(j, n-1) * (f(j) + X_(j, n-1) * xm1);                 \
        f(j) = value;                                                       \
        x -= u_(j, n-1) * value;                                            \
      }                                                                     \
      xm1 = x;                                                              \
      result += x * x / D_(n);                                              \
    }                                                                       \

    if (SIZE == Eigen::Dynamic) {
      switch (J) {
        case 1:  { FIXED_SIZE_HACKZ(1)  break; }
        case 2:  { FIXED_SIZE_HACKZ(2)  break; }
        case 3:  { FIXED_SIZE_HACKZ(3)  break; }
        case 4:  { FIXED_SIZE_HACKZ(4)  break; }
        case 5:  { FIXED_SIZE_HACKZ(5)  break; }
        case 6:  { FIXED_SIZE_HACKZ(6)  break; }
        case 7:  { FIXED_SIZE_HACKZ(7)  break; }
        case 8:  { FIXED_SIZE_HACKZ(8)  break; }
        case 9:  { FIXED_SIZE_HACKZ(9)  break; }
        case 10: { FIXED_SIZE_HACKZ(10) break; }
        case 11: { FIXED_SIZE_HACKZ(11) break; }
        case 12: { FIXED_SIZE_HACKZ(12) break; }
        case 13: { FIXED_SIZE_HACKZ(13) break; }
        case 14: { FIXED_SIZE_HACKZ(14) break; }
        case 15: { FIXED_SIZE_HACKZ(15) break; }
        case 16: { FIXED_SIZE_HACKZ(16) break; }
        default: FIXED_SIZE_HACKZ(Eigen::Dynamic)
      }
    } else {
      // The size was already specified at compile time.
      FIXED_SIZE_HACKZ(SIZE)
    }

#undef FIXED_SIZE_HACKZ

  } else {

    vector_j_t f(J);
    f.setZero();
    T x = T(b(0));
    result = x * (x / D_(0));
    for (int n = 1; n < N; ++n) {
      f = phi_.col(n-1).asDiagonal() * (f + X_.col(n-1) * x);
      x = b(n) - u_.col(n-1).transpose().dot(f);
      result += x * x / D_(n);
    }

  }

  return result;
};

/// Compute the dot product of the square root of a celerite matrix
///
/// This method computes ``L.z`` where ``A = L.L^T`` is the matrix defined in
/// ``compute``.
///
/// @param z The matrix z from above.
matrix_t dot_L (const Eigen::MatrixXd& z) const {
  if (z.rows() != this->N_) throw dimension_mismatch();
  if (!(this->computed_)) throw compute_exception();

  T tmp;
  int N = z.rows(), nrhs = z.cols();
  Eigen::Array<T, Eigen::Dynamic, 1> D = sqrt(D_.array());
  vector_j_t f(J_);
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

/// Compute the dot product of a celerite matrix with another matrix
///
/// @param jitter The jitter of the kernel.
/// @param a_real The coefficients of the real terms.
/// @param c_real The exponents of the real terms.
/// @param a_comp The real part of the coefficients of the complex terms.
/// @param b_comp The imaginary part of the of the complex terms.
/// @param c_comp The real part of the exponents of the complex terms.
/// @param d_comp The imaginary part of the exponents of the complex terms.
/// @param x The _sorted_ array of input coordinates.
/// @param z The matrix that will be dotted with the celerite matrix.
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
  if (SIZE != Eigen::Dynamic && J != SIZE) throw dimension_mismatch();
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

  vector_j_t f(J);
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

/// Compute the conditional mean for a GP in O(N)
///
/// This method computes ``K_*.(K+sigma)^{-1}.y`` where ``K`` is the matrix
/// defined in ``compute``.
///
/// @param y The data vector to condition on
/// @param x The input coordinates to compute the predictions at
vector_t predict (const Eigen::VectorXd& y, const Eigen::VectorXd& x) const {
  if (y.rows() != this->N_) throw dimension_mismatch();
  if (!(this->computed_)) throw compute_exception();

  T tmp, cd, sd, pm, alphan;
  double tref, dt, tn, xm;
  int m, n, j, k, N = y.rows(), M = x.rows(), J = J_,
      J_real = a_real_.rows(), J_comp = a_comp_.rows();

  vector_t alpha = this->solve(y),
           pred(M);
  pred.setZero();
  Eigen::Array<T, SIZE, 1> Q(J);
  Q.setZero();

  // Forward pass
  m = 0;
  while (m < M && x(m) <= t_(0)) ++m;
  for (n = 0; n < N; ++n) {
    alphan = alpha(n);
    if (n < N-1) tref = t_(n+1);
    else tref = t_(N-1);

    tn = t_(n);
    dt = tref - tn;
    for (j = 0; j < J_real; ++j) {
      Q(j) += alphan;
      Q(j) *= exp(-c_real_(j)*dt);
    }
    for (j = 0, k = J_real; j < J_comp; ++j, k+=2) {
      tmp = exp(-c_comp_(j)*dt);
      Q(k)   += alphan * cos(d_comp_(j)*tn);
      Q(k)   *= tmp;
      Q(k+1) += alphan * sin(d_comp_(j)*tn);
      Q(k+1) *= tmp;
    }

    while (m < M && (n == N-1 || x(m) <= tref)) {
      xm = x(m);
      dt = xm - tref;
      pm = T(0.0);
      for (j = 0; j < J_real; ++j) {
        pm += a_real_(j)*exp(-c_real_(j)*dt) * Q(j);
      }
      for (j = 0, k = J_real; j < J_comp; ++j, k+=2) {
        cd = cos(d_comp_(j)*xm);
        sd = sin(d_comp_(j)*xm);
        tmp = exp(-c_comp_(j)*dt);
        pm += (a_comp_(j)*cd + b_comp_(j)*sd)*tmp * Q(k);
        pm += (a_comp_(j)*sd - b_comp_(j)*cd)*tmp * Q(k+1);
      }
      pred(m) = pm;
      ++m;
    }
  }

  // Backward pass
  m = M - 1;
  while (m >= 0 && x(m) > t_(N-1)) --m;
  Q.setZero();
  for (n = N-1; n >= 0; --n) {
    alphan = alpha(n);
    if (n > 0) tref = t_(n-1);
    else tref = t_(0);
    tn = t_(n);
    dt = tn - tref;

    for (j = 0; j < J_real; ++j) {
      Q(j) += alphan*a_real_(j);
      Q(j) *= exp(-c_real_(j)*dt);
    }
    for (j = 0, k = J_real; j < J_comp; ++j, k+=2) {
      cd = cos(d_comp_(j)*tn);
      sd = sin(d_comp_(j)*tn);
      tmp = exp(-c_comp_(j)*dt);
      Q(k)   += alphan*(a_comp_(j)*cd + b_comp_(j)*sd);
      Q(k)   *= tmp;
      Q(k+1) += alphan*(a_comp_(j)*sd - b_comp_(j)*cd);
      Q(k+1) *= tmp;
    }

    while (m >= 0 && (n == 0 || x(m) > tref)) {
      xm = x(m);
      dt = tref-xm;
      pm = T(0.0);
      for (j = 0; j < J_real; ++j) {
        pm += exp(-c_real_(j)*dt) * Q(j);
      }
      for (j = 0, k = J_real; j < J_comp; ++j, k+=2) {
        tmp = exp(-c_comp_(j)*dt);
        pm += cos(d_comp_(j)*xm)*tmp * Q(k);
        pm += sin(d_comp_(j)*xm)*tmp * Q(k+1);
      }
      pred(m) += pm;
      --m;
    }
  }

  return pred;
};

using Solver<T>::compute;

protected:
int J_;
matrix_j_t u_, phi_, X_;
vector_t D_, a_real_, c_real_, a_comp_, b_comp_, c_comp_, d_comp_;
Eigen::VectorXd t_;

};

};
};

#endif
