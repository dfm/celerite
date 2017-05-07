#ifndef _CELERITE_UTILS_H_
#define _CELERITE_UTILS_H_

#include <cmath>
#include <vector>
#include <Eigen/Core>

#include "celerite/poly.h"

#ifndef M_PI
#define M_PI 3.141592653589793238462643383279502884e+00
#endif

namespace celerite {

template <typename T1, typename T2>
inline bool isclose (const T1& a, const T2& b) {
  using std::abs;
  return (abs(a - b) <= 1e-6);
}

template <typename T>
inline T _logsumexp (const T& a, const T& b) {
  return b + log(T(1.0) + exp(a - b));
}

template <typename Derived>
inline bool check_coefficients (
  const Eigen::DenseBase<Derived>& alpha_real,
  const Eigen::DenseBase<Derived>& beta_real,
  const Eigen::DenseBase<Derived>& alpha_complex_real,
  const Eigen::DenseBase<Derived>& alpha_complex_imag,
  const Eigen::DenseBase<Derived>& beta_complex_real,
  const Eigen::DenseBase<Derived>& beta_complex_imag
) {
  using std::abs;
  typedef typename Derived::Scalar T;
  typedef Eigen::Matrix<T, Eigen::Dynamic, 1> vector_t;

  if (alpha_real.rows() != beta_real.rows()) return false;
  if (alpha_complex_real.rows() != alpha_complex_imag.rows()) return false;
  if (alpha_complex_real.rows() != beta_complex_real.rows()) return false;
  if (alpha_complex_real.rows() != beta_complex_imag.rows()) return false;

  // Start by building the polynomials for each term.
  int n = alpha_real.rows() + alpha_complex_real.rows();
  std::vector<vector_t> num(n), denom(n);
  T a, b, c, d, c2, d2, w0;
  int k = 0;
  for (int i = 0; i < alpha_real.rows(); ++i, ++k) {
    a = alpha_real[i];
    c = beta_real[i];
    c2 = c*c;

    num[k].resize(2);
    num[k][0] = a*c;
    num[k][1] = a*c*c2;

    denom[k].resize(3);
    denom[k][0] = T(1.0);
    denom[k][1] = 2.0*c2;
    denom[k][2] = c2*c2;
  }

  for (int i = 0; i < alpha_complex_real.rows(); ++i, ++k) {
    a = alpha_complex_real[i];
    b = alpha_complex_imag[i];
    c = beta_complex_real[i];
    d = beta_complex_imag[i];
    c2 = c*c;
    d2 = d*d;
    w0 = c2 + d2;

    num[k].resize(2);
    num[k][0] = a*c - b*d;
    num[k][1] = (a*c + b*d) * w0;

    denom[k].resize(3);
    denom[k][0] = T(1.0);
    denom[k][1] = 2.0*(c2 - d2);
    denom[k][2] = w0 * w0;
  }

  // Compute the full numerator.
  vector_t poly(1), tmp;
  poly.setConstant(T(0.0));
  for (int i = 0; i < n; ++i) {
    tmp = num[i];
    for (int j = 0; j < n; ++j) {
      if (i != j) tmp = polymul(tmp, denom[j]);
    }
    poly = polyadd(poly, tmp);
  }

  // Deal with over/underflow.
  while (poly.rows() > 1 && abs(poly[0]) < POLYTOL)
    poly = poly.tail(poly.rows() - 1);

  if (polyval(poly, 0.0) < 0.0) return false;

  // Count the number of roots.
  int nroots = polycountroots(poly);
  return (nroots == 0);
}

template <typename Derived>
typename Derived::Scalar
inline get_kernel_value (
  const Eigen::DenseBase<Derived>& alpha_real,
  const Eigen::DenseBase<Derived>& beta_real,
  const Eigen::DenseBase<Derived>& alpha_complex_real,
  const Eigen::DenseBase<Derived>& alpha_complex_imag,
  const Eigen::DenseBase<Derived>& beta_complex_real,
  const Eigen::DenseBase<Derived>& beta_complex_imag,
  typename Derived::Scalar tau
) {
  using std::abs;
  typedef typename Derived::Scalar T;

  T t = abs(tau), k = T(0.0);

  for (int i = 0; i < alpha_real.rows(); ++i)
    k += alpha_real[i] * exp(-beta_real[i] * t);

  for (int i = 0; i < alpha_complex_real.rows(); ++i)
    k += exp(-beta_complex_real[i] * t) * (
      alpha_complex_real[i] * cos(beta_complex_imag[i] * t) +
      alpha_complex_imag[i] * sin(beta_complex_imag[i] * t)
    );

  return k;
}

template <typename Derived>
typename Derived::Scalar
inline get_psd_value (
  const Eigen::DenseBase<Derived>& alpha_real,
  const Eigen::DenseBase<Derived>& beta_real,
  const Eigen::DenseBase<Derived>& alpha_complex_real,
  const Eigen::DenseBase<Derived>& alpha_complex_imag,
  const Eigen::DenseBase<Derived>& beta_complex_real,
  const Eigen::DenseBase<Derived>& beta_complex_imag,
  typename Derived::Scalar& omega
) {
  typedef typename Derived::Scalar T;
  T w2 = omega * omega, p = T(0.0), a, b, c, d, w02;
  for (int i = 0; i < alpha_real.rows(); ++i) {
    a = alpha_real[i];
    c = beta_real[i];
    p += a*c / (c*c + w2);
  }

  for (int i = 0; i < alpha_complex_real.rows(); ++i) {
    a = alpha_complex_real[i];
    b = alpha_complex_imag[i];
    c = beta_complex_real[i];
    d = beta_complex_imag[i];
    w02 = c*c+d*d;
    p += ((a*c+b*d)*w02+(a*c-b*d)*w2) / (w2*w2 + 2.0*(c*c-d*d)*w2+w02*w02);
  }

  return sqrt(2.0 / M_PI) * p;
}

};

#endif
