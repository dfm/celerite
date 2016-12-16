#ifndef _GENRP_UTILS_H_
#define _GENRP_UTILS_H_

#include <vector>
#include <Eigen/Core>

#include "genrp/poly.h"

namespace genrp {

template <typename T>
bool check_coefficients (
  const Eigen::Matrix<T, Eigen::Dynamic, 1>& alpha_real,
  const Eigen::Matrix<T, Eigen::Dynamic, 1>& beta_real,
  const Eigen::Matrix<T, Eigen::Dynamic, 1>& alpha_complex_real,
  const Eigen::Matrix<T, Eigen::Dynamic, 1>& alpha_complex_imag,
  const Eigen::Matrix<T, Eigen::Dynamic, 1>& beta_complex_real,
  const Eigen::Matrix<T, Eigen::Dynamic, 1>& beta_complex_imag
) {
  typedef Eigen::Matrix<T, Eigen::Dynamic, 1> vector_t;

  if (alpha_real.rows() != beta_real.rows()) return false;
  if (alpha_complex_real.rows() != alpha_complex_imag.rows()) return false;
  if (alpha_complex_real.rows() != beta_complex_real.rows()) return false;
  if (alpha_complex_real.rows() != beta_complex_imag.rows()) return false;

  // Start by building the polynomials for each term.
  int n = alpha_real.rows() + alpha_complex_real.rows();
  std::vector<vector_t> num(n), denom(n);
  T ar, br, ai, bi;
  int k = 0;
  for (int i = 0; i < alpha_real.rows(); ++i, ++k) {
    ar = alpha_real[i];
    br = beta_real[i];

    num[k].resize(2);
    num[k][0] = ar*br;
    num[k][1] = ar*br*br*br;

    denom[k].resize(3);
    denom[k][0] = T(1.0);
    denom[k][1] = 2.0*br*br;
    denom[k][2] = br*br*br*br;
  }

  for (int i = 0; i < alpha_complex_real.rows(); ++i, ++k) {
    ar = alpha_complex_real[i];
    br = beta_complex_real[i];
    ai = alpha_complex_imag[i];
    bi = beta_complex_imag[i];

    num[k].resize(2);
    num[k][0] = ar*br - ai*bi;
    num[k][1] = (ar*br + ai*bi) * (br*br + bi*bi);

    denom[k].resize(3);
    denom[k][0] = T(1.0);
    denom[k][1] = 2.0*(br*br - bi*bi);
    denom[k][2] = br*br + bi*bi;
    denom[k][2] *= denom[k][2];
  }

  // Compute the full numerator.
  vector_t poly = vector_t::Zero(1), tmp;
  for (int i = 0; i < n; ++i) {
    tmp = num[i];
    for (int j = 0; j < n; ++j) {
      if (i != j) tmp = polymul(tmp, denom[j]);
    }
    poly = polyadd(poly, tmp);
  }

  if (polyval(poly, 0.0) < 0.0) return false;

  // Count the number of roots.
  int nroots = polycountroots(poly);
  return (nroots == 0);
}

template <typename T>
T get_kernel_value (
  const Eigen::Matrix<T, Eigen::Dynamic, 1>& alpha_real,
  const Eigen::Matrix<T, Eigen::Dynamic, 1>& beta_real,
  const Eigen::Matrix<T, Eigen::Dynamic, 1>& alpha_complex_real,
  const Eigen::Matrix<T, Eigen::Dynamic, 1>& alpha_complex_imag,
  const Eigen::Matrix<T, Eigen::Dynamic, 1>& beta_complex_real,
  const Eigen::Matrix<T, Eigen::Dynamic, 1>& beta_complex_imag,
  T tau
) {
  using std::abs;

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

template <typename T>
T get_psd_value (
  const Eigen::Matrix<T, Eigen::Dynamic, 1>& alpha_real,
  const Eigen::Matrix<T, Eigen::Dynamic, 1>& beta_real,
  const Eigen::Matrix<T, Eigen::Dynamic, 1>& alpha_complex_real,
  const Eigen::Matrix<T, Eigen::Dynamic, 1>& alpha_complex_imag,
  const Eigen::Matrix<T, Eigen::Dynamic, 1>& beta_complex_real,
  const Eigen::Matrix<T, Eigen::Dynamic, 1>& beta_complex_imag,
  T omega
) {
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
