#ifndef _CELERITE_POLY_H_
#define _CELERITE_POLY_H_

#include <cmath>
#include <cfloat>
#include <vector>
#include <Eigen/Core>

namespace celerite {

#define POLYTOL 1e-10

template <typename T>
inline T polyval (const Eigen::Matrix<T, Eigen::Dynamic, 1>& p, const double x) {
  T result = T(0.0);
  for (int i = 0; i < p.rows(); ++i) result = result * x + p[i];
  return result;
}

template <typename T>
inline Eigen::Matrix<T, Eigen::Dynamic, 1> polyadd (const Eigen::Matrix<T, Eigen::Dynamic, 1>& p1, const Eigen::Matrix<T, Eigen::Dynamic, 1>& p2) {
  int n1 = p1.rows(),
      n2 = p2.rows(),
      n = std::max(p1.rows(), p2.rows());
  Eigen::Matrix<T, Eigen::Dynamic, 1> result(n);
  result.setConstant(0.0);
  for (int i = 0; i < n; ++i) {
    if (i < n1) result[n - i - 1] += p1[n1 - i - 1];
    if (i < n2) result[n - i - 1] += p2[n2 - i - 1];
  }
  return result;
}

template <typename T>
inline Eigen::Matrix<T, Eigen::Dynamic, 1> polymul (const Eigen::Matrix<T, Eigen::Dynamic, 1>& p1, const Eigen::Matrix<T, Eigen::Dynamic, 1>& p2) {
  int n1 = p1.rows(),
      n2 = p2.rows(),
      n = n1 + n2 - 1;
  Eigen::Matrix<T, Eigen::Dynamic, 1> result(n);
  result.setConstant(0.0);
  for (int i = 0; i < n; ++i)
    for (int j = 0; j < n2; ++j)
      if (i - j >= 0 && i - j < n1 && j >= 0)
        result[i] += p1[i - j] * p2[j];
  return result;
}

template <typename T>
inline Eigen::Matrix<T, Eigen::Dynamic, 1> polyrem (const Eigen::Matrix<T, Eigen::Dynamic, 1>& u, const Eigen::Matrix<T, Eigen::Dynamic, 1>& v) {
  int m = u.rows() - 1,
      n = v.rows() - 1,
      p = m - n + 1;
  using std::abs;
  T d, scale = T(1.0) / v[0];
  Eigen::Matrix<T, Eigen::Dynamic, 1> q(std::max(p, 1)), r = u; // This makes a copy!
  q.setConstant(T(0.0));
  for (int k = 0; k < p; ++k) {
    d = scale * r[k];
    q[k] = d;
    for (int i = 0; i < n+1; ++i) r[k+i] -= d*v[i];
  }
  int strt;
  for (strt = 0; strt < m; ++strt) {
    if (abs(r[strt]) >= T(POLYTOL)) {
      return r.tail(m + 1 - strt);
    }
  }
  return r.tail(1);
}

template <typename T>
inline Eigen::Matrix<T, Eigen::Dynamic, 1> polyder (const Eigen::Matrix<T, Eigen::Dynamic, 1>& p) {
  int n = p.rows() - 1;
  Eigen::Matrix<T, Eigen::Dynamic, 1> d = p;  // Copy.
  for (int i = 0; i < n; ++i) {
    d[i] *= n - i;
  }
  return d.head(n);
}

template <typename T>
inline std::vector<Eigen::Matrix<T, Eigen::Dynamic, 1> > polysturm (const Eigen::Matrix<T, Eigen::Dynamic, 1>& p) {
  int n = p.rows() - 1;
  std::vector<Eigen::Matrix<T, Eigen::Dynamic, 1> > sturm;
  Eigen::Matrix<T, Eigen::Dynamic, 1> p0 = p, p1 = polyder(p0), tmp;
  sturm.push_back(p0);
  sturm.push_back(p1);
  for (int k = 0; k < n; ++k) {
    tmp = p1;
    p1 = polyrem(p0, p1);
    p1 *= -1.0;
    p0 = tmp;
    sturm.push_back(p1);
    if (p1.rows() == 1) break;
  }
  return sturm;
}

template <typename T>
inline int sgn(T val) {
    return (T(0) < val) - (val < T(0));
}

// Count the positive roots of a polynomial using Sturm's theorem.
template <typename T>
inline int polycountroots (const Eigen::Matrix<T, Eigen::Dynamic, 1>& p) {
  if (p.rows() <= 1) return 0;

  int n = p.rows() - 1,
      count = 0;

  // Compute the initial signs and count any initial sign change.
  Eigen::Matrix<T, Eigen::Dynamic, 1> p0 = p, p1 = polyder(p0), tmp;
  int s_0 = sgn(p1[p1.rows() - 1]),
      s_inf = sgn(p1[0]),
      s;
  count += (sgn(p0[p0.rows() - 1]) != s_0);
  count -= (sgn(p0[0]) != s_inf);

  // Loop over the Sturm sequence and compute each polynomial.
  for (int k = 0; k < n; ++k) {
    tmp = p1;
    p1 = polyrem(p0, p1);
    p1 *= -1.0;
    p0 = tmp;

    // Count the roots for this next polynomial.
    s = s_0;
    s_0 = sgn(p1[p1.rows() - 1]);
    count += (s != s_0);
    s = s_inf;
    s_inf = sgn(p1[0]);
    count -= (s != s_inf);

    if (p1.rows() == 1) break;
  }
  return count;
}

};

#endif
