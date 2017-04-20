#include <iostream>
#include <vector>

#include <Eigen/Core>
#include <stan/math.hpp>

#include "celerite/solver/cholesky.h"


#define DO_TEST(NAME, VAR1, VAR2)                            \
{                                                            \
  double base, comp, delta;                                  \
  base = VAR1;                                               \
  comp = VAR2;                                               \
  delta = std::abs(base - comp);                             \
  if (delta > 1e-10) {                                       \
    std::cerr << "Test failed: '" << #NAME << "' - error: " << delta << " " << base << " " << comp << std::endl; \
    return 1;                                                \
  } else                                                     \
    std::cerr << "Test passed: '" << #NAME << "' - error: " << delta << std::endl; \
}

int main (int argc, char* argv[])
{
  typedef stan::math::var g_t;
  typedef Eigen::Matrix<g_t, Eigen::Dynamic, 1> v_t;
  typedef Eigen::Matrix<g_t, Eigen::Dynamic, Eigen::Dynamic> m_t;

  srand(42);

  size_t N = 1024;
  if (argc >= 2) N = atoi(argv[1]);
  size_t niter = 10;
  if (argc >= 3) niter = atoi(argv[2]);

  // Set up the coefficients.
  size_t p_real = 2, p_complex = 1;
  g_t jitter = g_t(0.01);
  v_t a_real(p_real), c_real(p_real),
      a_comp(p_complex), b_comp(p_complex), c_comp(p_complex), d_comp(p_complex);

  a_real << 1.3, 1.5;
  c_real << 0.5, 0.2;
  a_comp << 1.0;
  b_comp << 0.1;
  c_comp << 1.0;
  d_comp << 1.0;

  // Generate some fake data.
  Eigen::VectorXd x = Eigen::VectorXd::Random(N),
                  yerr2 = Eigen::VectorXd::Random(N),
                  y;

  // Set the scale of the uncertainties.
  yerr2.array() *= 0.1;
  yerr2.array() += 0.3;

  // The times need to be sorted.
  std::sort(x.data(), x.data() + x.size());

  // Compute the y values.
  y = sin(x.array());

  celerite::solver::CholeskySolver<g_t> cholesky;
  cholesky.compute(jitter, a_real, c_real, a_comp, b_comp, c_comp, d_comp, x, yerr2);

  std::vector<g_t> params;
  for (int i = 0; i < p_real; ++i) {
    params.push_back(a_real(i));
    params.push_back(c_real(i));
  }
  for (int i = 0; i < p_complex; ++i) {
    params.push_back(a_comp(i));
    params.push_back(b_comp(i));
    params.push_back(c_comp(i));
    params.push_back(d_comp(i));
  }
  std::vector<double> g;

  g_t ll = -0.5 * (cholesky.dot_solve(y) + cholesky.log_determinant());
  ll.grad(params, g);
  for (int i = 0; i < g.size(); ++i) std::cout << g[i] << std::endl;

  return 0;
}
