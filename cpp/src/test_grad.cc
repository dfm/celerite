#include <iostream>
#include <Eigen/Core>
#include <unsupported/Eigen/AutoDiff>

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
  typedef Eigen::AutoDiffScalar<Eigen::VectorXd> g_t;
  typedef Eigen::Matrix<g_t, Eigen::Dynamic, 1> v_t;
  typedef Eigen::Matrix<g_t, Eigen::Dynamic, Eigen::Dynamic> m_t;

  srand(42);

  size_t N = 1024;
  if (argc >= 2) N = atoi(argv[1]);
  size_t niter = 10;
  if (argc >= 3) niter = atoi(argv[2]);

  // Set up the coefficients.
  size_t p_real = 2, p_complex = 1;
  int nder = 9, j = 1;
  g_t jitter = g_t(0.01, nder, 0);
  v_t a_real(p_real), c_real(p_real),
      a_comp(p_complex), b_comp(p_complex), c_comp(p_complex), d_comp(p_complex);

  a_real << g_t(1.3, nder, j), g_t(1.5, nder, j+1);
  j += 2;
  c_real  << g_t(0.5, nder, j), g_t(0.2, nder, j+1);
  j += 2;
  a_comp << g_t(1.0, nder, j++);
  b_comp << g_t(0.1, nder, j++);
  c_comp << g_t(1.0, nder, j++);
  d_comp << g_t(1.0, nder, j++);

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

  Eigen::MatrixXd U(3, N), V(3, N);
  Eigen::VectorXd A(N);
  for (int n = 0; n < N; ++n) {
    A(n) = 1e-6;
    for (int j = 0; j < 3; ++j) {
      U(j, n) = pow(x(n), j);
      V(j, n) = pow(x(n), j) / (1.0 + j);
      A(n) += U(j, n) * V(j, n);
    }
  }

  celerite::solver::CholeskySolver<g_t> cholesky;
  cholesky.compute(jitter, a_real, c_real, a_comp, b_comp, c_comp, d_comp, x, yerr2);
  std::cout << cholesky.log_determinant().derivatives() << std::endl;
  std::cout << cholesky.dot_solve(y).derivatives() << std::endl;

  cholesky.compute(jitter, a_real, c_real, a_comp, b_comp, c_comp, d_comp, A, U, V, x, yerr2);
  std::cout << cholesky.log_determinant().derivatives() << std::endl;
  std::cout << cholesky.dot_solve(y).derivatives() << std::endl;

  v_t empty(0);
  cholesky.compute(jitter, empty, empty, empty, empty, empty, empty, A, U, V, x, yerr2);
  std::cout << cholesky.log_determinant().derivatives() << std::endl;
  std::cout << cholesky.dot_solve(y).derivatives() << std::endl;

  return 0;
}
