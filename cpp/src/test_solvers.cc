#include <iostream>
#include <Eigen/Core>

#include "celerite/solver/direct.h"
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
  srand(42);

  size_t N = 1024;
  if (argc >= 2) N = atoi(argv[1]);
  size_t niter = 10;
  if (argc >= 3) niter = atoi(argv[2]);

  // Set up the coefficients.
  size_t p_real = 2, p_complex = 2;
  double jitter = 0.01;
  Eigen::VectorXd alpha_real(p_real),
                  alpha_complex_real(p_complex),
                  alpha_complex_imag(p_complex),
                  beta_real(p_real),
                  beta_complex_real(p_complex),
                  beta_complex_imag(p_complex);

  alpha_real << 1.3, 1.5;
  beta_real  << 0.5, 0.2;
  //alpha_complex_real << 1.0;
  //alpha_complex_imag << 0.1;
  //beta_complex_real  << 1.0;
  //beta_complex_imag  << 1.0;
  alpha_complex_real << 1.0, 2.0;
  alpha_complex_imag << 0.1, 0.05;
  beta_complex_real  << 1.0, 0.8;
  beta_complex_imag  << 1.0, 0.1;

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

  celerite::solver::DirectSolver<double> direct;
  celerite::solver::CholeskySolver<double> cholesky;

  cholesky.compute(jitter, alpha_real, beta_real, x, yerr2);
  direct.compute(jitter, alpha_real, beta_real, x, yerr2);
  DO_TEST(cholesky_real_log_det, cholesky.log_determinant(), direct.log_determinant())
  DO_TEST(cholesky_real_dot_solve, direct.dot_solve(y), cholesky.dot_solve(y))

  direct.compute(jitter, alpha_complex_real, alpha_complex_imag, beta_complex_real, beta_complex_imag, x, yerr2);
  cholesky.compute(jitter, alpha_complex_real, alpha_complex_imag, beta_complex_real, beta_complex_imag, x, yerr2);
  DO_TEST(cholesky_complex_log_det, direct.log_determinant(), cholesky.log_determinant())
  DO_TEST(cholesky_complex_dot_solve, cholesky.dot_solve(y), direct.dot_solve(y))

  direct.compute(jitter, alpha_real, beta_real, alpha_complex_real, alpha_complex_imag, beta_complex_real, beta_complex_imag, x, yerr2);
  cholesky.compute(jitter, alpha_real, beta_real, alpha_complex_real, alpha_complex_imag, beta_complex_real, beta_complex_imag, x, yerr2);
  DO_TEST(cholesky_mixed_log_det, direct.log_determinant(), cholesky.log_determinant())
  DO_TEST(cholesky_mixed_dot_solve, cholesky.dot_solve(y), direct.dot_solve(y))

  direct.compute(jitter, alpha_real, beta_real, alpha_complex_real, alpha_complex_imag, beta_complex_real, beta_complex_imag, A, U, V, x, yerr2);
  cholesky.compute(jitter, alpha_real, beta_real, alpha_complex_real, alpha_complex_imag, beta_complex_real, beta_complex_imag, A, U, V, x, yerr2);
  DO_TEST(cholesky_general_log_det, direct.log_determinant(), cholesky.log_determinant())
  DO_TEST(cholesky_general_dot_solve, cholesky.dot_solve(y), direct.dot_solve(y))

  return 0;
}
