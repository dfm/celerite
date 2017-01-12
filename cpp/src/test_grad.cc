#include <iostream>
#include <Eigen/Core>
#include <unsupported/Eigen/AutoDiff>
#include "celerite/solvers/band.h"

#define DO_TEST(FUNC, VAR1, VAR2)                            \
{                                                            \
  int flag = 1;                                              \
  double value, delta, eps0 = eps;                                       \
  while (flag != 0) {                                        \
    VAR1 += eps0;                                               \
    flag = solver.compute(alpha_real, beta_real,                      \
        alpha_complex, beta_complex_real, beta_complex_imag,   \
        x, yerr2);                                             \
    value = FUNC;                                              \
    VAR1 -= 2.0*eps0;                                           \
    flag += solver.compute(alpha_real, beta_real,                      \
        alpha_complex, beta_complex_real, beta_complex_imag,   \
        x, yerr2);                                             \
    value -= FUNC;                                             \
    VAR1 += eps0;                                               \
    value /= 2.0 * eps0;                                        \
    eps0 /= 2.0;                                               \
  }                                                              \
  delta = std::abs(value - VAR2);                            \
  if (delta > 7e-5) {                                        \
    std::cerr << "Test failed: '" << #FUNC << ", " << #VAR1 << "': |" << value << " - " << VAR2 << "| = " << delta << std::endl; \
    return 1;                                                \
  } else                                                     \
    std::cerr << "Test passed: '" << #FUNC << " " << #VAR1 << "' - error: " << delta << std::endl; \
}

int main (int argc, char* argv[])
{
  typedef Eigen::AutoDiffScalar<Eigen::VectorXd> ad_t;

  srand(42);

  size_t nterms = 3;
  if (argc >= 2) nterms = atoi(argv[1]);
  size_t N = 1024;
  if (argc >= 3) N = atoi(argv[2]);
  size_t niter = 10;
  if (argc >= 4) niter = atoi(argv[3]);

  // Set up the coefficients.
  Eigen::VectorXd alpha_real = Eigen::VectorXd::Random(nterms),
                  beta_real = Eigen::VectorXd::Random(nterms),
                  alpha_complex = Eigen::VectorXd::Random(nterms),
                  beta_complex_real = Eigen::VectorXd::Random(nterms),
                  beta_complex_imag = Eigen::VectorXd::Random(nterms);
  alpha_real.array() += 1.;
  alpha_complex.array() += 1.;
  beta_real.array() += 1.;
  beta_complex_real.array() += 1.;
  beta_complex_imag.array() += 1.;

  // Generate some fake data.
  Eigen::VectorXd x = Eigen::VectorXd::Random(N),
                  yerr2 = Eigen::VectorXd::Random(N),
                  y;

  // Set the scale of the uncertainties.
  yerr2.array() *= 0.1;
  yerr2.array() += 2.5;

  // The times need to be sorted.
  std::sort(x.data(), x.data() + x.size());

  // Compute the y values.
  y = sin(x.array());

  // Set up the gradients
  size_t nparams = 5 * nterms + 1;
  Eigen::Matrix<ad_t, Eigen::Dynamic, 1> alpha_real_grad(nterms),
                                         beta_real_grad(nterms),
                                         alpha_complex_grad(nterms),
                                         beta_complex_real_grad(nterms),
                                         beta_complex_imag_grad(nterms),
                                         yerr2_grad(N);
  for (size_t i = 0; i < N; ++i)
    yerr2_grad(i) = ad_t(yerr2(i), nparams, 0);

  size_t par = 1;
  for (size_t i = 0; i < nterms; ++i) {
    alpha_real_grad(i) = ad_t(alpha_real(i), nparams, par++);
    beta_real_grad(i) = ad_t(beta_real(i), nparams, par++);
    alpha_complex_grad(i) = ad_t(alpha_complex(i), nparams, par++);
    beta_complex_real_grad(i) = ad_t(beta_complex_real(i), nparams, par++);
    beta_complex_imag_grad(i) = ad_t(beta_complex_imag(i), nparams, par++);
  }

  celerite::BandSolver<double> solver;
  celerite::BandSolver<ad_t> grad_solver;
  grad_solver.compute(alpha_real_grad, beta_real_grad,
      alpha_complex_grad, beta_complex_real_grad, beta_complex_imag_grad,
      x, yerr2_grad);

  ad_t grad_log_det = grad_solver.log_determinant(),
       grad_dot_solve = grad_solver.dot_solve(y);

  double eps = 1.23e-6;
  DO_TEST(solver.log_determinant(), yerr2.array(), grad_log_det.derivatives()(0))
  DO_TEST(solver.dot_solve(y), yerr2.array(), grad_dot_solve.derivatives()(0))

  par = 1;
  for (size_t i = 0; i < nterms; ++i) {
    DO_TEST(solver.log_determinant(), alpha_real(i), grad_log_det.derivatives()(par))
    DO_TEST(solver.dot_solve(y), alpha_real(i), grad_dot_solve.derivatives()(par++))

    DO_TEST(solver.log_determinant(), beta_real(i), grad_log_det.derivatives()(par))
    DO_TEST(solver.dot_solve(y), beta_real(i), grad_dot_solve.derivatives()(par++))

    DO_TEST(solver.log_determinant(), alpha_complex(i), grad_log_det.derivatives()(par))
    DO_TEST(solver.dot_solve(y), alpha_complex(i), grad_dot_solve.derivatives()(par++))

    DO_TEST(solver.log_determinant(), beta_complex_real(i), grad_log_det.derivatives()(par))
    DO_TEST(solver.dot_solve(y), beta_complex_real(i), grad_dot_solve.derivatives()(par++))

    DO_TEST(solver.log_determinant(), beta_complex_imag(i), grad_log_det.derivatives()(par))
    DO_TEST(solver.dot_solve(y), beta_complex_imag(i), grad_dot_solve.derivatives()(par++))
  }

  return 0;
}
