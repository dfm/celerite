#include <iostream>
#include <Eigen/Core>

#include "genrp/solvers/direct.h"
#include "genrp/solvers/band.h"

#define DO_TEST(NAME, VAR1, VAR2)                            \
{                                                            \
  double base, comp, delta;                                  \
  base = VAR1;                                               \
  comp = VAR2;                                               \
  delta = std::abs(base - comp);                             \
  if (delta > 1e-10) {                                       \
    std::cerr << "Test failed: '" << #NAME << "' - error: " << delta << std::endl; \
    return 1;                                                \
  } else                                                     \
    std::cerr << "Test passed: '" << #NAME << "' - error: " << delta << std::endl; \
}

int main (int argc, char* argv[])
{
  srand(42);

  size_t nterms = 3;
  if (argc >= 2) nterms = atoi(argv[1]);
  size_t N = 1024;
  if (argc >= 3) N = atoi(argv[2]);
  size_t niter = 10;
  if (argc >= 4) niter = atoi(argv[3]);

  // Set up the coefficients.
  Eigen::VectorXd alpha_real = Eigen::VectorXd::Random(nterms + 1),
                  alpha_complex = Eigen::VectorXd::Random(nterms),
                  beta_real = Eigen::VectorXd::Random(nterms + 1),
                  beta_complex_real = Eigen::VectorXd::Random(nterms),
                  beta_complex_imag = Eigen::VectorXd::Random(nterms);
  alpha_real.array() += 1.0;
  alpha_complex.array() += 1.0;
  beta_real.array() += 1.0;
  beta_complex_real.array() += 1.0;
  beta_complex_imag.array() += 1.0;

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

  genrp::DirectSolver<double> direct;
  genrp::BandSolver<double> band;

  direct.compute(alpha_real, beta_real, x, yerr2);
  band.compute(alpha_real, beta_real, x, yerr2);
  DO_TEST(band_real_log_det, direct.log_determinant(), band.log_determinant())
  DO_TEST(band_real_dot_solve, direct.dot_solve(y), band.dot_solve(y))

  band.compute(alpha_complex, beta_complex_real, beta_complex_imag, x, yerr2);
  direct.compute(alpha_complex, beta_complex_real, beta_complex_imag, x, yerr2);
  DO_TEST(band_complex_dot_solve, direct.dot_solve(y), band.dot_solve(y))
  DO_TEST(band_complex_log_det, direct.log_determinant(), band.log_determinant())

  band.compute(alpha_real, beta_real, alpha_complex, beta_complex_real, beta_complex_imag, x, yerr2);
  direct.compute(alpha_real, beta_real, alpha_complex, beta_complex_real, beta_complex_imag, x, yerr2);
  DO_TEST(band_mixed_dot_solve, direct.dot_solve(y), band.dot_solve(y))
  DO_TEST(band_mixed_log_det, direct.log_determinant(), band.log_determinant())

  return 0;
}
