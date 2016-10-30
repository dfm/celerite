#include <iostream>
#include <sys/time.h>
#include <Eigen/Core>

#include "genrp/solvers/basic.h"
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
  Eigen::VectorXd alpha = Eigen::VectorXd::Random(nterms),
                  beta_real = Eigen::VectorXd::Random(nterms),
                  alpha_all(3*nterms);
  Eigen::VectorXcd beta_complex = Eigen::VectorXcd::Random(nterms),
                   beta_all(3*nterms);
  alpha.array() += 1.0;
  beta_real.array() += 1.0;
  beta_complex.array() += std::complex<double>(1.0, 1.0);
  alpha_all << alpha, 0.5 * alpha.array(), 0.5 * alpha.array();
  beta_all << beta_real.cast<std::complex<double> >(), beta_complex, beta_complex.conjugate();

  // Generate some fake data.
  Eigen::VectorXd x = Eigen::VectorXd::Random(N),
                  yerr2 = Eigen::VectorXd::Random(N),
                  y;

  // Set the scale of the uncertainties.
  yerr2.array() *= 0.1;
  yerr2.array() += 1.0;

  // The times need to be sorted.
  std::sort(x.data(), x.data() + x.size());

  // Compute the y values.
  y = sin(x.array());

  genrp::BasicSolver<double> basic_real(alpha, beta_real);
  basic_real.compute(x, yerr2);
  genrp::BasicSolver<std::complex<double> > basic_complex(alpha_all, beta_all);
  basic_complex.compute(x, yerr2);

  genrp::DirectSolver direct_real(alpha, beta_real);
  direct_real.compute(x, yerr2);
  genrp::DirectSolver direct_complex(alpha, beta_real, alpha, beta_complex);
  direct_complex.compute(x, yerr2);

  genrp::BandSolver band_real(alpha, beta_real);
  band_real.compute(x, yerr2);
  genrp::BandSolver band_complex(alpha, beta_real, alpha, beta_complex);
  band_complex.compute(x, yerr2);

  DO_TEST(direct_real_dot_solve, basic_real.dot_solve(y), direct_real.dot_solve(y))
  DO_TEST(direct_real_log_det, basic_real.log_determinant(), direct_real.log_determinant())
  DO_TEST(band_real_dot_solve, direct_real.dot_solve(y), band_real.dot_solve(y))
  DO_TEST(band_real_log_det, basic_real.log_determinant(), band_real.log_determinant())

  DO_TEST(direct_complex_dot_solve, basic_complex.dot_solve(y), direct_complex.dot_solve(y))
  DO_TEST(direct_complex_log_det, basic_complex.log_determinant(), direct_complex.log_determinant())
  DO_TEST(band_complex_dot_solve, direct_complex.dot_solve(y), band_complex.dot_solve(y))
  DO_TEST(band_complex_log_det, basic_complex.log_determinant(), band_complex.log_determinant())

  return 0;
}
