#include <iostream>
#include <Eigen/Core>

#include "genrp/genrp.h"

#define DO_TEST(NAME, VAR1, VAR2)                            \
{                                                            \
  double base, comp, delta;                                  \
  base = VAR1;                                               \
  comp = VAR2;                                               \
  delta = std::abs(base - comp);                             \
  if (delta > 1e-8) {                                       \
    std::cerr << "Test failed: '" << #NAME << "' - error: " << delta << std::endl; \
    return 1;                                                \
  } else                                                     \
    std::cerr << "Test passed: '" << #NAME << "' - error: " << delta << std::endl; \
}

double randu (double mn=0.0, double mx=1.0) {
  return mn + ((double) rand() / (RAND_MAX)) * (mx - mn);
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

  genrp::Kernel real_kernel, complex_kernel;
  real_kernel.add_term(randu(-1.0, 1.0), randu(-1.0, 1.0));
  complex_kernel.add_term(randu(-1.0, 1.0), randu(-1.0, 1.0));

  for (size_t i = 1; i < nterms; ++i) {
    real_kernel.add_term(randu(-1.0, 1.0), randu(-1.0, 1.0));
    complex_kernel.add_term(randu(-1.0, 1.0), randu(-1.0, 1.0), randu(-1.0, 1.0));
  }

  genrp::GaussianProcess<genrp::DirectSolver<double> > direct_real(real_kernel),
                                                       direct_complex(complex_kernel);
  genrp::GaussianProcess<genrp::BandSolver<double> > band_real(real_kernel),
                                                     band_complex(complex_kernel);

  // Generate some fake data.
  Eigen::VectorXd x = Eigen::VectorXd::Random(N),
                  yerr = Eigen::VectorXd::Random(N),
                  y;

  // Set the scale of the uncertainties.
  yerr.array() *= 0.1;
  yerr.array() += 4.0;

  // The times need to be sorted.
  std::sort(x.data(), x.data() + x.size());

  // Compute the y values.
  y = sin(x.array());

  direct_real.compute(x, yerr);
  band_real.compute(x, yerr);
  direct_complex.compute(x, yerr);
  band_complex.compute(x, yerr);

  DO_TEST(real_log_likelihood, direct_real.log_likelihood(y), band_real.log_likelihood(y))
  DO_TEST(complex_log_likelihood, direct_complex.log_likelihood(y), band_complex.log_likelihood(y))

  return 0;
}
