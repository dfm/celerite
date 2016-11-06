#include <iostream>
#include <Eigen/Core>
#include <unsupported/Eigen/AutoDiff>

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
  typedef Eigen::AutoDiffScalar<Eigen::VectorXd> ad_t;

  srand(42);

  size_t nterms = 3;
  if (argc >= 2) nterms = atoi(argv[1]);
  size_t N = 1024;
  if (argc >= 3) N = atoi(argv[2]);
  size_t niter = 10;
  if (argc >= 4) niter = atoi(argv[3]);

  size_t npars = 2 + (nterms - 1) * 3;
  genrp::Kernel<ad_t> kernel;
  kernel.add_term(
    ad_t(randu(-1.0, 1.0), npars, 0),
    ad_t(randu(-1.0, 1.0), npars, 1)
  );

  for (size_t i = 1, par = 2; i < nterms; ++i, par += 3)
    kernel.add_term(
      ad_t(randu(-1.0, 1.0), npars, par),
      ad_t(randu(-1.0, 1.0), npars, par + 1),
      ad_t(randu(-1.0, 1.0), npars, par + 2)
    );

  genrp::GaussianProcess<genrp::BandSolver<ad_t>, ad_t> solver(kernel);

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

  solver.compute(x, yerr);
  ad_t ll = solver.log_likelihood(y);
  std::cout << ll.derivatives() << std::endl;

  // DO_TEST(real_log_likelihood, direct_real.log_likelihood(y), band_real.log_likelihood(y))
  // DO_TEST(complex_log_likelihood, direct_complex.log_likelihood(y), band_complex.log_likelihood(y))

  return 0;
}
