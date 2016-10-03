//
// To build this example, run something like:
//
//  g++ -o demo demo.cpp -Iinclude -I/usr/local/include/eigen3 -O3
//  ./demo 1024 10
//
// This will benchmark the solver with 1024 data points by averaging the
// compute time over 10 iterations.
//

#include <iostream>
#include <sys/time.h>
#include <Eigen/Dense>

#include "genrp/genrp.h"
#include "genrp/solvers/band2.h"
#include "genrp/banded.h"

// Timer for the benchmark.
double get_timestamp ()
{
  struct timeval now;
  gettimeofday (&now, NULL);
  return double(now.tv_usec) * 1.0e-6 + double(now.tv_sec);
}

int main (int argc, char* argv[])
{
  /* Eigen::internal::BandMatrix<double> ab(7, 7, 2, 1); */
  /* ab.coeffs().setConstant(0.0); */
  /* ab.diagonal() << 3, 1, 6, 8, 3, 4, 4; */
  /* ab.diagonal(-1) << 1, 5, 5, 9, 2, 6; */
  /* ab.diagonal(1) << 4, 2, 5, 9, 8, 4; */
  /* ab.diagonal(2) << 9, 3, 7, 3, 2; */

  /* std::cout << ab.toDenseMatrix() << std::endl << std::endl; */
  /* std::cout << ab.coeffs().transpose() << std::endl; */

  srand(42);

  size_t N = 1024;
  if (argc >= 2) N = atoi(argv[1]);
  size_t niter = 10;
  if (argc >= 3) niter = atoi(argv[2]);

  // Generate some fake data.
  Eigen::VectorXd x = Eigen::VectorXd::Random(N),
                  yerr = Eigen::VectorXd::Random(N),
                  y;

  // Set the scale of the uncertainties.
  yerr.array() *= 0.1;
  yerr.array() += 0.8;

  // The times need to be sorted.
  std::sort(x.data(), x.data() + x.size());

  // Compute the y values.
  y = sin(x.array());

  // Set up the kernel.
  genrp::Kernel kernel;
  kernel.add_term(1.0, 0.1);
  kernel.add_term(-0.6, 0.7, 1.0);

  // Set up the GP solver.
  genrp::GaussianProcess<genrp::BandSolver<std::complex<double> > > gp_band(kernel);
  gp_band.compute(x, yerr);
  genrp::GaussianProcess<genrp::BandSolver2<std::complex<double> > > gp_band2(kernel);
  gp_band2.compute(x, yerr);
  std::cout << gp_band.log_likelihood(y) << std::endl;
  std::cout << gp_band2.log_likelihood(y) << std::endl;

  // Benchmark the solver.
  double strt, compute_time = 0.0, log_like_time = 0.0, log_like;
  for (size_t i = 0; i < niter; ++i) {
    strt = get_timestamp();
    gp_band2.compute(x, yerr);
    compute_time += get_timestamp() - strt;

    strt = get_timestamp();
    log_like = gp_band2.log_likelihood(y);
    log_like_time += get_timestamp() - strt;
  }

  // Print the results.
  std::cout << "N = " << N << " [averaging " << niter << "]\n";
  std::cout << "compute cost = " << compute_time / niter << " sec\n";
  std::cout << "log likelihood cost = " << log_like_time / niter << " sec\n";
  std::cout << "log likelihood = " << log_like << "\n";

  return 0;
}
