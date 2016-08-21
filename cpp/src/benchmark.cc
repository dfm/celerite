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

// Timer for the benchmark.
double get_timestamp ()
{
  struct timeval now;
  gettimeofday (&now, NULL);
  return double(now.tv_usec) * 1.0e-6 + double(now.tv_sec);
}

int main (int argc, char* argv[])
{
  srand(42);

  size_t N_max = pow(2, 19),
         niter = 10;
  std::cout << N_max << std::endl;

  // Generate some fake data.
  Eigen::VectorXd x0 = Eigen::VectorXd::Random(N_max),
                  yerr0 = Eigen::VectorXd::Random(N_max),
                  y0;

  // Set the scale of the uncertainties.
  yerr0.array() *= 0.1;
  yerr0.array() += 0.3;

  // The times need to be sorted.
  std::sort(x0.data(), x0.data() + x0.size());

  // Compute the y values.
  y0 = sin(x0.array());

  // Set up the kernel.
  genrp::Kernel kernel;
  kernel.add_term(-0.5, 0.1);
  kernel.add_term(-0.6, 0.7, 1.0);

  // Set up the GP solver.
  genrp::GaussianProcess<genrp::BandSolver<std::complex<double> > > gp_band(kernel);

  for (size_t N = 64; N <= N_max; N *= 2) {
    Eigen::VectorXd x = x0.topRows(N),
                    yerr = yerr0.topRows(N),
                    y = y0.topRows(N);

    // Benchmark the solver.
    double strt, compute_time = 0.0, log_like_time = 0.0, log_like;
    for (size_t i = 0; i < niter; ++i) {
      strt = get_timestamp();
      gp_band.compute(x, yerr);
      compute_time += get_timestamp() - strt;

      strt = get_timestamp();
      log_like = gp_band.log_likelihood(y);
      log_like_time += get_timestamp() - strt;
    }

    // Print the results.
    std::cout << N;
    std::cout << " ";
    std::cout << compute_time / niter;
    std::cout << " ";
    std::cout << log_like_time / niter;
    std::cout << "\n";
  }

  return 0;
}
