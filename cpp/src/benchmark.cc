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
#include <Eigen/Core>

#include "genrp/solvers.h"

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

  size_t nterms = 3;
  if (argc >= 2) nterms = atoi(argv[1]);
  size_t N_max = pow(2, 19);
  if (argc >= 3) N_max = atoi(argv[2]);
  size_t niter = 5;
  if (argc >= 4) niter = atoi(argv[3]);

  // Set up the coefficients.
  Eigen::VectorXd alpha = Eigen::VectorXd::Random(nterms),
                  beta_real = Eigen::VectorXd::Random(nterms),
                  beta_complex_real = Eigen::VectorXd::Random(nterms),
                  beta_complex_imag = Eigen::VectorXd::Random(nterms);
  alpha.array() += 1.0;
  beta_real.array() += 1.0;
  beta_complex_real.array() += 1.0;
  beta_complex_imag.array() += 1.0;

  // Generate some fake data.
  Eigen::VectorXd x0 = Eigen::VectorXd::Random(N_max),
                  yerr0 = Eigen::VectorXd::Random(N_max),
                  y0;

  // Set the scale of the uncertainties.
  yerr0.array() *= 0.1;
  yerr0.array() += 1.0;

  // The times need to be sorted.
  std::sort(x0.data(), x0.data() + x0.size());

  // Compute the y values.
  y0 = sin(x0.array());

  genrp::BandSolver band_real(alpha, beta_real);
  genrp::BandSolver band_complex(alpha, beta_real, alpha, beta_complex_real, beta_complex_imag);

  for (size_t N = 64; N <= N_max; N *= 2) {
    Eigen::VectorXd x = x0.topRows(N),
                    yerr = yerr0.topRows(N),
                    y = y0.topRows(N);

    // Benchmark the solver.
    double strt,
           real_compute_time = 0.0, real_solve_time = 0.0,
           complex_compute_time = 0.0, complex_solve_time = 0.0;
    for (size_t i = 0; i < niter; ++i) {
      strt = get_timestamp();
      band_real.compute(x, yerr);
      real_compute_time += get_timestamp() - strt;

      strt = get_timestamp();
      band_real.dot_solve(y);
      real_solve_time += get_timestamp() - strt;

      strt = get_timestamp();
      band_complex.compute(x, yerr);
      complex_compute_time += get_timestamp() - strt;

      strt = get_timestamp();
      band_complex.dot_solve(y);
      complex_solve_time += get_timestamp() - strt;
    }

    // Print the results.
    std::cout << N;
    std::cout << " ";
    std::cout << real_compute_time / niter;
    std::cout << " ";
    std::cout << real_solve_time / niter;
    std::cout << " ";
    std::cout << complex_compute_time / niter;
    std::cout << " ";
    std::cout << complex_solve_time / niter;
    std::cout << "\n";
  }

  return 0;
}
