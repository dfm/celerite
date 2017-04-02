#include <iostream>
#include <Eigen/Core>

#include "celerite/celerite.h"

#include <chrono>


// Timer for the benchmark.
//http://jakascorner.com/blog/2016/04/time-measurement.html
double get_timestamp ()
{
  using micro_s = std::chrono::microseconds;
  auto tnow = std::chrono::steady_clock::now();
  auto d_micro = std::chrono::duration_cast<micro_s>(tnow.time_since_epoch()).count();
  return double(d_micro) * 1.0e-6;
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

  celerite::solver::BandSolver<double> solver;

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
      solver.compute(alpha, beta_real, x, yerr);
      real_compute_time += get_timestamp() - strt;
    }

    for (size_t i = 0; i < niter; ++i) {
      strt = get_timestamp();
      solver.dot_solve(y);
      real_solve_time += get_timestamp() - strt;
    }

    for (size_t i = 0; i < niter; ++i) {
      strt = get_timestamp();
      solver.compute(alpha, beta_real, alpha, beta_complex_real, beta_complex_imag, x, yerr);
      complex_compute_time += get_timestamp() - strt;
    }

    for (size_t i = 0; i < niter; ++i) {
      strt = get_timestamp();
      solver.dot_solve(y);
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
