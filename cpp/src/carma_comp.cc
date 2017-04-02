#include <iostream>
#include <cmath>
#include <complex>
#include <Eigen/Core>

#include "celerite/celerite.h"
#include "celerite/carma.h"
#include "celerite/utils.h"
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

  size_t max_terms = 10;
  if (argc >= 2) max_terms = atoi(argv[1]);
  size_t N = pow(2, 16);
  if (argc >= 3) N = atoi(argv[2]);
  size_t niter = 5;
  if (argc >= 4) niter = atoi(argv[3]);

  double sigma = 1.0;

  // Generate some fake data.
  Eigen::VectorXd x = Eigen::VectorXd::Random(N),
                  yerr = Eigen::VectorXd::Random(N),
                  y, diag;
  yerr.array() *= 0.1;
  yerr.array() += 1.0;
  diag = yerr.array() * yerr.array();
  std::sort(x.data(), x.data() + x.size());
  y = sin(x.array());

  double carma_ll, celerite_ll, strt;
  Eigen::MatrixXd compute_times(max_terms, 3);
  compute_times.setConstant(0.0);
  for (size_t nterms = 1; nterms <= max_terms; ++nterms) {
    Eigen::VectorXd carma_arparams,
                    carma_maparams,
                    alpha_real, beta_real, alpha_complex_real, alpha_complex_imag,
                    beta_complex_real, beta_complex_imag;

    compute_times(nterms - 1, 0) = nterms;

    bool is_ok = false;
    while (!is_ok) {
      // Resample the parameters until good ones are chosen
      carma_arparams = Eigen::VectorXd::Random(nterms);
      carma_maparams = Eigen::VectorXd::Random(nterms-1);
      carma_arparams.array() += 1.0;
      carma_maparams.array() += 1.0;
      carma_arparams.array() /= 2.0 * nterms;
      carma_maparams.array() /= 2.0 * nterms;
      celerite::carma::CARMASolver carma_solver(0.0, carma_arparams, carma_maparams);
      carma_solver.get_celerite_coeffs(alpha_real, beta_real,
        alpha_complex_real, alpha_complex_imag, beta_complex_real, beta_complex_imag);

      is_ok = celerite::check_coefficients(alpha_real, beta_real, alpha_complex_real, alpha_complex_imag, beta_complex_real, beta_complex_imag);
    }

    // Compute using the CARMA model.
    strt = get_timestamp();
    for (size_t i = 0; i < niter; ++i) {
      celerite::carma::CARMASolver carma_solver(0.0, carma_arparams, carma_maparams);
      carma_solver.setup();
      carma_ll = carma_solver.log_likelihood(x, y, yerr);
    }
    compute_times(nterms - 1, 1) = (get_timestamp() - strt) / niter;

    // Get the celerite parameters for the CARMA model.
    celerite::carma::CARMASolver carma_solver(0.0, carma_arparams, carma_maparams);
    carma_solver.get_celerite_coeffs(alpha_real, beta_real,
      alpha_complex_real, alpha_complex_imag, beta_complex_real, beta_complex_imag);

    // Compute using the celerite model.
    celerite::solver::BandSolver<double> solver;
    int flag = -1;
    strt = get_timestamp();
    for (size_t i = 0; i < niter; ++i) {
      flag = solver.compute(alpha_real, beta_real, alpha_complex_real, alpha_complex_imag, beta_complex_real, beta_complex_imag, x, diag);
      if (flag) {
        celerite_ll = NAN;
        break;
      }
      celerite_ll = -0.5*(solver.dot_solve(y) + solver.log_determinant() + x.rows() * log(2.0 * M_PI));
    }
    double direct_ll = NAN;
    if (!flag) {
      celerite::solver::DirectSolver<double> dsolver;
      dsolver.compute(alpha_real, beta_real, alpha_complex_real, alpha_complex_imag, beta_complex_real, beta_complex_imag, x, diag);
      direct_ll = -0.5*(dsolver.dot_solve(y) + dsolver.log_determinant() + x.rows() * log(2.0 * M_PI));

      compute_times(nterms - 1, 2) = (get_timestamp() - strt) / niter;
    } else {
      compute_times(nterms - 1, 2) = NAN;
    }


    std::cerr << nterms << " " << direct_ll << " " << carma_ll << " " << celerite_ll << std::endl;

    // if (flag) {
    //   std::cerr << "alpha_real: " << alpha_real.transpose() << std::endl;
    //   std::cerr << "beta_real: "  << beta_real.transpose()  << std::endl;
    //   std::cerr << "alpha_complex_real: " << alpha_complex_real.transpose() << std::endl;
    //   std::cerr << "alpha_complex_imag: " << alpha_complex_imag.transpose() << std::endl;
    //   std::cerr << "beta_complex_real: "  << beta_complex_real.transpose()  << std::endl;
    //   std::cerr << "beta_complex_imag: "  << beta_complex_imag.transpose()  << std::endl;
    //   std::cerr << std::endl;

    //   for (double t = 0.0; t <= 5000.0; t += 0.01) {
    //     double p = celerite::get_psd_value(alpha_real, beta_real, alpha_complex_real, alpha_complex_imag, beta_complex_real, beta_complex_imag, t);
    //     if (p < 0.0) {
    //       std::cerr << t << " " << p << std::endl;
    //     }
    //   }
    //   std::cerr << std::endl;
    //   // return 1;
    // }
  }

  std::cout<< compute_times << std::endl;

  return 0;
}

//   // Set up the coefficients.
//   /* size_t n_complex = nterms / 2, */
//   /*        n_real = nterms - n_complex * 2; */
//   size_t n_complex = 0, n_real = nterms;
//   std::cout << n_real << " " << n_complex << "\n";
//   Eigen::VectorXd alpha_real = Eigen::VectorXd::Random(n_real),
//                   beta_real = Eigen::VectorXd::Random(n_real),
//                   alpha_complex = Eigen::VectorXd::Random(n_complex),
//                   beta_complex_real = Eigen::VectorXd::Random(n_complex),
//                   beta_complex_imag = Eigen::VectorXd::Random(n_complex);
//   if (n_real > 0) {
//     alpha_real.array() += 1.0;
//     beta_real.array() += 1.0;
//   }
//   if (n_complex > 0) {
//     alpha_complex.array() += 1.0;
//     beta_complex_real.array() += 1.0;
//     beta_complex_imag.array() += 1.0;
//   }
//
//   // Generate some fake data.
//   Eigen::VectorXd x0 = Eigen::VectorXd::Random(N_max),
//                   yerr0 = Eigen::VectorXd::Random(N_max),
//                   y0;
//
//   // Set the scale of the uncertainties.
//   yerr0.array() *= 0.1;
//   yerr0.array() += 1.0;
//
//   // The times need to be sorted.
//   std::sort(x0.data(), x0.data() + x0.size());
//
//   // Compute the y values.
//   y0 = sin(x0.array());
//
//   celerite::BandSolver<double> solver;
//
//   for (size_t N = 64; N <= N_max; N *= 2) {
//     Eigen::VectorXd x = x0.topRows(N),
//                     yerr = yerr0.topRows(N),
//                     y = y0.topRows(N);
//
//     // Benchmark the solver.
//     double strt, compute_time = 0.0, solve_time = 0.0, carma_time = 0.0;
//
//     for (size_t i = 0; i < niter; ++i) {
//       strt = get_timestamp();
//       solver.compute(alpha_real, beta_real, alpha_complex, beta_complex_real, beta_complex_imag, x, yerr);
//       compute_time += get_timestamp() - strt;
//     }
//
//     for (size_t i = 0; i < niter; ++i) {
//       strt = get_timestamp();
//       solver.dot_solve(y);
//       solve_time += get_timestamp() - strt;
//     }
//
//     for (size_t i = 0; i < niter; ++i) {
//       strt = get_timestamp();
//       celerite::carma::CARMASolver carma_solver(0.0, carma_arparams, carma_maparams);
//       carma_solver.setup();
//       carma_solver.log_likelihood(x, y, yerr);
//       carma_time += get_timestamp() - strt;
//     }
//
//     // Print the results.
//     std::cout << N;
//     std::cout << " ";
//     std::cout << (compute_time + solve_time) / niter;
//     std::cout << " ";
//     std::cout << carma_time / niter;
//     std::cout << "\n";
//   }
//
//   return 0;
// }
