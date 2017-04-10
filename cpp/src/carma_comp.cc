#include <iostream>
#include <cmath>
#include <complex>
#include <sys/time.h>
#include <Eigen/Core>

#include "celerite/celerite.h"
#include "celerite/carma.h"
#include "celerite/utils.h"

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

    std::cout << "CARMA\n";

    // Get the celerite parameters for the CARMA model.
    celerite::carma::CARMASolver carma_solver(0.0, carma_arparams, carma_maparams);
    carma_solver.get_celerite_coeffs(alpha_real, beta_real,
      alpha_complex_real, alpha_complex_imag, beta_complex_real, beta_complex_imag);

    std::cout << "coeffs\n";

    // Compute using the celerite model.
    celerite::solver::CholeskySolver<double> solver;
    strt = get_timestamp();
    for (size_t i = 0; i < niter; ++i) {
      solver.compute(alpha_real, beta_real, alpha_complex_real, alpha_complex_imag, beta_complex_real, beta_complex_imag, x, diag);
      celerite_ll = -0.5*(solver.dot_solve(y) + solver.log_determinant() + x.rows() * log(2.0 * M_PI));
    }
    std::cout << "celerite\n";

    //if (!flag) {
    //  celerite::solver::DirectSolver<double> dsolver;
    //  dsolver.compute(alpha_real, beta_real, alpha_complex_real, alpha_complex_imag, beta_complex_real, beta_complex_imag, x, diag);
    //  direct_ll = -0.5*(dsolver.dot_solve(y) + dsolver.log_determinant() + x.rows() * log(2.0 * M_PI));

    //  compute_times(nterms - 1, 2) = (get_timestamp() - strt) / niter;
    //} else {
    //  compute_times(nterms - 1, 2) = NAN;
    //}

    std::cerr << nterms << " " << carma_ll << " " << celerite_ll << std::endl;
  }

  std::cout<< compute_times << std::endl;

  return 0;
}
