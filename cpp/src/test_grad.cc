#include <iostream>
#include <Eigen/Core>
#include <unsupported/Eigen/AutoDiff>

#include "genrp/solvers/band.h"

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

  // Set up the coefficients.
  Eigen::VectorXd alpha_real = Eigen::VectorXd::Random(nterms),
                  beta_real = Eigen::VectorXd::Random(nterms),
                  alpha_complex = Eigen::VectorXd::Random(nterms),
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

  // Set up the gradients
  size_t nparams = 3 * nterms + 2;
  Eigen::Matrix<ad_t, Eigen::Dynamic, 1> alpha_real_grad(nterms),
                                         beta_real_grad(nterms),
                                         alpha_complex_grad(nterms),
                                         beta_complex_real_grad(nterms),
                                         beta_complex_imag_grad(nterms),
                                         yerr2_grad(N);

  ad_t white_noise = ad_t(-5.0, nparams, 0);
  for (size_t i = 0; i < N; ++i)
    yerr2_grad(i) = yerr2(i) + exp(white_noise);

  size_t par = 1;
  for (size_t i = 0; i < nterms; ++i) {
    alpha_real_grad(i) = exp(ad_t(log(alpha_real(i)), nparams, par++));
    beta_real_grad(i) = exp(ad_t(log(beta_real(i)), nparams, par++));
    alpha_complex_grad(i) = exp(ad_t(log(alpha_complex(i)), nparams, par++));
    beta_complex_real_grad(i) = exp(ad_t(log(beta_complex_real(i)), nparams, par++));
    beta_complex_imag_grad(i) = exp(ad_t(log(beta_complex_imag(i)), nparams, par++));
  }

  genrp::BandSolver<ad_t> solver;
  solver.compute(alpha_real_grad, beta_real_grad,
      alpha_complex_grad, beta_complex_real_grad, beta_complex_imag_grad,
      x, yerr2_grad);

  ad_t ld = solver.log_determinant();

  ad_t val = solver.dot_solve(y);

  ad_t ll = -0.5 * val - 0.5 * ld;
  std::cout << ll.derivatives() << std::endl;

  // double eps = 1.23e-4, delta = 0.0;
  // beta_real(0) += eps;

  // solver = genrp::BandSolver(alpha_real, beta_real);
  // solver.compute(x, yerr2);
  // delta = solver.log_determinant();

  // beta_real(0) -= 2*eps;
  // solver = genrp::BandSolver(alpha_real, beta_real);
  // solver.compute(x, yerr2);
  // delta -= solver.log_determinant();

  // std::cout << (0.5 * delta / eps) << std::endl;

  return 0;
}
