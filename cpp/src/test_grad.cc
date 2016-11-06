#include <iostream>
#include <Eigen/Core>
#include <unsupported/Eigen/AutoDiff>

#include "genrp/solvers/band.h"

int main (int argc, char* argv[])
{
  srand(42);

  size_t N = 3;

  // Set up the coefficients.
  Eigen::VectorXd alpha_real = Eigen::VectorXd::Random(1),
                  beta_real = Eigen::VectorXd::Random(1);
  alpha_real.array() += 1.0;
  beta_real.array() += 1.0;

  typedef Eigen::AutoDiffScalar<Eigen::VectorXd> ad_t;
  Eigen::Matrix<ad_t, Eigen::Dynamic, 1> alpha_real_grad(1), beta_real_grad(1),
                                         alpha_complex_grad, beta_complex_real_grad,
                                         beta_complex_imag_grad;
  Eigen::Matrix<ad_t, Eigen::Dynamic, Eigen::Dynamic> a, al;
  Eigen::VectorXi ipiv;
  alpha_real_grad(0) = ad_t(alpha_real(0), 2, 0);
  beta_real_grad(0) = ad_t(beta_real(0), 2, 1);

  // Generate some fake data.
  Eigen::VectorXd x = Eigen::VectorXd::Random(N),
                  yerr2 = Eigen::VectorXd::Random(N),
                  y;

  // Set the scale of the uncertainties.
  yerr2.array() *= 0.1;
  yerr2.array() += 1.0;
  yerr2.setConstant(0.5);

  // The times need to be sorted.
  std::sort(x.data(), x.data() + x.size());

  // Compute the y values.
  y = sin(x.array());

  genrp::BandSolver solver(alpha_real, beta_real);
  solver.compute(x, yerr2);

  ad_t ld = solver.build_matrix(alpha_real_grad, beta_real_grad, alpha_complex_grad,
                      beta_complex_real_grad,
                      beta_complex_imag_grad, x, yerr2, a, al, ipiv);
  std::cout << ld.derivatives() << std::endl;

  double eps = 1.23e-4, delta = 0.0;
  beta_real(0) += eps;

  solver = genrp::BandSolver(alpha_real, beta_real);
  solver.compute(x, yerr2);
  delta = solver.log_determinant();

  beta_real(0) -= 2*eps;
  solver = genrp::BandSolver(alpha_real, beta_real);
  solver.compute(x, yerr2);
  delta -= solver.log_determinant();

  std::cout << (0.5 * delta / eps) << std::endl;

  return 0;
}
