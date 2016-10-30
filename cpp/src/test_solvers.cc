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

#include "genrp/solvers/basic.h"
#include "genrp/solvers/direct.h"
#include "genrp/solvers/band.h"

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
  size_t N = 1024;
  if (argc >= 3) N = atoi(argv[2]);
  size_t niter = 10;
  if (argc >= 4) niter = atoi(argv[3]);

  // Set up the coefficients.
  Eigen::VectorXd alpha = Eigen::VectorXd::Random(nterms),
                  beta_real = Eigen::VectorXd::Random(nterms),
                  alpha_all(3*nterms);
  Eigen::VectorXcd beta_complex = Eigen::VectorXcd::Random(nterms),
                   beta_all(3*nterms);
  alpha.array() += 1.0;
  beta_real.array() += 1.0;
  beta_complex.array() += std::complex<double>(1.0, 1.0);
  alpha_all << alpha, 0.5 * alpha.array(), 0.5 * alpha.array();
  beta_all << beta_real.cast<std::complex<double> >(), beta_complex, beta_complex.conjugate();

  // Generate some fake data.
  Eigen::VectorXd x = Eigen::VectorXd::Random(N),
                  yerr2 = Eigen::VectorXd::Random(N),
                  y;

  // Set the scale of the uncertainties.
  yerr2.array() *= 0.1;
  yerr2.array() += 1.0;

  // The times need to be sorted.
  std::sort(x.data(), x.data() + x.size());

  // Compute the y values.
  y = sin(x.array());

  genrp::BasicSolver<double> basic_real(alpha, beta_real);
  basic_real.compute(x, yerr2);
  genrp::BasicSolver<std::complex<double> > basic_complex(alpha_all, beta_all);
  basic_complex.compute(x, yerr2);

  genrp::DirectSolver direct_real(alpha, beta_real);
  direct_real.compute(x, yerr2);
  genrp::DirectSolver direct_complex(alpha, beta_real, alpha, beta_complex);
  direct_complex.compute(x, yerr2);

  genrp::BandSolver band_real(alpha, beta_real);
  band_real.compute(x, yerr2);
  genrp::BandSolver band_complex(alpha, beta_real, alpha, beta_complex);
  band_complex.compute(x, yerr2);

  std::cout << basic_real.dot_solve(y) << " ";
  std::cout << direct_real.dot_solve(y) << " " ;
  std::cout << band_real.dot_solve(y) << " " ;
  std::cout << std::endl;
  std::cout << basic_real.log_determinant() << " ";
  std::cout << direct_real.log_determinant() << " " ;
  std::cout << band_real.log_determinant() << " " ;
  std::cout << std::endl;

  std::cout << basic_complex.dot_solve(y) << " ";
  std::cout << direct_complex.dot_solve(y) << " " ;
  std::cout << band_complex.dot_solve(y) << " " ;
  std::cout << std::endl;
  std::cout << basic_complex.log_determinant() << " ";
  std::cout << direct_complex.log_determinant() << " ";
  std::cout << band_complex.log_determinant() << " ";
  std::cout << std::endl;

  // genrp::SparseSolver<double> sparse_real(alpha, beta_real);
  // sparse_real.compute(x, yerr2);
  // genrp::SparseSolver<std::complex<double> > sparse_complex(alpha, beta_complex);
  // sparse_complex.compute(x, yerr2);

  // std::cout << band_real.dot_solve(y) << " " << direct_real.dot_solve(y) << " " << sparse_real.dot_solve(y) << std::endl;
  // std::cout << band_real.log_determinant() << " " << direct_real.log_determinant() << " " << sparse_real.log_determinant() << std::endl;
  // std::cout << band_complex.dot_solve(y) << " " << direct_complex.dot_solve(y) << " " << sparse_complex.dot_solve(y) << std::endl;
  // std::cout << band_complex.log_determinant() << " " << direct_complex.log_determinant() << " " << sparse_complex.log_determinant() << std::endl;

  return 0;
}
