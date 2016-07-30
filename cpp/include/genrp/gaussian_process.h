#ifndef _GENRP_GAUSSIAN_PROCESS_
#define _GENRP_GAUSSIAN_PROCESS_

#include <complex>
#include <Eigen/Dense>

#include "genrp/kernel.h"
#include "genrp/band_solver.h"

namespace genrp {

#define GAUSSIAN_PROCESS_MUST_COMPUTE       -1
#define GAUSSIAN_PROCESS_DIMENSION_MISMATCH -2

// 0.5 * log(2 * pi)
#define GAUSSIAN_PROCESS_CONSTANT 0.91893853320467267

class GaussianProcess {
public:
  GaussianProcess (Kernel kernel) : kernel_(kernel), dim_(0), computed_(false) {}
  size_t size () const { return kernel_.size(); };
  size_t num_terms () const { return kernel_.num_terms(); };
  size_t num_coeffs () const { return kernel_.num_coeffs(); };
  Eigen::VectorXd params () const { return kernel_.params(); };
  void params (const Eigen::VectorXd& pars) { kernel_.params(pars); };

  void compute (const Eigen::VectorXd& x, const Eigen::VectorXd& yerr);
  void compute (const Eigen::VectorXd& params, const Eigen::VectorXd& x, const Eigen::VectorXd& yerr);
  double log_likelihood (const Eigen::VectorXd& y) const;

  // Eigen-free interface.
  void compute (size_t n, const double* x, const double* yerr);
  double log_likelihood (const double* y) const;
  double kernel_value (double dt) const;
  double kernel_psd (double w) const;
  void get_params (double* pars) const;
  void set_params (const double* pars);

  void get_alpha (double* alpha) const;
  void get_beta (std::complex<double>* beta) const;

private:
  Kernel kernel_;
  BandSolver<std::complex<double> > solver_;
  size_t dim_;
  bool computed_;

};

void GaussianProcess::compute (
    const Eigen::VectorXd& params, const Eigen::VectorXd& x, const Eigen::VectorXd& yerr) {
  kernel_.params(params);
  compute(x, yerr);
}

void GaussianProcess::compute (
    const Eigen::VectorXd& x, const Eigen::VectorXd& yerr) {
  dim_ = x.rows();
  solver_.alpha_and_beta(kernel_.alpha(), kernel_.beta());
  solver_.compute(x, yerr.array() * yerr.array());
  computed_ = true;
}

double GaussianProcess::log_likelihood (const Eigen::VectorXd& y) const {
  if (!computed_) throw GAUSSIAN_PROCESS_MUST_COMPUTE;
  if (y.rows() != dim_) throw GAUSSIAN_PROCESS_DIMENSION_MISMATCH;
  Eigen::VectorXd alpha(dim_);
  double nll = 0.5 * solver_.dot_solve(y);
  nll += 0.5 * solver_.log_determinant() + y.rows() * GAUSSIAN_PROCESS_CONSTANT;
  return -nll;
}

// Eigen-free interface.
void GaussianProcess::compute (size_t n, const double* x, const double* yerr) {
  typedef Eigen::Map<const Eigen::Matrix<double, Eigen::Dynamic, 1> > vector_t;
  compute(vector_t(x, n), vector_t(yerr, n));
}

double GaussianProcess::log_likelihood (const double* y) const {
  typedef Eigen::Map<const Eigen::Matrix<double, Eigen::Dynamic, 1> > vector_t;
  return log_likelihood(vector_t(y, dim_));
}

double GaussianProcess::kernel_value (double dt) const {
  return kernel_.value(dt);
}

double GaussianProcess::kernel_psd (double w) const {
  return kernel_.psd(w);
}

void GaussianProcess::get_params (double* pars) const {
  Eigen::VectorXd p = kernel_.params();
  for (size_t i = 0; i < p.rows(); ++i) pars[i] = p(i);
}

void GaussianProcess::set_params (const double* pars) {
  typedef Eigen::Map<const Eigen::Matrix<double, Eigen::Dynamic, 1> > vector_t;
  kernel_.params(vector_t(pars, kernel_.size()));
}

void GaussianProcess::get_alpha (double* alpha) const {
  Eigen::VectorXd a = kernel_.alpha();
  for (size_t i = 0; i < a.rows(); ++i) alpha[i] = a(i);
}

void GaussianProcess::get_beta (std::complex<double>* beta) const {
  Eigen::VectorXcd b = kernel_.beta();
  for (size_t i = 0; i < b.rows(); ++i) beta[i] = b(i);
}

};

#endif
