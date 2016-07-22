#ifndef _GENRP_GAUSSIAN_PROCESS_
#define _GENRP_GAUSSIAN_PROCESS_

#include <complex>
#include <Eigen/Dense>

#include "genrp/kernel.h"
#include "genrp/genrp_solver.h"

namespace genrp {

#define GAUSSIAN_PROCESS_MUST_COMPUTE       -1
#define GAUSSIAN_PROCESS_DIMENSION_MISMATCH -2

// 0.5 * log(2 * pi)
#define GAUSSIAN_PROCESS_CONSTANT 0.91893853320467267

class GaussianProcess {
public:
  GaussianProcess (Kernel kernel) : kernel_(kernel), dim_(0), computed_(false) {}
  size_t size () const { return kernel_.size(); };
  Eigen::VectorXd params () const { return kernel_.params(); };
  void params (const Eigen::VectorXd& pars) { kernel_.params(pars); };

  void compute (const Eigen::VectorXd x, const Eigen::VectorXd& yerr);
  void compute (const Eigen::VectorXd& params, const Eigen::VectorXd& x, const Eigen::VectorXd& yerr);
  double log_likelihood (const Eigen::VectorXd& y) const;
  double grad_log_likelihood (const Eigen::VectorXd& y, double* grad) const;

  // Eigen-free interface.
  void compute (const double* params, size_t n, const double* x, const double* yerr);
  double log_likelihood (const double* y) const;
  double kernel_value (double dt) const;
  void get_params (double* pars) const;
  void set_params (const double* pars);

private:
  Kernel kernel_;
  GenRPSolver<std::complex<double> > solver_;
  size_t dim_;
  bool computed_;
  Eigen::VectorXd x_;

};

void GaussianProcess::compute (
    const Eigen::VectorXd& params, const Eigen::VectorXd& x, const Eigen::VectorXd& yerr) {
  kernel_.params(params);
  compute(x, yerr);
}

void GaussianProcess::compute (
    const Eigen::VectorXd x, const Eigen::VectorXd& yerr) {
  x_ = x;
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

double GaussianProcess::grad_log_likelihood (const Eigen::VectorXd& y, double* grad) const {
  if (!computed_) throw GAUSSIAN_PROCESS_MUST_COMPUTE;
  if (y.rows() != dim_) throw GAUSSIAN_PROCESS_DIMENSION_MISMATCH;
  Eigen::VectorXd alpha(dim_);
  solver_.solve(y, &(alpha(0)));

  // Compute the likelihood.
  double nll = 0.5 * solver_.dot_solve(y);
  nll += 0.5 * solver_.log_determinant() + y.rows() * GAUSSIAN_PROCESS_CONSTANT;

  // Compute 'alpha.alpha^T - K^-1' matrix.
  Eigen::MatrixXd Kinv = solver_.get_inverse();
  Kinv -= alpha * alpha.transpose();
  Kinv.array() *= -1.0;

  // Compute the gradient matrix.
  size_t grad_size = kernel_.size();
  Eigen::MatrixXd dKdt(grad_size, dim_*dim_);
  kernel_.grad(0.0, &(dKdt(0, 0)));
  for (size_t i = 0; i < dim_; ++i) {
    dKdt.col(i*dim_ + i) = dKdt.col(0);
    for (size_t j = i+1; j < dim_; ++j) {
      kernel_.grad(x_(j) - x_(i), &(dKdt(0, i*dim_ + j)));
      dKdt.col(j*dim_ + i) = dKdt.col(i*dim_ + j);
    }
  }

  // Compute the gradient.
  Eigen::Map<Eigen::VectorXd> grad_map(grad, grad_size);
  grad_map.array() = Eigen::VectorXd::Zero(grad_size);
  for (size_t i = 0; i < dim_; ++i)
    grad_map += Kinv.row(i) * dKdt.block(0, i*dim_, grad_size, dim_).transpose();
  grad_map.array() *= 0.5;

  return -nll;
}

// Eigen-free interface.
void GaussianProcess::compute (const double* params, size_t n, const double* x, const double* yerr) {
  typedef Eigen::Map<const Eigen::Matrix<double, Eigen::Dynamic, 1> > vector_t;
  compute(vector_t(params, kernel_.size()), vector_t(x, n), vector_t(yerr, n));
}

double GaussianProcess::log_likelihood (const double* y) const {
  typedef Eigen::Map<const Eigen::Matrix<double, Eigen::Dynamic, 1> > vector_t;
  return log_likelihood(vector_t(y, dim_));
}

double GaussianProcess::kernel_value (double dt) const {
  return kernel_.value(dt);
}

void GaussianProcess::get_params (double* pars) const {
  Eigen::VectorXd p = kernel_.params();
  for (size_t i = 0; i < p.rows(); ++i) pars[i] = p(i);
}

void GaussianProcess::set_params (const double* pars) {
  typedef Eigen::Map<const Eigen::Matrix<double, Eigen::Dynamic, 1> > vector_t;
  kernel_.params(vector_t(pars, kernel_.size()));
}

};

#endif
