#ifndef _GENRP_GAUSSIAN_PROCESS_H_
#define _GENRP_GAUSSIAN_PROCESS_H_

#include <complex>
#include <Eigen/Dense>

#include "genrp/kernel.h"

namespace genrp {

#define GAUSSIAN_PROCESS_MUST_COMPUTE       -1
#define GAUSSIAN_PROCESS_DIMENSION_MISMATCH -2

// 0.5 * log(2 * pi)
#define GAUSSIAN_PROCESS_CONSTANT 0.91893853320467267

template <typename SolverType>
class GaussianProcess {
public:
  GaussianProcess (Kernel kernel) : kernel_(kernel), dim_(0), computed_(false) {}

  size_t size () const { return kernel_.size(); };

  const Kernel& kernel () const { return kernel_; };

  Eigen::VectorXd params () const { return kernel_.params(); };
  void params (const Eigen::VectorXd& pars) { kernel_.params(pars); };

  void compute (const Eigen::VectorXd& x, const Eigen::VectorXd& yerr);
  void compute (const Eigen::VectorXd& params, const Eigen::VectorXd& x, const Eigen::VectorXd& yerr);
  double log_likelihood (const Eigen::VectorXd& y) const;

  const SolverType& solver () const {
    check_computed();
    return solver_;
  };

  // Eigen-free interface.
  void compute (size_t n, const double* x, const double* yerr);
  double log_likelihood (const double* y) const;
  double kernel_value (double dt) const;
  double kernel_psd (double w) const;
  void get_params (double* pars) const;
  void set_params (const double* pars);

  void get_alpha_real (double* alpha) const;
  void get_beta_real (double* beta) const;
  void get_alpha_complex (double* alpha) const;
  void get_beta_complex_real (double* beta) const;
  void get_beta_complex_imag (double* beta) const;

private:
  Kernel kernel_;
  SolverType solver_;
  size_t dim_;
  bool computed_;

  void check_computed () const {
    assert(computed_ && "YOU MUST COMPUTE THE GAUSSIAN_PROCESS");
  };
};

template <typename SolverType>
void GaussianProcess<SolverType>::compute (
    const Eigen::VectorXd& params, const Eigen::VectorXd& x, const Eigen::VectorXd& yerr) {
  kernel_.params(params);
  compute(x, yerr);
}

template <typename SolverType>
void GaussianProcess<SolverType>::compute (
    const Eigen::VectorXd& x, const Eigen::VectorXd& yerr) {
  dim_ = x.rows();
  solver_.compute(
    kernel_.alpha_real(), kernel_.beta_real(),
    kernel_.alpha_complex(), kernel_.beta_complex_real(), kernel_.beta_complex_imag(),
    x, yerr.array() * yerr.array()
  );
  computed_ = true;
}

template <typename SolverType>
double GaussianProcess<SolverType>::log_likelihood (const Eigen::VectorXd& y) const {
  if (!computed_) throw GAUSSIAN_PROCESS_MUST_COMPUTE;
  if (y.rows() != dim_) throw GAUSSIAN_PROCESS_DIMENSION_MISMATCH;
  Eigen::VectorXd alpha(dim_);
  double nll = 0.5 * solver_.dot_solve(y);
  nll += 0.5 * solver_.log_determinant() + y.rows() * GAUSSIAN_PROCESS_CONSTANT;
  return -nll;
}

// Eigen-free interface.
template <typename SolverType>
void GaussianProcess<SolverType>::compute (size_t n, const double* x, const double* yerr) {
  typedef Eigen::Map<const Eigen::Matrix<double, Eigen::Dynamic, 1> > vector_t;
  compute(vector_t(x, n), vector_t(yerr, n));
}

template <typename SolverType>
double GaussianProcess<SolverType>::log_likelihood (const double* y) const {
  typedef Eigen::Map<const Eigen::Matrix<double, Eigen::Dynamic, 1> > vector_t;
  return log_likelihood(vector_t(y, dim_));
}

template <typename SolverType>
double GaussianProcess<SolverType>::kernel_value (double dt) const {
  return kernel_.value(dt);
}

template <typename SolverType>
double GaussianProcess<SolverType>::kernel_psd (double w) const {
  return kernel_.psd(w);
}

template <typename SolverType>
void GaussianProcess<SolverType>::get_params (double* pars) const {
  Eigen::VectorXd p = kernel_.params();
  for (size_t i = 0; i < p.rows(); ++i) pars[i] = p(i);
}

template <typename SolverType>
void GaussianProcess<SolverType>::set_params (const double* pars) {
  typedef Eigen::Map<const Eigen::Matrix<double, Eigen::Dynamic, 1> > vector_t;
  kernel_.params(vector_t(pars, kernel_.size()));
}

template <typename SolverType>
void GaussianProcess<SolverType>::get_alpha_real (double* alpha) const {
  Eigen::VectorXd a = kernel_.alpha_real();
  for (size_t i = 0; i < a.rows(); ++i) alpha[i] = a(i);
}

template <typename SolverType>
void GaussianProcess<SolverType>::get_beta_real (double* beta) const {
  Eigen::VectorXd a = kernel_.beta_real();
  for (size_t i = 0; i < a.rows(); ++i) beta[i] = a(i);
}

template <typename SolverType>
void GaussianProcess<SolverType>::get_alpha_complex (double* alpha) const {
  Eigen::VectorXd a = kernel_.alpha_complex();
  for (size_t i = 0; i < a.rows(); ++i) alpha[i] = a(i);
}

template <typename SolverType>
void GaussianProcess<SolverType>::get_beta_complex_real (double* beta) const {
  Eigen::VectorXd a = kernel_.beta_complex_real();
  for (size_t i = 0; i < a.rows(); ++i) beta[i] = a(i);
}

template <typename SolverType>
void GaussianProcess<SolverType>::get_beta_complex_imag (double* beta) const {
  Eigen::VectorXd a = kernel_.beta_complex_imag();
  for (size_t i = 0; i < a.rows(); ++i) beta[i] = a(i);
}

};

#endif
