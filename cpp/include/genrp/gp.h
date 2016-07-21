#ifndef _GENRP_GP_
#define _GENRP_GP_

#include <cmath>
#include <vector>
#include <complex>
#include <Eigen/Dense>
#include <Eigen/Sparse>

#include "genrp/kernel.h"

namespace genrp {

#define GP_MUST_COMPUTE -1

// 0.5 * log(2 * pi)
#define GP_CONSTANT 0.91893853320467267

template <typename SolverType>
class GaussianProcess {
public:
  GaussianProcess (Kernel kernel) : kernel_(kernel), dim_(0), computed_(false) {}
  size_t size () const { return kernel_.size(); };
  Eigen::VectorXd params () const { return kernel_.params(); };
  void params (const Eigen::VectorXd& pars) const { kernel_.params(pars); };

  void compute (const Eigen::VectorXd& x, const Eigen::VectorXd& yerr);
  void compute (const Eigen::VectorXd& params, const Eigen::VectorXd& x, const Eigen::VectorXd& yerr);
  double log_likelihood (const Eigen::VectorXd& y) const;
  double grad_log_likelihood (const Eigen::VectorXd& x, double* grad) const;

private:
  Kernel kernel_;
  SolverType solver_;
  size_t dim_;
  bool computed_;

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
  solver_ = SolverType(kernel_.alpha(), kernel_.beta());
  solver_.compute(x, yerr.array() * yerr.array());
  computed_ = true;
}

template <typename SolverType>
double GaussianProcess<SolverType>::log_likelihood (const Eigen::VectorXd& y) const {
  if (!computed_) throw GP_MUST_COMPUTE;
  Eigen::VectorXd alpha(y.rows());
  solver_.solve(y, &(alpha(0)));
  double ll = -0.5 * y.transpose() * alpha;
  ll -= 0.5 * solver_.log_determinant() + y.rows() * GP_CONSTANT;
  return ll;
}

template <typename SolverType>
double GaussianProcess<SolverType>::grad_log_likelihood (const Eigen::VectorXd& x, double* grad) const {
  if (!computed_) throw GP_MUST_COMPUTE;
  Eigen::VectorXd alpha(y.rows());
  solver_.solve(y, &(alpha(0)));

  // Compute the likelihood.
  double ll = -0.5 * y.transpose() * alpha;
  ll -= 0.5 * solver_.log_determinant() + y.rows() * GP_CONSTANT;

  // Compute the gradient.

  return ll;
}

};

#endif
