#ifndef _GENRP_GAUSSIAN_PROCESS_H_
#define _GENRP_GAUSSIAN_PROCESS_H_

#include <Eigen/Core>
#include "genrp/kernel.h"

namespace genrp {

// 0.5 * log(2 * pi)
#define GAUSSIAN_PROCESS_CONSTANT 0.91893853320467267

template <typename SolverType, typename T>
class GaussianProcess {
  typedef Eigen::Matrix<T, Eigen::Dynamic, 1> vector_t;
public:
  GaussianProcess (Kernel<T> kernel) : kernel_(kernel), dim_(0), computed_(false) {}

  size_t size () const { return kernel_.size(); };

  const Kernel<T>& kernel () const { return kernel_; };

  vector_t params () const { return kernel_.params(); };
  void params (const vector_t& pars) { kernel_.params(pars); };

  void compute (const Eigen::VectorXd& x, const Eigen::VectorXd& yerr) {
    dim_ = x.rows();
    solver_.compute(
      kernel_.alpha_real(), kernel_.beta_real(),
      kernel_.alpha_complex(), kernel_.beta_complex_real(), kernel_.beta_complex_imag(),
      x, (yerr.array() * yerr.array()).cast<T>()
    );
    computed_ = true;
  }
  void compute (const vector_t& params, const Eigen::VectorXd& x, const Eigen::VectorXd& yerr) {
    kernel_.params(params);
    compute(x, yerr);
  };
  T log_likelihood (const Eigen::VectorXd& y) const {
    check_computed();
    assert((y.rows() == dim_) && "DIMENSION MISMATCH");
    vector_t alpha(dim_);
    T nll = 0.5 * solver_.dot_solve(y);
    nll += 0.5 * solver_.log_determinant() + y.rows() * GAUSSIAN_PROCESS_CONSTANT;
    return -nll;
  }

  const SolverType& solver () const {
    check_computed();
    return solver_;
  };

  // Eigen-free interface.
  void compute (size_t n, const double* x, const double* yerr);
  T log_likelihood (const double* y) const;
  T kernel_value (double dt) const;
  T kernel_psd (double w) const;
  void get_params (T* pars) const;
  void set_params (const T* const pars);

  void get_alpha_real (T* alpha) const;
  void get_beta_real (T* beta) const;
  void get_alpha_complex (T* alpha) const;
  void get_beta_complex_real (T* beta) const;
  void get_beta_complex_imag (T* beta) const;

private:
  Kernel<T> kernel_;
  SolverType solver_;
  size_t dim_;
  bool computed_;

  void check_computed () const {
    assert(computed_ && "YOU MUST COMPUTE THE GAUSSIAN_PROCESS");
  };
};

// Eigen-free interface.
template <typename SolverType, typename T>
void GaussianProcess<SolverType, T>::compute (size_t n, const double* x, const double* yerr) {
  typedef Eigen::Map<const Eigen::Matrix<double, Eigen::Dynamic, 1> > map_t;
  compute(map_t(x, n), map_t(yerr, n));
}

template <typename SolverType, typename T>
T GaussianProcess<SolverType, T>::log_likelihood (const double* y) const {
  typedef Eigen::Map<const Eigen::Matrix<double, Eigen::Dynamic, 1> > map_t;
  return log_likelihood(map_t(y, dim_));
}

template <typename SolverType, typename T>
T GaussianProcess<SolverType, T>::kernel_value (double dt) const {
  return kernel_.value(dt);
}

template <typename SolverType, typename T>
T GaussianProcess<SolverType, T>::kernel_psd (double w) const {
  return kernel_.psd(w);
}

template <typename SolverType, typename T>
void GaussianProcess<SolverType, T>::get_params (T* pars) const {
  vector_t p = kernel_.params();
  for (size_t i = 0; i < p.rows(); ++i) pars[i] = p(i);
}

template <typename SolverType, typename T>
void GaussianProcess<SolverType, T>::set_params (const T* const pars) {
  typedef Eigen::Map<const Eigen::Matrix<double, Eigen::Dynamic, 1> > map_t;
  kernel_.params(map_t(pars, kernel_.size()));
}

template <typename SolverType, typename T>
void GaussianProcess<SolverType, T>::get_alpha_real (T* alpha) const {
  vector_t a = kernel_.alpha_real();
  for (size_t i = 0; i < a.rows(); ++i) alpha[i] = a(i);
}

template <typename SolverType, typename T>
void GaussianProcess<SolverType, T>::get_beta_real (T* beta) const {
  vector_t a = kernel_.beta_real();
  for (size_t i = 0; i < a.rows(); ++i) beta[i] = a(i);
}

template <typename SolverType, typename T>
void GaussianProcess<SolverType, T>::get_alpha_complex (T* alpha) const {
  vector_t a = kernel_.alpha_complex();
  for (size_t i = 0; i < a.rows(); ++i) alpha[i] = a(i);
}

template <typename SolverType, typename T>
void GaussianProcess<SolverType, T>::get_beta_complex_real (T* beta) const {
  vector_t a = kernel_.beta_complex_real();
  for (size_t i = 0; i < a.rows(); ++i) beta[i] = a(i);
}

template <typename SolverType, typename T>
void GaussianProcess<SolverType, T>::get_beta_complex_imag (T* beta) const {
  vector_t a = kernel_.beta_complex_imag();
  for (size_t i = 0; i < a.rows(); ++i) beta[i] = a(i);
}

};

#endif
