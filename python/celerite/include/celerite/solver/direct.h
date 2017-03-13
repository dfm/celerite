#ifndef _CELERITE_SOLVER_DIRECT_H_
#define _CELERITE_SOLVER_DIRECT_H_

#include <cmath>
#include <Eigen/Dense>

#include "celerite/utils.h"
#include "celerite/exceptions.h"
#include "celerite/solver/solver.h"

namespace celerite {
namespace solver {

template <typename T>
class DirectSolver : public Solver<T> {
public:
  DirectSolver () : Solver<T>() {};
  ~DirectSolver () {};

  int compute (
    const Eigen::Matrix<T, Eigen::Dynamic, 1>& alpha_real,
    const Eigen::Matrix<T, Eigen::Dynamic, 1>& beta_real,
    const Eigen::Matrix<T, Eigen::Dynamic, 1>& alpha_complex_real,
    const Eigen::Matrix<T, Eigen::Dynamic, 1>& alpha_complex_imag,
    const Eigen::Matrix<T, Eigen::Dynamic, 1>& beta_complex_real,
    const Eigen::Matrix<T, Eigen::Dynamic, 1>& beta_complex_imag,
    const Eigen::VectorXd& x,
    const Eigen::Matrix<T, Eigen::Dynamic, 1>& diag
  );
  void solve (const Eigen::MatrixXd& b, T* x) const;

  using Solver<T>::compute;
  using Solver<T>::solve;

private:
  Eigen::LDLT<Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> > factor_;

};

template <typename T>
int DirectSolver<T>::compute (
  const Eigen::Matrix<T, Eigen::Dynamic, 1>& alpha_real,
  const Eigen::Matrix<T, Eigen::Dynamic, 1>& beta_real,
  const Eigen::Matrix<T, Eigen::Dynamic, 1>& alpha_complex_real,
  const Eigen::Matrix<T, Eigen::Dynamic, 1>& alpha_complex_imag,
  const Eigen::Matrix<T, Eigen::Dynamic, 1>& beta_complex_real,
  const Eigen::Matrix<T, Eigen::Dynamic, 1>& beta_complex_imag,
  const Eigen::VectorXd& x,
  const Eigen::Matrix<T, Eigen::Dynamic, 1>& diag
)
{
  this->computed_ = false;
  if (x.rows() != diag.rows()) return SOLVER_DIMENSION_MISMATCH;

  // Save the dimensions for later use
  this->p_real_ = alpha_real.rows();
  this->p_complex_ = alpha_complex_real.rows();
  this->n_ = x.rows();

  // Build the matrix.
  double dx;
  T v, asum = alpha_real.sum() + alpha_complex_real.sum();
  Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> K(this->n_, this->n_);
  for (int i = 0; i < this->n_; ++i) {
    K(i, i) = asum + diag(i);

    for (int j = i+1; j < this->n_; ++j) {
      dx = x(j) - x(i);
      v = get_kernel_value(alpha_real, beta_real, alpha_complex_real, alpha_complex_imag, beta_complex_real, beta_complex_imag, dx);
      K(i, j) = v;
      K(j, i) = v;
    }
  }

  // Factorize the matrix.
  factor_ = K.ldlt();
  this->log_det_ = log(factor_.vectorD().array()).sum();
  this->computed_ = true;

  return 0;
}

template <typename T>
void DirectSolver<T>::solve (const Eigen::MatrixXd& b, T* x) const {
  if (b.rows() != this->n_) throw dimension_mismatch();
  if (!(this->computed_)) throw compute_exception();
  int nrhs = b.cols();

  Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>
    result = factor_.solve(b.cast<T>());

  // Copy the output.
  for (int j = 0; j < nrhs; ++j)
    for (int i = 0; i < this->n_; ++i)
      x[i+j*this->n_] = result(i, j);
}

};
};

#endif
