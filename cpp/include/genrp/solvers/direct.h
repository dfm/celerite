#ifndef _GENRP_SOLVER_DIRECT_H_
#define _GENRP_SOLVER_DIRECT_H_

#include <cmath>
#include <Eigen/Dense>

#include "genrp/solvers/solver.h"

namespace genrp {

template <typename T>
class DirectSolver : public Solver<T> {
public:
  DirectSolver () {};
  ~DirectSolver () {};

  void compute (
    const Eigen::Matrix<T, Eigen::Dynamic, 1>& alpha_real,
    const Eigen::Matrix<T, Eigen::Dynamic, 1>& beta_real,
    const Eigen::Matrix<T, Eigen::Dynamic, 1>& alpha_complex,
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
void DirectSolver<T>::compute (
  const Eigen::Matrix<T, Eigen::Dynamic, 1>& alpha_real,
  const Eigen::Matrix<T, Eigen::Dynamic, 1>& beta_real,
  const Eigen::Matrix<T, Eigen::Dynamic, 1>& alpha_complex,
  const Eigen::Matrix<T, Eigen::Dynamic, 1>& beta_complex_real,
  const Eigen::Matrix<T, Eigen::Dynamic, 1>& beta_complex_imag,
  const Eigen::VectorXd& x,
  const Eigen::Matrix<T, Eigen::Dynamic, 1>& diag
)
{
  // Check the dimensions
  assert ((alpha_real.rows() == beta_real.rows()) && "DIMENSION_MISMATCH");
  assert ((alpha_complex.rows() == beta_complex_real.rows()) && "DIMENSION_MISMATCH");
  assert ((alpha_complex.rows() == beta_complex_imag.rows()) && "DIMENSION_MISMATCH");
  assert ((x.rows() == diag.rows()) && "DIMENSION_MISMATCH");

  // Save the dimensions for later use
  this->p_real_ = alpha_real.rows();
  this->p_complex_ = alpha_complex.rows();
  this->n_ = x.rows();

  // Build the matrix.
  double dx;
  T v, asum = alpha_real.sum() + alpha_complex.sum();
  Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> K(this->n_, this->n_);
  for (size_t i = 0; i < this->n_; ++i) {
    K(i, i) = asum + diag(i);

    for (size_t j = i+1; j < this->n_; ++j) {
      v = 0.0;
      dx = fabs(x(j) - x(i));
      v += (alpha_real.array() * exp(-beta_real.array() * dx)).sum();
      v += (alpha_complex.array() * exp(-beta_complex_real.array() * dx) * cos(beta_complex_imag.array() * dx)).sum();
      K(i, j) = v;
      K(j, i) = v;
    }
  }

  // Factorize the matrix.
  factor_ = K.ldlt();
  this->log_det_ = log(factor_.vectorD().array()).sum();
}

template <typename T>
void DirectSolver<T>::solve (const Eigen::MatrixXd& b, T* x) const {
  assert ((b.rows() == this->n_) && "DIMENSION_MISMATCH");
  size_t nrhs = b.cols();

  Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>
    result = factor_.solve(b.cast<T>());

  // Copy the output.
  for (size_t j = 0; j < nrhs; ++j)
    for (size_t i = 0; i < this->n_; ++i)
      x[i+j*nrhs] = result(i, j);
}

};

#endif
