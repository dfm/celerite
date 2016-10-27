#ifndef _GENRP_SOLVER_BASIC_
#define _GENRP_SOLVER_BASIC_

#include <cmath>
#include <vector>
#include <complex>
#include <Eigen/Dense>
#include <Eigen/Sparse>

#include "genrp/utils.h"

namespace genrp {

template <typename entry_t>
class BasicSolver {
  typedef Eigen::Matrix<entry_t, Eigen::Dynamic, 1> vector_t;
  typedef Eigen::Matrix<entry_t, Eigen::Dynamic, Eigen::Dynamic> matrix_t;

public:
  Eigen::VectorXd alpha_;
  vector_t beta_;
  size_t n_, p_;
  double log_det_;

public:
  BasicSolver () {};
  BasicSolver (const Eigen::VectorXd alpha, const vector_t beta);
  virtual ~BasicSolver () {};
  void alpha_and_beta (const Eigen::VectorXd alpha, const vector_t beta);
  virtual void compute (const Eigen::VectorXd& x, const Eigen::VectorXd& diag);
  virtual void solve (const Eigen::MatrixXd& b, double* x) const;
  double dot_solve (const Eigen::VectorXd& b) const;
  double log_determinant () const;

private:
  Eigen::LDLT<matrix_t> factor_;

};

template <typename entry_t>
BasicSolver<entry_t>::BasicSolver (const Eigen::VectorXd alpha, const Eigen::Matrix<entry_t, Eigen::Dynamic, 1> beta)
  : alpha_(alpha),
    beta_(beta),
    p_(alpha.rows())
{
  assert ((alpha_.rows() == beta_.rows()) && "DIMENSION_MISMATCH");
}

template <typename entry_t>
void BasicSolver<entry_t>::alpha_and_beta (const Eigen::VectorXd alpha, const Eigen::Matrix<entry_t, Eigen::Dynamic, 1> beta) {
  p_ = alpha.rows();
  alpha_ = alpha;
  beta_ = beta;
}

template <typename entry_t>
void BasicSolver<entry_t>::compute (const Eigen::VectorXd& x, const Eigen::VectorXd& diag) {
  assert ((x.rows() == diag.rows()) && "DIMENSION_MISMATCH");
  n_ = x.rows();

  // Build the matrix.
  entry_t v, asum = alpha_.sum();
  matrix_t K(n_, n_);
  for (size_t i = 0; i < n_; ++i) {
    K(i, i) = asum + diag(i);

    for (size_t j = i+1; j < n_; ++j) {
      v = entry_t(0.0);
      for (size_t p = 0; p < p_; ++p)
        v += alpha_(p) * exp(-beta_(p) * fabs(x(j) - x(i)));
      K(i, j) = v;
      K(j, i) = get_conj(v);
    }
  }

  // Factorize the matrix.
  factor_ = K.ldlt();
  log_det_ = get_real(log(factor_.vectorD().array()).sum());
}

template <typename entry_t>
void BasicSolver<entry_t>::solve (const Eigen::MatrixXd& b, double* x) const {
  assert ((b.rows() == n_) && "DIMENSION_MISMATCH");
  size_t nrhs = b.cols();

  matrix_t result = factor_.solve(b.cast<entry_t>());

  // Copy the output.
  for (size_t j = 0; j < nrhs; ++j)
    for (size_t i = 0; i < n_; ++i)
      x[i+j*nrhs] = get_real(result(i, j));
}

template <typename entry_t>
double BasicSolver<entry_t>::dot_solve (const Eigen::VectorXd& b) const {
  Eigen::VectorXd out(n_);
  solve(b, &(out(0)));
  return b.transpose() * out;
}

template <typename entry_t>
double BasicSolver<entry_t>::log_determinant () const {
  return log_det_;
}

};

#endif
