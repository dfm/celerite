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

private:
typedef Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> matrix_t;
typedef Eigen::Matrix<T, Eigen::Dynamic, 1> vector_t;
Eigen::LDLT<matrix_t> factor_;

public:

DirectSolver () : Solver<T>() {};
~DirectSolver () {};

void compute (
  const T& jitter,
  const vector_t& a_real,
  const vector_t& c_real,
  const vector_t& a_comp,
  const vector_t& b_comp,
  const vector_t& c_comp,
  const vector_t& d_comp,
  const Eigen::VectorXd& A,
  const Eigen::MatrixXd& U,
  const Eigen::MatrixXd& V,
  const Eigen::VectorXd& x,
  const Eigen::VectorXd& diag
) {
  this->computed_ = false;
  if (x.rows() != diag.rows()) throw dimension_mismatch();
  if (a_real.rows() != c_real.rows()) throw dimension_mismatch();
  if (a_comp.rows() != b_comp.rows()) throw dimension_mismatch();
  if (a_comp.rows() != c_comp.rows()) throw dimension_mismatch();
  if (a_comp.rows() != d_comp.rows()) throw dimension_mismatch();

  int N = x.rows();
  bool has_general = (A.rows() != 0);
  if (has_general && A.rows() != N) throw dimension_mismatch();
  if (has_general && U.cols() != N) throw dimension_mismatch();
  if (has_general && V.cols() != N) throw dimension_mismatch();
  if (U.rows() != V.rows()) throw dimension_mismatch();

  // Save the dimensions for later use
  this->N_ = x.rows();
  int J_general = U.rows();

  // Build the matrix.
  double dx;
  T v, asum = jitter + a_real.sum() + a_comp.sum();
  matrix_t K(this->N_, this->N_);
  for (int i = 0; i < this->N_; ++i) {
    K(i, i) = asum + diag(i);
    if (has_general) K(i, i) += A(i);

    for (int j = i+1; j < this->N_; ++j) {
      dx = x(j) - x(i);
      v = get_kernel_value(a_real, c_real, a_comp, b_comp, c_comp, d_comp, dx);

      if (has_general) {
        for (int k = 0; k < J_general; ++k) v += U(k, i) * V(k, j);
      }

      K(i, j) = v;
      K(j, i) = v;
    }
  }

  // Factorize the matrix.
  factor_ = K.ldlt();
  if (factor_.info() != Eigen::Success) throw linalg_exception();

  this->log_det_ = log(factor_.vectorD().array()).sum();
  this->computed_ = true;
};

matrix_t solve (const Eigen::MatrixXd& b) const {
  if (b.rows() != this->N_) throw dimension_mismatch();
  if (!(this->computed_)) throw compute_exception();
  return factor_.solve(b.cast<T>());
};

matrix_t dot (
  const T& jitter,
  const vector_t& a_real,
  const vector_t& c_real,
  const vector_t& a_comp,
  const vector_t& b_comp,
  const vector_t& c_comp,
  const vector_t& d_comp,
  const Eigen::VectorXd& x,
  const Eigen::MatrixXd& z
) {
  if (x.rows() != z.rows()) throw dimension_mismatch();
  if (a_real.rows() != c_real.rows()) throw dimension_mismatch();
  if (a_comp.rows() != b_comp.rows()) throw dimension_mismatch();
  if (a_comp.rows() != c_comp.rows()) throw dimension_mismatch();
  if (a_comp.rows() != d_comp.rows()) throw dimension_mismatch();

  int N = z.rows();
  double dx;
  T v, asum = jitter + a_real.sum() + a_comp.sum();
  matrix_t K(N, N);
  for (int i = 0; i < N; ++i) {
    K(i, i) = asum;

    for (int j = i+1; j < N; ++j) {
      dx = x(j) - x(i);
      v = get_kernel_value(a_real, c_real, a_comp, b_comp, c_comp, d_comp, dx);
      K(i, j) = v;
      K(j, i) = v;
    }
  }

  return K * z;
}


matrix_t dot_L (const Eigen::MatrixXd& b) const {
  if (b.rows() != this->N_) throw dimension_mismatch();
  if (!(this->computed_)) throw compute_exception();
  vector_t sqrtD = sqrt(factor_.vectorD().array());
  return factor_.matrixL() * (sqrtD.asDiagonal() * b);
};

using Solver<T>::compute;

};

};
};

#endif
