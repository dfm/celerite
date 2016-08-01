#ifndef _GENRP_SOLVER_SPARSE_
#define _GENRP_SOLVER_SPARSE_

#include <cmath>
#include <vector>
#include <Eigen/Dense>
#include <Eigen/Sparse>

#include "genrp/utils.h"
#include "genrp/solvers/direct.h"

namespace genrp {

template <typename entry_t>
class SparseSolver : public DirectSolver<entry_t> {
  typedef Eigen::Matrix<entry_t, Eigen::Dynamic, 1> vector_t;
  typedef Eigen::Matrix<entry_t, Eigen::Dynamic, Eigen::Dynamic> matrix_t;

public:
  SparseSolver () : DirectSolver<entry_t>() {};
  SparseSolver (const Eigen::VectorXd alpha, const vector_t beta) : DirectSolver<entry_t>(alpha, beta) {};
  SparseSolver (size_t p, const double* alpha, const entry_t* beta) : DirectSolver<entry_t>(p, alpha, beta) {};

  void compute (const Eigen::VectorXd& x, const Eigen::VectorXd& diag);
  void solve (const Eigen::MatrixXd& b, double* x) const;

  using DirectSolver<entry_t>::compute;
  using DirectSolver<entry_t>::solve;

private:
  size_t block_size_, dim_ext_;
  Eigen::SparseLU<Eigen::SparseMatrix<entry_t> > factor_;

};

template <typename entry_t>
void SparseSolver<entry_t>::compute (const Eigen::VectorXd& x, const Eigen::VectorXd& diag) {
  typedef Eigen::Triplet<entry_t> triplet_t;

  // Check dimensions.
  if (x.rows() != diag.rows()) throw GENRP_DIMENSION_MISMATCH;
  this->n_ = x.rows();

  // Dimensions from superclass.
  size_t p_ = this->p_,
         n_ = this->n_;

  // Pre-compute gamma: Equation (58)
  matrix_t gamma(p_, n_ - 1);
  for (size_t i = 0; i < n_ - 1; ++i) {
    double delta = fabs(x(i+1) - x(i));
    for (size_t k = 0; k < p_; ++k)
      gamma(k, i) = exp(-this->beta_(k) * delta);
  }

  // Pre-compute sum(alpha) -- it will be added to the diagonal.
  double sum_alpha = this->alpha_.sum();

  // Compute the block sizes: Algorithm 1
  block_size_ = 2 * p_ + 1;
  dim_ext_ = block_size_ * n_ - 2 * p_;

  // Find the non-zero entries in the sparse representation using Algorithm 1.
  size_t nnz = (n_-1)*(6*p_+1) + 2*(n_-2)*p_ + 1, count = 0;
  std::vector<triplet_t> triplets(nnz);

  for (size_t i = 0; i < n_; ++i)  // Line 3
    triplets[count++] = triplet_t(i * block_size_, i * block_size_, diag(i)+sum_alpha);

  size_t a, b;
  entry_t value;
  for (size_t i = 0; i < n_ - 1; ++i) {  // Line 5
    size_t im1n = i * block_size_,        // (i - 1) * n
           in = (i + 1) * block_size_;    // i * n
    for (size_t k = 0; k < p_; ++k) {
      // Lines 6-7:
      a = im1n;
      b = im1n+k+1;
      value = gamma(k, i);
      triplets[count++] = triplet_t(a, b, value);
      triplets[count++] = triplet_t(b, a, get_conj(value));

      // Lines 8-9:
      a = in;
      b = im1n+p_+k+1;
      value = this->alpha_(k);
      triplets[count++] = triplet_t(a, b, value);
      triplets[count++] = triplet_t(b, a, value);

      // Lines 10-11:
      a = im1n+k+1;
      b = im1n+p_+k+1;
      value = -1.0;
      triplets[count++] = triplet_t(a, b, value);
      triplets[count++] = triplet_t(b, a, value);
    }
  }

  for (size_t i = 0; i < n_ - 2; ++i) {  // Line 13
    size_t im1n = i * block_size_,        // (i - 1) * n
           in = (i + 1) * block_size_;    // i * n
    for (size_t k = 0; k < p_; ++k) {
      // Lines 14-15:
      a = im1n+p_+k+1;
      b = in+k+1;
      value = gamma(k, i+1);
      triplets[count++] = triplet_t(a, b, value);
      triplets[count++] = triplet_t(b, a, get_conj(value));
    }
  }

  // Build and factorize the sparse matrix.
  Eigen::SparseMatrix<entry_t> A_ex(dim_ext_, dim_ext_);
  A_ex.setFromTriplets(triplets.begin(), triplets.end());
  factor_.compute(A_ex);
  this->log_det_ = get_real(factor_.logAbsDeterminant());
}

template <typename entry_t>
void SparseSolver<entry_t>::solve (const Eigen::MatrixXd& b, double* x) const {
  if (b.rows() != this->n_) throw GENRP_DIMENSION_MISMATCH;
  size_t nrhs = b.cols();

  // Pad the input vector to the extended size.
  matrix_t bex = matrix_t::Zero(dim_ext_, nrhs);
  for (size_t j = 0; j < nrhs; ++j)
    for (size_t i = 0; i < this->n_; ++i)
      bex(i*block_size_, j) = b(i, j);

  // Solve the extended system.
  matrix_t xex = factor_.solve(bex);

  // Copy the output.
  for (size_t j = 0; j < nrhs; ++j)
    for (size_t i = 0; i < this->n_; ++i)
      x[i+j*nrhs] = get_real(xex(i*block_size_, j));
}

};

#endif
