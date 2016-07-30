#ifndef _GENRP_BAND_SOLVER_
#define _GENRP_BAND_SOLVER_

#include <cmath>
#include <vector>
#include <complex>
#include <Eigen/Dense>
#include <Eigen/Sparse>

#include "genrp/utils.h"
#include "genrp/lapack.h"

namespace genrp {

#define GENRP_DIMENSION_MISMATCH 1

template <typename entry_t>
class BandSolver {
  typedef Eigen::Matrix<entry_t, Eigen::Dynamic, 1> vector_t;
  typedef Eigen::Matrix<entry_t, Eigen::Dynamic, Eigen::Dynamic> matrix_t;

public:
  BandSolver () {};
  BandSolver (const Eigen::VectorXd alpha, const vector_t beta);
  void alpha_and_beta (const Eigen::VectorXd alpha, const vector_t beta);
  void compute (const Eigen::VectorXd& x, const Eigen::VectorXd& diag);
  void solve (const Eigen::MatrixXd& b, double* x) const;
  double dot_solve (const Eigen::VectorXd& b) const;
  double log_determinant () const;

  // Eigen-free interface.
  BandSolver (size_t p, const double* alpha, const entry_t* beta);
  void compute (size_t n, const double* t, const double* diag);
  void solve (const double* b, double* x) const;
  double dot_solve (const double* b) const;

private:
  Eigen::VectorXd alpha_;
  vector_t beta_;
  size_t n_, p_, block_size_, dim_ext_;

  matrix_t factor_;
  Eigen::VectorXi ipiv_;
  double log_det_;

};

template <typename entry_t>
BandSolver<entry_t>::BandSolver (const Eigen::VectorXd alpha, const Eigen::Matrix<entry_t, Eigen::Dynamic, 1> beta)
  : alpha_(alpha),
    beta_(beta),
    p_(alpha.rows())
{
  if (alpha_.rows() != beta_.rows()) throw GENRP_DIMENSION_MISMATCH;
}

template <typename entry_t>
void BandSolver<entry_t>::alpha_and_beta (const Eigen::VectorXd alpha, const Eigen::Matrix<entry_t, Eigen::Dynamic, 1> beta) {
  p_ = alpha.rows();
  alpha_ = alpha;
  beta_ = beta;
}

template <typename entry_t>
void BandSolver<entry_t>::compute (const Eigen::VectorXd& x, const Eigen::VectorXd& diag) {
  // Check dimensions.
  if (x.rows() != diag.rows()) throw GENRP_DIMENSION_MISMATCH;
  n_ = x.rows();

  // Pre-compute gamma: Equation (58)
  matrix_t gamma(p_, n_ - 1);
  for (size_t i = 0; i < n_ - 1; ++i) {
    double delta = fabs(x(i+1) - x(i));
    for (size_t k = 0; k < p_; ++k)
      gamma(k, i) = exp(-beta_(k) * delta);
  }

  // Pre-compute sum(alpha) -- it will be added to the diagonal.
  double sum_alpha = alpha_.sum();

  // Compute the block sizes: Algorithm 1
  block_size_ = 2 * p_ + 1;
  dim_ext_ = block_size_ * n_ - 2 * p_;

  // Set up the extended matrix.
  Eigen::internal::BandMatrix<entry_t> ab(dim_ext_, dim_ext_, 2*(p_+1), p_+1);
  ipiv_.resize(dim_ext_);

  // Initialize to zero.
  ab.diagonal().setConstant(0.0);
  for (size_t i = 1; i < p_; ++i) {
    ab.diagonal(i).setConstant(0.0);
    ab.diagonal(-i).setConstant(0.0);
  }

  for (size_t i = 0; i < n_; ++i)  // Line 3
    ab.diagonal()(i*block_size_) = diag(i) + sum_alpha;
    /* triplets[count++] = triplet_t(i * block_size_, i * block_size_, diag(i)+sum_alpha); */

  int a, b;
  entry_t value;
  for (size_t i = 0; i < n_ - 1; ++i) {  // Line 5
    size_t im1n = i * block_size_,        // (i - 1) * n
           in = (i + 1) * block_size_;    // i * n
    for (size_t k = 0; k < p_; ++k) {
      // Lines 6-7:
      a = im1n;
      b = im1n+k+1;
      value = gamma(k, i);
      ab.diagonal(b-a)(a) = value;
      ab.diagonal(a-b)(a) = get_conj(value);
      /* triplets[count++] = triplet_t(a, b, value); */
      /* triplets[count++] = triplet_t(b, a, get_conj(value)); */

      // Lines 8-9:
      a = im1n+p_+k+1;
      b = in;
      value = alpha_(k);
      ab.diagonal(b-a)(a) = value;
      ab.diagonal(a-b)(a) = value;
      /* triplets[count++] = triplet_t(a, b, value); */
      /* triplets[count++] = triplet_t(b, a, value); */

      // Lines 10-11:
      a = im1n+k+1;
      b = im1n+p_+k+1;
      value = -1.0;
      ab.diagonal(b-a)(a) = value;
      ab.diagonal(a-b)(a) = value;
      /* triplets[count++] = triplet_t(a, b, value); */
      /* triplets[count++] = triplet_t(b, a, value); */
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
      ab.diagonal(b-a)(a) = value;
      ab.diagonal(a-b)(a) = get_conj(value);
      /* triplets[count++] = triplet_t(a, b, value); */
      /* triplets[count++] = triplet_t(b, a, get_conj(value)); */
    }
  }

  // Build and factorize the sparse matrix.
  band_factorize(ab, ipiv_);
  factor_ = ab.coeffs();
  log_det_ = log(ab.diagonal().array()).sum().real();
}

template <typename entry_t>
void BandSolver<entry_t>::solve (const Eigen::MatrixXd& b, double* x) const {
  if (b.rows() != n_) throw GENRP_DIMENSION_MISMATCH;
  size_t nrhs = b.cols();

  // Pad the input vector to the extended size.
  matrix_t bex = matrix_t::Zero(dim_ext_, nrhs);
  for (size_t j = 0; j < nrhs; ++j)
    for (size_t i = 0; i < n_; ++i)
      bex(i*block_size_, j) = b(i, j);

  // Solve the extended system.
  band_solve(p_+1, p_+1, factor_, ipiv_, bex);

  // Copy the output.
  for (size_t j = 0; j < nrhs; ++j)
    for (size_t i = 0; i < n_; ++i)
      x[i+j*nrhs] = get_real(bex(i*block_size_, j));
}

template <typename entry_t>
double BandSolver<entry_t>::dot_solve (const Eigen::VectorXd& b) const {
  Eigen::VectorXd out(n_);
  solve(b, &(out(0)));
  return b.transpose() * out;
}

template <typename entry_t>
double BandSolver<entry_t>::log_determinant () const {
  return log_det_;
}

// Eigen-free interface:
template <typename entry_t>
BandSolver<entry_t>::BandSolver (size_t p, const double* alpha, const entry_t* beta) {
  p_ = p;
  alpha_ = Eigen::Map<const Eigen::Matrix<double, Eigen::Dynamic, 1> >(alpha, p, 1);
  beta_ = Eigen::Map<const Eigen::Matrix<entry_t, Eigen::Dynamic, 1> >(beta, p, 1);
}

template <typename entry_t>
void BandSolver<entry_t>::compute (size_t n, const double* t, const double* diag) {
  typedef Eigen::Map<const Eigen::Matrix<double, Eigen::Dynamic, 1> > vector_t;
  compute(vector_t(t, n, 1), vector_t(diag, n, 1));
}

template <typename entry_t>
void BandSolver<entry_t>::solve (const double* b, double* x) const {
  Eigen::Map<const Eigen::Matrix<double, Eigen::Dynamic, 1> > bin(b, n_, 1);
  solve(bin, x);
}

template <typename entry_t>
double BandSolver<entry_t>::dot_solve (const double* b) const {
  Eigen::Map<const Eigen::Matrix<double, Eigen::Dynamic, 1> > bin(b, n_, 1);
  return dot_solve(bin);
}

};

#endif
