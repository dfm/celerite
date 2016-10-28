#ifndef _GENRP_SOLVER_BAND_
#define _GENRP_SOLVER_BAND_

#include <cmath>
#include <Eigen/Dense>

#include "genrp/utils.h"
#include "genrp/lapack.h"
#include "genrp/solvers/direct.h"

namespace genrp {

class BandSolver : public DirectSolver<entry_t> {
  typedef Eigen::Matrix<double, Eigen::Dynamic, 1> real_vector_t;
  typedef Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> real_matrix_t;
  typedef Eigen::Matrix<std::complex<double>, Eigen::Dynamic, 1> complex_vector_t;
  typedef Eigen::Matrix<std::complex<double>, Eigen::Dynamic, Eigen::Dynamic> complex_matrix_t;

public:
  BandSolver () : DirectSolver() {};
  BandSolver (const real_vector_t alpha, const real_vector_t beta)
    : DirectSolver(alpha, beta) {};
  BandSolver (const real_vector_t alpha, const complex_vector_t beta)
    : DirectSolver(alpha, beta) {};
  BandSolver (const real_vector_t alpha_real, const real_vector_t beta_real,
              const real_vector_t alpha_complex, const complex_vector_t beta_complex)
    : DirectSolver(alpha_real, beta_real, alpha_complex, beta_complex) {};
  // BandSolver (size_t p, const double* alpha, const entry_t* beta) : DirectSolver<entry_t>(p, alpha, beta) {};

  void compute (const real_vector_t& x, const real_vector_t& diag);
  void solve (const real_matrix_t& b, double* x) const;

  // // Needed for the Eigen-free interface.
  // using DirectSolver<entry_t>::compute;
  // using DirectSolver<entry_t>::solve;

private:
  size_t block_size_, dim_ext_;
  real_matrix_t factor_;
  Eigen::VectorXi ipiv_;

};

void BandSolver::compute (const Eigen::VectorXd& x, const Eigen::VectorXd& diag) {
  // Check dimensions.
  assert ((x.rows() != diag.rows()) && "DIMENSION_MISMATCH");
  this->n_ = x.rows();

  // Dimensions.
  size_t p_real = this->p_real_,
         p_complex = this->p_complex_,
         p = p_real + p_complex,
         n = this->n_,
         m1 = p_real + 2*(p - p_real + 1);

  if (p_complex == 0) m1 = p_real + 1;

  // Compute the block sizes: Algorithm 1
  block_size_ = 2 * m1 + 1;
  dim_ext_ = (4 * p_complex_ + 2 * p_real_ + 1) * (n - 1) + 1;

  // Set up the extended matrix.
  Eigen::internal::BandMatrix<entry_t> ab(dim_ext_, dim_ext_, 2*(p_+1), p_+1);
  ipiv_.resize(dim_ext_);

  double ebt, phi;
  real_vector_t gamma_real(p), gamma_imag = real_vector_t::Zero(p);

  // Special case for the first row. Eq 61 and l_1 = 0.
  aex
  for ()


  real_matrix_t gamma_real(p, n_ - 1),
                gamma_imag = real_matrix_t::Zero(p, n_ - 1);
  for (size_t i = 0; i < n_ - 1; ++i) {
    double delta = fabs(x(i+1) - x(i));
    for (size_t k = 0; k < p_real; ++k)
      gamma_real_(k, i) = exp(-this->beta_real_(k) * delta);
    for (size_t k = 0; k < p_real; ++k) {
      ebt = exp(-this->beta_complex_(k).real() * delta);
      phi = this->beta_complex_(k).imag() * delta;
      gamma_real(p_real + k, i) = ebt * cos(phi);
      gamma_imag(p_real + k, i) = -ebt * sin(phi);
    }
  }

  // Pre-compute sum(alpha) -- it will be added to the diagonal.
  double sum_alpha = this->alpha_real_.sum();

  // Initialize to zero.
  ab.coeffs().setConstant(0.0);

  for (size_t i = 0; i < n_; ++i)  // Line 3
    ab.diagonal()(i*block_size_) = diag(i) + sum_alpha;

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

      // Lines 8-9:
      a = im1n+p_+k+1;
      b = in;
      value = this->alpha_(k);
      ab.diagonal(b-a)(a) = value;
      ab.diagonal(a-b)(a) = value;

      // Lines 10-11:
      a = im1n+k+1;
      b = im1n+p_+k+1;
      value = -1.0;
      ab.diagonal(b-a)(a) = value;
      ab.diagonal(a-b)(a) = value;
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
    }
  }

  // Build and factorize the sparse matrix.
  band_factorize(ab, ipiv_);
  factor_ = ab.coeffs();

  // Deal with negative values in the diagonal.
  Eigen::VectorXcd d = ab.diagonal().template cast<std::complex<double> >();

  this->log_det_ = get_real(log(d.array()).sum());
}

template <typename entry_t>
void BandSolver<entry_t>::solve (const Eigen::MatrixXd& b, double* x) const {
  assert ((b.rows() != this->n_) && "DIMENSION_MISMATCH");
  size_t nrhs = b.cols();

  // Pad the input vector to the extended size.
  matrix_t bex = matrix_t::Zero(dim_ext_, nrhs);
  for (size_t j = 0; j < nrhs; ++j)
    for (size_t i = 0; i < this->n_; ++i)
      bex(i*block_size_, j) = b(i, j);

  // Solve the extended system.
  band_solve(this->p_+1, this->p_+1, factor_, ipiv_, bex);

  // Copy the output.
  for (size_t j = 0; j < nrhs; ++j)
    for (size_t i = 0; i < this->n_; ++i)
      x[i+j*nrhs] = get_real(bex(i*block_size_, j));
}

};

#endif
