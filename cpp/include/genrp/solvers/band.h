#ifndef _GENRP_SOLVER_BAND_
#define _GENRP_SOLVER_BAND_

#include <cmath>
#include <Eigen/Dense>

#include "genrp/utils.h"
#include "genrp/lapack.h"
#include "genrp/solvers/direct.h"

namespace genrp {

class BandSolver : public DirectSolver {
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
  assert ((x.rows() == diag.rows()) && "DIMENSION_MISMATCH");
  this->n_ = x.rows();

  // Dimensions.
  size_t p_real = this->p_real_,
         p_complex = this->p_complex_,
         p = p_real + p_complex,
         n = this->n_,
         j, k;

  // Compute the block sizes.
  block_size_ = 2 * p_real + 4 * p_complex + 1;
  dim_ext_ = block_size_ * (n - 1) + 1;

  // Set up the extended matrix.
  Eigen::internal::BandMatrix<double> ab(dim_ext_, dim_ext_, 2*block_size_, block_size_);
  ab.coeffs().setConstant(0.0);
  ipiv_.resize(dim_ext_);

  // Start with the diagonal.
  double sum_alpha = this->alpha_real_.sum() + 2.0 * this->alpha_complex_.sum();
  for (k = 0; k < n_; ++k)
    ab.diagonal()(k*block_size_) = diag(k) + sum_alpha;

  // Fill in all but the last block.
  int block_id, start_a, start_b, a, b;
  double dt, value;
  Eigen::Array<double, Eigen::Dynamic, 1> ebt, phi,
                                          gamma_real(p_real),
                                          gamma_complex_real(p_complex),
                                          gamma_complex_imag(p_complex);
  for (k = 0; k < n_ - 1; ++k) {
    // Pre-compute the gammas.
    dt = std::abs(x(k+1) - x(k));
    gamma_real = exp(-this->beta_real_.array() * dt);
    ebt = exp(-this->beta_complex_.real().array() * dt);
    phi = this->beta_complex_.imag().array() * dt;
    gamma_complex_real = ebt * cos(phi);
    gamma_complex_imag = -ebt * sin(phi);

    // Equations for the rs:
    block_id = block_size_ * k;
    start_b = block_id + 1;
    for (j = 0; j < p_real; ++j) {
      a = block_id;
      b = start_b + j;
      value = gamma_real(j);
      ab.diagonal(b-a)(a) = value;
      ab.diagonal(a-b)(a) = value;
    }
    start_b += p_real;
    for (j = 0; j < p_complex; ++j) {
      a = block_id;
      b = start_b + 2*j;
      value = 2.0 * gamma_complex_real(j);
      ab.diagonal(b-a)(a) = value;
      ab.diagonal(a-b)(a) = value;

      b = start_b + 2*j + 1;
      value = 2.0 * gamma_complex_imag(j);
      ab.diagonal(b-a)(a) = value;
      ab.diagonal(a-b)(a) = value;
    }

    // Equations for the ls:
    start_a = block_id + 1;
    start_b += 2*p_complex;
    for (j = 0; j < p_real; ++j) {
      a = start_a + j;
      b = start_b + j;
      value = -1.0;
      ab.diagonal(b-a)(a) = value;
      ab.diagonal(a-b)(a) = value;
    }
    start_a += p_real;
    start_b += p_real;
    for (j = 0; j < p_complex; ++j) {
      a = start_a + 2*j;
      b = start_b + 2*j;
      value = -1.0;
      ab.diagonal(b-a)(a) = value;
      ab.diagonal(a-b)(a) = value;

      a += 1;
      b += 1;
      value = 1.0;
      ab.diagonal(b-a)(a) = value;
      ab.diagonal(a-b)(a) = value;
    }

    // Equations for the k+1 terms:
    start_a += 2*p_complex;
    start_b += 2*p_complex;
    for (j = 0; j < p_real; ++j) {
      a = start_a + j;
      b = start_b;
      value = this->alpha_real_(j);
      ab.diagonal(b-a)(a) = value;
      ab.diagonal(a-b)(a) = value;

      if (k > 0) {
        a -= block_size_;
        b = start_b + 1 + j - block_size_;
        value = gamma_real(j);
        ab.diagonal(b-a)(a) = value;
        ab.diagonal(a-b)(a) = value;
      }
    }
    start_a += p_real;
    for (j = 0; j < p_complex; ++j) {
      a = start_a + 2*j;
      b = start_b;
      value = this->alpha_complex_(j);
      ab.diagonal(b-a)(a) = value;
      ab.diagonal(a-b)(a) = value;

      if (k > 0) {
        a -= block_size_;
        b = start_b + 1 + p_real + 2*j - block_size_;
        value = gamma_complex_real(j);
        ab.diagonal(b-a)(a) = value;
        ab.diagonal(a-b)(a) = value;

        b += 1;
        value = gamma_complex_imag(j);
        ab.diagonal(b-a)(a) = value;
        ab.diagonal(a-b)(a) = value;

        a += 1;
        b -= 1;
        ab.diagonal(b-a)(a) = value;
        ab.diagonal(a-b)(a) = value;

        b += 1;
        value = -gamma_complex_real(j);
        ab.diagonal(b-a)(a) = value;
        ab.diagonal(a-b)(a) = value;
      }
    }
  }

  /* std::cout << ab.toDenseMatrix() << std::endl; */

  // Build and factorize the sparse matrix.
  band_factorize(ab, ipiv_);
  factor_ = ab.coeffs();

  // Deal with negative values in the diagonal.
  Eigen::VectorXcd d = ab.diagonal().cast<std::complex<double> >();

  this->log_det_ = log(d.array()).real().sum();
}

void BandSolver::solve (const Eigen::MatrixXd& b, double* x) const {
  assert ((b.rows() == this->n_) && "DIMENSION_MISMATCH");
  size_t nrhs = b.cols();

  // Pad the input vector to the extended size.
  real_matrix_t bex = real_matrix_t::Zero(dim_ext_, nrhs);
  for (size_t j = 0; j < nrhs; ++j)
    for (size_t i = 0; i < this->n_; ++i)
      bex(i*block_size_, j) = b(i, j);

  // Solve the extended system.
  band_solve(block_size_, block_size_, factor_, ipiv_, bex);

  // Copy the output.
  for (size_t j = 0; j < nrhs; ++j)
    for (size_t i = 0; i < this->n_; ++i)
      x[i+j*nrhs] = bex(i*block_size_, j);
}

};

#endif
