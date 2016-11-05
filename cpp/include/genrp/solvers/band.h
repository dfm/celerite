#ifndef _GENRP_SOLVER_BAND_
#define _GENRP_SOLVER_BAND_

#include <cmath>
#include <iostream>
#include <Eigen/Core>

#include "genrp/banded.h"
#include "genrp/solvers/direct.h"

namespace genrp {

class BandSolver : public DirectSolver {
public:
  BandSolver () : DirectSolver() {};
  BandSolver (const Eigen::VectorXd alpha, const Eigen::VectorXd beta)
    : DirectSolver(alpha, beta) {};
  BandSolver (const Eigen::VectorXd alpha, const Eigen::VectorXcd beta)
    : DirectSolver(alpha, beta) {};
  BandSolver (const Eigen::VectorXd alpha_real, const Eigen::VectorXd beta_real,
              const Eigen::VectorXd alpha_complex, const Eigen::VectorXcd beta_complex)
    : DirectSolver(alpha_real, beta_real, alpha_complex, beta_complex) {};
  BandSolver (size_t p, const double* alpha, const double* beta)
    : DirectSolver(p, alpha, beta) {};
  BandSolver (size_t p, const double* alpha, const std::complex<double>* beta)
    : DirectSolver(p, alpha, beta) {};
  BandSolver (size_t p_real, const double* alpha_real, const double* beta_real,
              size_t p_complex, const double* alpha_complex, const std::complex<double>* beta_complex)
    : DirectSolver(p_real, alpha_real, beta_real, p_complex, alpha_complex, beta_complex) {};

  void compute (const Eigen::VectorXd& x, const Eigen::VectorXd& diag);
  void solve (const Eigen::MatrixXd& b, double* x) const;

  size_t dim_ext () const { return dim_ext_; };
  void solve_extended (Eigen::MatrixXd& b) const;

  // Needed for the Eigen-free interface.
  using DirectSolver::compute;
  using DirectSolver::solve;

private:
  size_t width_, block_size_, dim_ext_;

  Eigen::MatrixXd a_, al_;
  Eigen::VectorXi ipiv_;

};

// Function for working with band matrices.
inline double& get_band_element (Eigen::MatrixXd& m, int width, int i, int j) {
  return m(width - i, std::max(0, i) + j);
}

void BandSolver::compute (const Eigen::VectorXd& x, const Eigen::VectorXd& diag) {
  // Check dimensions.
  assert ((x.rows() == diag.rows()) && "DIMENSION_MISMATCH");
  this->n_ = x.rows();

  // Dimensions.
  size_t p_real = this->p_real_,
         p_complex = this->p_complex_,
         n = this->n_,
         j, k;

  // Compute the block sizes.
  width_ = p_real + 2 * p_complex + 2;
  if (p_complex == 0) width_ = p_real + 1;
  block_size_ = 2 * p_real + 4 * p_complex + 1;
  dim_ext_ = block_size_ * (n - 1) + 1;

  // Set up the extended matrix.
  a_.resize(1+2*width_, dim_ext_);
  a_.setConstant(0.0);
  al_.resize(width_, dim_ext_);
  ipiv_.resize(dim_ext_);

  // Start with the diagonal.
  double sum_alpha = this->alpha_real_.sum() + this->alpha_complex_.sum();
  for (k = 0; k < n_; ++k)
    get_band_element(a_, width_, 0, k*block_size_) = diag(k) + sum_alpha;

  // Fill in all but the last block.
  int block_id, start_a, start_b, a, b;
  double dt, value;
  Eigen::ArrayXd ebt, phi, gamma_real(p_real), gamma_complex_real(p_complex),
                 gamma_complex_imag(p_complex);
  for (k = 0; k < n_ - 1; ++k) {
    // Pre-compute the gammas.
    dt = x(k+1) - x(k);
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
      get_band_element(a_, width_, b-a, a) = value;
      get_band_element(a_, width_, a-b, a) = value;
    }
    start_b += p_real;
    for (j = 0; j < p_complex; ++j) {
      a = block_id;
      b = start_b + 2*j;
      value = 2.0 * gamma_complex_real(j);
      get_band_element(a_, width_, b-a, a) = value;
      get_band_element(a_, width_, a-b, a) = value;

      b = start_b + 2*j + 1;
      value = 2.0 * gamma_complex_imag(j);
      get_band_element(a_, width_, b-a, a) = value;
      get_band_element(a_, width_, a-b, a) = value;
    }

    // Equations for the ls:
    start_a = block_id + 1;
    start_b += 2*p_complex;
    for (j = 0; j < p_real; ++j) {
      a = start_a + j;
      b = start_b + j;
      value = -1.0;
      get_band_element(a_, width_, b-a, a) = value;
      get_band_element(a_, width_, a-b, a) = value;
    }
    start_a += p_real;
    start_b += p_real;
    for (j = 0; j < p_complex; ++j) {
      a = start_a + 2*j;
      b = start_b + 2*j;
      value = -1.0;
      get_band_element(a_, width_, b-a, a) = value;
      get_band_element(a_, width_, a-b, a) = value;

      a += 1;
      b += 1;
      value = 1.0;
      get_band_element(a_, width_, b-a, a) = value;
      get_band_element(a_, width_, a-b, a) = value;
    }

    // Equations for the k+1 terms:
    start_a += 2*p_complex;
    start_b += 2*p_complex;
    for (j = 0; j < p_real; ++j) {
      a = start_a + j;
      b = start_b;
      value = this->alpha_real_(j);
      get_band_element(a_, width_, b-a, a) = value;
      get_band_element(a_, width_, a-b, a) = value;

      if (k > 0) {
        a -= block_size_;
        b = start_b + 1 + j - block_size_;
        value = gamma_real(j);
        get_band_element(a_, width_, b-a, a) = value;
        get_band_element(a_, width_, a-b, a) = value;
      }
    }
    start_a += p_real;
    for (j = 0; j < p_complex; ++j) {
      a = start_a + 2*j;
      b = start_b;
      value = 0.5 * this->alpha_complex_(j);
      get_band_element(a_, width_, b-a, a) = value;
      get_band_element(a_, width_, a-b, a) = value;

      if (k > 0) {
        a -= block_size_;
        b = start_b + 1 + p_real + 2*j - block_size_;
        value = gamma_complex_real(j);
        get_band_element(a_, width_, b-a, a) = value;
        get_band_element(a_, width_, a-b, a) = value;

        b += 1;
        value = gamma_complex_imag(j);
        get_band_element(a_, width_, b-a, a) = value;
        get_band_element(a_, width_, a-b, a) = value;

        a += 1;
        b -= 1;
        get_band_element(a_, width_, b-a, a) = value;
        get_band_element(a_, width_, a-b, a) = value;

        b += 1;
        value = -gamma_complex_real(j);
        get_band_element(a_, width_, b-a, a) = value;
        get_band_element(a_, width_, a-b, a) = value;
      }
    }
  }

  // Build and factorize the sparse matrix.
  int nothing;
  bandec<double>(a_.data(), dim_ext_, width_, width_, al_.data(), ipiv_.data(), &nothing);

  // Deal with negative values in the diagonal.
  Eigen::VectorXcd d = a_.row(0).cast<std::complex<double> >();

  this->log_det_ = log(d.array()).real().sum();
}

void BandSolver::solve (const Eigen::MatrixXd& b, double* x) const {
  assert ((b.rows() == this->n_) && "DIMENSION_MISMATCH");
  size_t nrhs = b.cols();

  // Pad the input vector to the extended size.
  Eigen::MatrixXd bex = Eigen::MatrixXd::Zero(dim_ext_, nrhs);
  for (size_t j = 0; j < nrhs; ++j)
    for (size_t i = 0; i < this->n_; ++i)
      bex(i*block_size_, j) = b(i, j);

  // Solve the extended system.
  for (size_t i = 0; i < nrhs; ++i)
    banbks<double>(a_.data(), dim_ext_, width_, width_, al_.data(), ipiv_.data(), bex.col(i).data());

  // Copy the output.
  for (size_t j = 0; j < nrhs; ++j)
    for (size_t i = 0; i < this->n_; ++i)
      x[i+j*nrhs] = bex(i*block_size_, j);
}

void BandSolver::solve_extended (Eigen::MatrixXd& b) const {
  assert ((b.rows() == this->dim_ext_) && "DIMENSION_MISMATCH");
  size_t nrhs = b.cols();
  for (size_t i = 0; i < nrhs; ++i)
    banbks<double>(a_.data(), dim_ext_, width_, width_, al_.data(), ipiv_.data(), b.col(i).data());
}

};

#endif
