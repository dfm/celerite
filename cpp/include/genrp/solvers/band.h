#ifndef _GENRP_SOLVER_BAND_
#define _GENRP_SOLVER_BAND_

#include <cmath>
#include <iostream>
#include <Eigen/Core>

#include "genrp/banded.h"
#include "genrp/solvers/direct.h"

namespace genrp {

#define BLOCKSIZE                                      \
  size_t p_real = this->p_real_,                       \
         p_complex = this->p_complex_,                 \
         n = this->n_,                                 \
         width = p_real + 2 * p_complex + 2,           \
         block_size = 2 * p_real + 4 * p_complex + 1,  \
         dim_ext = block_size * (n - 1) + 1;           \
  if (p_complex == 0) width = p_real + 1;

class BandSolver : public DirectSolver {
public:
  BandSolver () : DirectSolver() {};
  BandSolver (const Eigen::VectorXd alpha, const Eigen::VectorXd beta)
    : DirectSolver(alpha, beta) {};
  BandSolver (const Eigen::VectorXd alpha, const Eigen::VectorXd beta_real, const Eigen::VectorXd beta_imag)
    : DirectSolver(alpha, beta_real, beta_imag) {};
  BandSolver (const Eigen::VectorXd alpha_real,
              const Eigen::VectorXd beta_real,
              const Eigen::VectorXd alpha_complex,
              const Eigen::VectorXd beta_complex_real,
              const Eigen::VectorXd beta_complex_imag)
    : DirectSolver(alpha_real, beta_real, alpha_complex, beta_complex_real, beta_complex_imag) {};
  BandSolver (size_t p, const double* alpha, const double* beta)
    : DirectSolver(p, alpha, beta) {};
  BandSolver (size_t p, const double* alpha, const double* beta_real, const double* beta_imag)
    : DirectSolver(p, alpha, beta_real, beta_imag) {};
  BandSolver (size_t p_real, const double* alpha_real, const double* beta_real,
              size_t p_complex, const double* alpha_complex, const double* beta_complex_real, const double* beta_complex_imag)
    : DirectSolver(p_real, alpha_real, beta_real, p_complex, alpha_complex, beta_complex_real, beta_complex_imag) {};

  void compute (const Eigen::VectorXd& x, const Eigen::VectorXd& diag);
  void solve (const Eigen::MatrixXd& b, double* x) const;

  size_t dim_ext () const {
    BLOCKSIZE
    return dim_ext;
  };
  void solve_extended (Eigen::MatrixXd& b) const;

  template <typename T>
  T build_matrix (
    const Eigen::Matrix<T, Eigen::Dynamic, 1>& alpha_real,
    const Eigen::Matrix<T, Eigen::Dynamic, 1>& beta_real,
    const Eigen::Matrix<T, Eigen::Dynamic, 1>& alpha_complex,
    const Eigen::Matrix<T, Eigen::Dynamic, 1>& beta_complex_real,
    const Eigen::Matrix<T, Eigen::Dynamic, 1>& beta_complex_imag,
    const Eigen::VectorXd& x,
    const Eigen::VectorXd& diag,
    Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>& ab,
    Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>& al,
    Eigen::VectorXi& ipiv
  );

  // Needed for the Eigen-free interface.
  using DirectSolver::compute;
  using DirectSolver::solve;

private:
  Eigen::MatrixXd a_, al_;
  Eigen::VectorXi ipiv_;

};

// Function for working with band matrices.
template <typename T>
inline T& get_band_element (Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>& m, int width, int i, int j) {
  return m(width - i, std::max(0, i) + j);
}

void BandSolver::compute (const Eigen::VectorXd& x, const Eigen::VectorXd& diag)
{
  this->log_det_ = this->build_matrix<double>(
    alpha_real_,
    beta_real_,
    alpha_complex_,
    beta_complex_real_,
    beta_complex_imag_,
    x,
    diag,
    a_,
    al_,
    ipiv_
  );
}

template <typename T>
T BandSolver::build_matrix (
  const Eigen::Matrix<T, Eigen::Dynamic, 1>& alpha_real,
  const Eigen::Matrix<T, Eigen::Dynamic, 1>& beta_real,
  const Eigen::Matrix<T, Eigen::Dynamic, 1>& alpha_complex,
  const Eigen::Matrix<T, Eigen::Dynamic, 1>& beta_complex_real,
  const Eigen::Matrix<T, Eigen::Dynamic, 1>& beta_complex_imag,
  const Eigen::VectorXd& x,
  const Eigen::VectorXd& diag,
  Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>& ab,
  Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>& al,
  Eigen::VectorXi& ipiv
) {
  // Check dimensions.
  assert ((x.rows() == diag.rows()) && "DIMENSION_MISMATCH");
  this->n_ = x.rows();

  // Dimensions.
  size_t j, k;

  BLOCKSIZE

  // Set up the extended matrix.
  ab.resize(1+2*width, dim_ext);
  ab.setConstant(T(0.0));
  al.resize(width, dim_ext);
  ipiv.resize(dim_ext);

  // Start with the diagonal.
  T sum_alpha = alpha_real.sum() + alpha_complex.sum();
  for (k = 0; k < n_; ++k)
    get_band_element(ab, width, 0, k*block_size) = diag(k) + sum_alpha;

  // Fill in all but the last block.
  int block_id, start_a, start_b, a, b;
  double dt;
  T value;
  Eigen::ArrayXd ebt, phi, gamma_real(p_real), gamma_complex_real(p_complex),
                 gamma_complex_imag(p_complex);
  for (k = 0; k < n_ - 1; ++k) {
    // Pre-compute the gammas.
    dt = x(k+1) - x(k);
    gamma_real = exp(-beta_real.array() * dt);
    ebt = exp(-beta_complex_real.array() * dt);
    phi = beta_complex_imag.array() * dt;
    gamma_complex_real = ebt * cos(phi);
    gamma_complex_imag = -ebt * sin(phi);

    // Equations for the rs:
    block_id = block_size * k;
    start_b = block_id + 1;
    for (j = 0; j < p_real; ++j) {
      a = block_id;
      b = start_b + j;
      value = gamma_real(j);
      get_band_element(ab, width, b-a, a) = value;
      get_band_element(ab, width, a-b, a) = value;
    }
    start_b += p_real;
    for (j = 0; j < p_complex; ++j) {
      a = block_id;
      b = start_b + 2*j;
      value = 2.0 * gamma_complex_real(j);
      get_band_element(ab, width, b-a, a) = value;
      get_band_element(ab, width, a-b, a) = value;

      b = start_b + 2*j + 1;
      value = 2.0 * gamma_complex_imag(j);
      get_band_element(ab, width, b-a, a) = value;
      get_band_element(ab, width, a-b, a) = value;
    }

    // Equations for the ls:
    start_a = block_id + 1;
    start_b += 2*p_complex;
    for (j = 0; j < p_real; ++j) {
      a = start_a + j;
      b = start_b + j;
      value = -1.0;
      get_band_element(ab, width, b-a, a) = value;
      get_band_element(ab, width, a-b, a) = value;
    }
    start_a += p_real;
    start_b += p_real;
    for (j = 0; j < p_complex; ++j) {
      a = start_a + 2*j;
      b = start_b + 2*j;
      value = -1.0;
      get_band_element(ab, width, b-a, a) = value;
      get_band_element(ab, width, a-b, a) = value;

      a += 1;
      b += 1;
      value = 1.0;
      get_band_element(ab, width, b-a, a) = value;
      get_band_element(ab, width, a-b, a) = value;
    }

    // Equations for the k+1 terms:
    start_a += 2*p_complex;
    start_b += 2*p_complex;
    for (j = 0; j < p_real; ++j) {
      a = start_a + j;
      b = start_b;
      value = alpha_real(j);
      get_band_element(ab, width, b-a, a) = value;
      get_band_element(ab, width, a-b, a) = value;

      if (k > 0) {
        a -= block_size;
        b = start_b + 1 + j - block_size;
        value = gamma_real(j);
        get_band_element(ab, width, b-a, a) = value;
        get_band_element(ab, width, a-b, a) = value;
      }
    }
    start_a += p_real;
    for (j = 0; j < p_complex; ++j) {
      a = start_a + 2*j;
      b = start_b;
      value = 0.5 * alpha_complex_(j);
      get_band_element(ab, width, b-a, a) = value;
      get_band_element(ab, width, a-b, a) = value;

      if (k > 0) {
        a -= block_size;
        b = start_b + 1 + p_real + 2*j - block_size;
        value = gamma_complex_real(j);
        get_band_element(ab, width, b-a, a) = value;
        get_band_element(ab, width, a-b, a) = value;

        b += 1;
        value = gamma_complex_imag(j);
        get_band_element(ab, width, b-a, a) = value;
        get_band_element(ab, width, a-b, a) = value;

        a += 1;
        b -= 1;
        get_band_element(ab, width, b-a, a) = value;
        get_band_element(ab, width, a-b, a) = value;

        b += 1;
        value = -gamma_complex_real(j);
        get_band_element(ab, width, b-a, a) = value;
        get_band_element(ab, width, a-b, a) = value;
      }
    }
  }

  // Build and factorize the sparse matrix.
  int nothing;
  bandec<double>(ab.data(), dim_ext, width, width, al.data(), ipiv.data(), &nothing);

  // Compute the determinant.
  T ld = T(0.0);
  for (size_t i = 0; i < dim_ext; ++i) if (ab(0, i) > T(0.0)) ld += log(ab(0, i));

  return ld;
}

void BandSolver::solve (const Eigen::MatrixXd& b, double* x) const {
  assert ((b.rows() == this->n_) && "DIMENSION_MISMATCH");
  size_t nrhs = b.cols();

  BLOCKSIZE

  // Pad the input vector to the extended size.
  Eigen::MatrixXd bex = Eigen::MatrixXd::Zero(dim_ext, nrhs);
  for (size_t j = 0; j < nrhs; ++j)
    for (size_t i = 0; i < this->n_; ++i)
      bex(i*block_size, j) = b(i, j);

  // Solve the extended system.
  for (size_t i = 0; i < nrhs; ++i)
    banbks<double>(a_.data(), dim_ext, width, width, al_.data(), ipiv_.data(), bex.col(i).data());

  // Copy the output.
  for (size_t j = 0; j < nrhs; ++j)
    for (size_t i = 0; i < this->n_; ++i)
      x[i+j*nrhs] = bex(i*block_size, j);
}

void BandSolver::solve_extended (Eigen::MatrixXd& b) const {
  assert ((b.rows() == this->dim_ext) && "DIMENSION_MISMATCH");
  size_t nrhs = b.cols();

  BLOCKSIZE

  for (size_t i = 0; i < nrhs; ++i)
    banbks<double>(a_.data(), dim_ext, width, width, al_.data(), ipiv_.data(), b.col(i).data());
}

#undef BLOCKSIZE

};

#endif
