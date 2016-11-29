#ifndef _GENRP_SOLVER_BAND_H_
#define _GENRP_SOLVER_BAND_H_

#include <cmath>
#include <iostream>
#include <Eigen/Core>

#include "genrp/banded.h"
#include "genrp/solvers/solver.h"

namespace genrp {

#define BLOCKSIZE                                      \
  size_t p_real = this->p_real_,                       \
         p_complex = this->p_complex_,                 \
         n = this->n_,                                 \
         width = p_real + 2 * p_complex + 2,           \
         block_size = 2 * p_real + 4 * p_complex + 1,  \
         dim_ext = block_size * (n - 1) + 1;           \
  if (p_complex == 0) width = p_real + 1;

template <typename T>
class BandSolver : public Solver<T> {
public:
  BandSolver () {};

  void compute (
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

  size_t dim_ext () const {
    BLOCKSIZE
    return dim_ext;
  };

  // Needed for the Eigen-free interface.
  using Solver<T>::compute;
  using Solver<T>::solve;

private:
  Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> a_, al_;
  Eigen::VectorXi ipiv_;

};

// Function for working with band matrices.
template <typename T>
inline T& get_band_element (Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>& m, int width, int i, int j) {
  return m(width - i, std::max(0, i) + j);
}

template <typename T>
void BandSolver<T>::compute (
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
  using std::abs;

  // Check the dimensions
  assert ((alpha_real.rows() == beta_real.rows()) && "DIMENSION_MISMATCH");
  assert ((alpha_complex_real.rows() == alpha_complex_imag.rows()) && "DIMENSION_MISMATCH");
  assert ((alpha_complex_real.rows() == beta_complex_real.rows()) && "DIMENSION_MISMATCH");
  assert ((alpha_complex_real.rows() == beta_complex_imag.rows()) && "DIMENSION_MISMATCH");
  assert ((x.rows() == diag.rows()) && "DIMENSION_MISMATCH");

  T ar = (alpha_complex_real.array() * beta_complex_real.array()).sum(),
    ai = (alpha_complex_imag.array() * beta_complex_imag.array()).sum();
  assert ((ar >= ai) && "INVALID PARAMETERS");

  // Save the dimensions for later use
  this->p_real_ = alpha_real.rows();
  this->p_complex_ = alpha_complex_real.rows();
  this->n_ = x.rows();

  // Dimensions.
  size_t j, k;

  BLOCKSIZE

  // Set up the extended matrix.
  a_.resize(1+2*width, dim_ext);
  a_.setConstant(T(0.0));
  al_.resize(width, dim_ext);
  ipiv_.resize(dim_ext);

  // Start with the diagonal.
  T sum_alpha = alpha_real.sum() + alpha_complex_real.sum() + alpha_complex_imag.sum();
  for (k = 0; k < n; ++k)
    get_band_element(a_, width, 0, k*block_size) = diag(k) + sum_alpha;

  // Fill in all but the last block.
  int block_id, start_a, start_b, a, b;
  double dt;
  T value;
  Eigen::Array<T, Eigen::Dynamic, 1> ebt, phi,
    gamma_real(p_real), gamma_complex_real(p_complex), gamma_complex_imag(p_complex);
  for (k = 0; k < n - 1; ++k) {
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
      get_band_element(a_, width, b-a, a) = value;
      get_band_element(a_, width, a-b, a) = value;
    }
    start_b += p_real;
    for (j = 0; j < p_complex; ++j) {
      a = block_id;
      b = start_b + 2*j;
      value = 2.0 * gamma_complex_real(j);
      get_band_element(a_, width, b-a, a) = value;
      get_band_element(a_, width, a-b, a) = value;

      b = start_b + 2*j + 1;
      value = 2.0 * gamma_complex_imag(j);
      get_band_element(a_, width, b-a, a) = value;
      get_band_element(a_, width, a-b, a) = value;
    }

    // Equations for the ls:
    start_a = block_id + 1;
    start_b += 2*p_complex;
    for (j = 0; j < p_real; ++j) {
      a = start_a + j;
      b = start_b + j;
      value = -1.0;
      get_band_element(a_, width, b-a, a) = value;
      get_band_element(a_, width, a-b, a) = value;
    }
    start_a += p_real;
    start_b += p_real;
    for (j = 0; j < p_complex; ++j) {
      a = start_a + 2*j;
      b = start_b + 2*j;
      value = -1.0;
      get_band_element(a_, width, b-a, a) = value;
      get_band_element(a_, width, a-b, a) = value;

      a += 1;
      b += 1;
      value = 1.0;
      get_band_element(a_, width, b-a, a) = value;
      get_band_element(a_, width, a-b, a) = value;
    }

    // Equations for the k+1 terms:
    start_a += 2*p_complex;
    start_b += 2*p_complex;
    for (j = 0; j < p_real; ++j) {
      a = start_a + j;
      b = start_b;
      value = alpha_real(j);
      get_band_element(a_, width, b-a, a) = value;
      get_band_element(a_, width, a-b, a) = value;

      if (k > 0) {
        a -= block_size;
        b = start_b + 1 + j - block_size;
        value = gamma_real(j);
        get_band_element(a_, width, b-a, a) = value;
        get_band_element(a_, width, a-b, a) = value;
      }
    }
    start_a += p_real;
    for (j = 0; j < p_complex; ++j) {
      a = start_a + 2*j;
      b = start_b;
      value = 0.5 * alpha_complex_real(j);
      get_band_element(a_, width, b-a, a) = value;
      get_band_element(a_, width, a-b, a) = value;

      a += 1;
      value = 0.5 * alpha_complex_imag(j);
      get_band_element(a_, width, b-a, a) = value;
      get_band_element(a_, width, a-b, a) = value;
      a -= 1;

      if (k > 0) {
        a -= block_size;
        b = start_b + 1 + p_real + 2*j - block_size;
        value = gamma_complex_real(j);
        get_band_element(a_, width, b-a, a) = value;
        get_band_element(a_, width, a-b, a) = value;

        b += 1;
        value = gamma_complex_imag(j);
        get_band_element(a_, width, b-a, a) = value;
        get_band_element(a_, width, a-b, a) = value;

        a += 1;
        b -= 1;
        get_band_element(a_, width, b-a, a) = value;
        get_band_element(a_, width, a-b, a) = value;

        b += 1;
        value = -gamma_complex_real(j);
        get_band_element(a_, width, b-a, a) = value;
        get_band_element(a_, width, a-b, a) = value;
      }
    }
  }

  // Build and factorize the sparse matrix.
  int nothing;
  bandec<T>(a_.data(), dim_ext, width, width, al_.data(), ipiv_.data(), &nothing);

  // Compute the determinant.
  T ld = T(0.0);
  for (size_t i = 0; i < dim_ext; ++i) ld += log(abs(a_(0, i)));
  this->log_det_ = ld;
}

template <typename T>
void BandSolver<T>::solve (const Eigen::MatrixXd& b, T* x) const {
  assert ((b.rows() == this->n_) && "DIMENSION_MISMATCH");
  size_t nrhs = b.cols();

  BLOCKSIZE

  // Pad the input vector to the extended size.
  Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> bex = Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>::Zero(dim_ext, nrhs);
  for (size_t j = 0; j < nrhs; ++j)
    for (size_t i = 0; i < this->n_; ++i)
      bex(i*block_size, j) = T(b(i, j));

  // Solve the extended system.
  for (size_t i = 0; i < nrhs; ++i)
    banbks<T>(a_.data(), dim_ext, width, width, al_.data(), ipiv_.data(), bex.col(i).data());

  // Copy the output.
  for (size_t j = 0; j < nrhs; ++j)
    for (size_t i = 0; i < this->n_; ++i)
      x[i+j*nrhs] = bex(i*block_size, j);
}

#undef BLOCKSIZE

};

#endif
