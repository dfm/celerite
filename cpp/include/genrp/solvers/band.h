#ifndef _GENRP_SOLVER_BAND_
#define _GENRP_SOLVER_BAND_

#include <cmath>
#include <iostream>
#include <Eigen/Core>

#include "genrp/banded.h"
#include "genrp/solvers/direct.h"

namespace genrp {

#define BLOCKSIZE                                      \
  size_t p_real = this->p_real,                       \
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
  BandSolver (const Eigen::VectorXd alpha, const Eigen::VectorXcd beta)
    : DirectSolver(alpha, beta) {};
  BandSolver (const Eigen::VectorXd alphabreal, const Eigen::VectorXd betabreal,
              const Eigen::VectorXd alphabcomplex, const Eigen::VectorXcd betabcomplex)
    : DirectSolver(alphabreal, betabreal, alphabcomplex, betabcomplex) {};
  BandSolver (size_t p, const double* alpha, const double* beta)
    : DirectSolver(p, alpha, beta) {};
  BandSolver (size_t p, const double* alpha, const std::complex<double>* beta)
    : DirectSolver(p, alpha, beta) {};
  BandSolver (size_t p_real, const double* alphabreal, const double* betabreal,
              size_t p_complex, const double* alphabcomplex, const std::complex<double>* betabcomplex)
    : DirectSolver(p_real, alphabreal, betabreal, p_complex, alphabcomplex, betabcomplex) {};

  void compute (const Eigen::VectorXd& x, const Eigen::VectorXd& diag);
  void solve (const Eigen::MatrixXd& b, double* x) const;

  size_t dim_ext () const { BLOCKSIZE return dim_ext; };
  void solve_extended (Eigen::MatrixXd& b) const;

  // Needed for the Eigen-free interface.
  using DirectSolver::compute;
  using DirectSolver::solve;

private:
  Eigen::MatrixXd ab, al;
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
    alpha_
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
  const Eigen::Matrix<T, Eigen::Dynamic, 1>& diag,
  Eigen::Matrix<T, Eigen::Dynamic, 1>& ab,
  Eigen::Matrix<T, Eigen::Dynamic, 1>& al,
  Eigen::VectorXi& ipiv,
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
  T sum_alpha = this->alphabreal.sum() + this->alphabcomplex_.sum();
  for (k = 0; k < n_; ++k)
    get_band_element(ab, width, 0, k*block_size) = diag(k) + sum_alpha;

  // Fill in all but the last block.
  int block_id, start_a, start_b, a, b;
  double dt;
  T value;
  Eigen::ArrayXd ebt, phi, gammabreal(p_real), gammabcomplex_real(p_complex),
                 gammabcomplex_imag(p_complex);
  for (k = 0; k < n_ - 1; ++k) {
    // Pre-compute the gammas.
    dt = x(k+1) - x(k);
    gammabreal = exp(-this->betabreal.array() * dt);
    ebt = exp(-this->betabcomplex_.real().array() * dt);
    phi = this->betabcomplex_.imag().array() * dt;
    gammabcomplex_real = ebt * cos(phi);
    gammabcomplex_imag = -ebt * sin(phi);

    // Equations for the rs:
    block_id = block_size * k;
    start_b = block_id + 1;
    for (j = 0; j < p_real; ++j) {
      a = block_id;
      b = start_b + j;
      value = gammabreal(j);
      get_band_element(ab, width, b-a, a) = value;
      get_band_element(ab, width, a-b, a) = value;
    }
    start_b += p_real;
    for (j = 0; j < p_complex; ++j) {
      a = block_id;
      b = start_b + 2*j;
      value = 2.0 * gammabcomplex_real(j);
      get_band_element(ab, width, b-a, a) = value;
      get_band_element(ab, width, a-b, a) = value;

      b = start_b + 2*j + 1;
      value = 2.0 * gammabcomplex_imag(j);
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
      value = this->alphabreal(j);
      get_band_element(ab, width, b-a, a) = value;
      get_band_element(ab, width, a-b, a) = value;

      if (k > 0) {
        a -= block_size;
        b = start_b + 1 + j - block_size;
        value = gammabreal(j);
        get_band_element(ab, width, b-a, a) = value;
        get_band_element(ab, width, a-b, a) = value;
      }
    }
    start_a += p_real;
    for (j = 0; j < p_complex; ++j) {
      a = start_a + 2*j;
      b = start_b;
      value = 0.5 * this->alphabcomplex_(j);
      get_band_element(ab, width, b-a, a) = value;
      get_band_element(ab, width, a-b, a) = value;

      if (k > 0) {
        a -= block_size;
        b = start_b + 1 + p_real + 2*j - block_size;
        value = gammabcomplex_real(j);
        get_band_element(ab, width, b-a, a) = value;
        get_band_element(ab, width, a-b, a) = value;

        b += 1;
        value = gammabcomplex_imag(j);
        get_band_element(ab, width, b-a, a) = value;
        get_band_element(ab, width, a-b, a) = value;

        a += 1;
        b -= 1;
        get_band_element(ab, width, b-a, a) = value;
        get_band_element(ab, width, a-b, a) = value;

        b += 1;
        value = -gammabcomplex_real(j);
        get_band_element(ab, width, b-a, a) = value;
        get_band_element(ab, width, a-b, a) = value;
      }
    }
  }

  // Build and factorize the sparse matrix.
  int nothing;
  bandec<double>(ab.data(), dim_ext, width, width, al.data(), ipiv.data(), &nothing);

  // Deal with negative values in the diagonal.
  Eigen::VectorXcd d = ab.row(0).cast<std::complex<double> >();

  return log(d.array()).real().sum();
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
