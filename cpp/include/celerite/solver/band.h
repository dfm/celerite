#ifndef _CELERITE_SOLVER_BAND_H_
#define _CELERITE_SOLVER_BAND_H_

#include <cmath>
#include <Eigen/Core>

#include "celerite/utils.h"
#include "celerite/exceptions.h"
#include "celerite/lapack.h"
#include "celerite/banded.h"
#include "celerite/solver/solver.h"

namespace celerite {
namespace solver {

#define BLOCKSIZE_BASE                              \
  int width = p_real + 2 * p_complex + 2,           \
      block_size = 2 * p_real + 4 * p_complex + 1,  \
      dim_ext = block_size * (n - 1) + 1;           \
  if (p_complex == 0) width = p_real + 1;

#define BLOCKSIZE                                   \
  int p_real = this->p_real_,                       \
      p_complex = this->p_complex_,                 \
      n = this->n_;                                 \
  BLOCKSIZE_BASE

/// The class provides all of the computation power for the fast GP solver
///
/// The `celerite::solver::BandSolver::compute` method must be called before
/// most of the other methods.
template <typename T>
class BandSolver : public Solver<T> {
public:
  BandSolver (bool use_lapack = false) : Solver<T>(), use_lapack_(use_lapack) {};
  ~BandSolver () {};

  void build_matrix (
    const Eigen::Matrix<T, Eigen::Dynamic, 1>& alpha_real,
    const Eigen::Matrix<T, Eigen::Dynamic, 1>& beta_real,
    const Eigen::Matrix<T, Eigen::Dynamic, 1>& alpha_complex_real,
    const Eigen::Matrix<T, Eigen::Dynamic, 1>& alpha_complex_imag,
    const Eigen::Matrix<T, Eigen::Dynamic, 1>& beta_complex_real,
    const Eigen::Matrix<T, Eigen::Dynamic, 1>& beta_complex_imag,
    int offset_factor,
    const Eigen::VectorXd& x,
    Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>& A
  );

  /// Compute the extended matrix and perform the banded LU decomposition
  ///
  /// @param alpha_real The coefficients of the real terms.
  /// @param beta_real The exponents of the real terms.
  /// @param alpha_complex_real The real part of the coefficients of the complex terms.
  /// @param alpha_complex_imag The imaginary part of the of the complex terms.
  /// @param beta_complex_real The real part of the exponents of the complex terms.
  /// @param beta_complex_imag The imaginary part of the exponents of the complex terms.
  /// @param x The _sorted_ array of input coordinates.
  /// @param diag An array that should be added to the diagonal of the matrix.
  ///             This often corresponds to measurement uncertainties and in that case,
  ///             ``diag`` should be the measurement _variance_ (i.e. sigma^2).
  ///
  /// @return ``0`` on success. ``1`` for mismatched dimensions.
  int compute (
    const Eigen::Matrix<T, Eigen::Dynamic, 1>& alpha_real,
    const Eigen::Matrix<T, Eigen::Dynamic, 1>& beta_real,
    const Eigen::Matrix<T, Eigen::Dynamic, 1>& alpha_complex_real,
    const Eigen::Matrix<T, Eigen::Dynamic, 1>& alpha_complex_imag,
    const Eigen::Matrix<T, Eigen::Dynamic, 1>& beta_complex_real,
    const Eigen::Matrix<T, Eigen::Dynamic, 1>& beta_complex_imag,
    const Eigen::VectorXd& x,
    const Eigen::Matrix<T, Eigen::Dynamic, 1>& diag
  );

  /// Solve a linear system for the matrix defined in ``compute``
  ///
  /// A previous call to `solver.Solver.compute` defines a matrix ``A``
  /// and this method solves for ``x`` in the matrix equation ``A.x = b``.
  ///
  /// @param b The right hand side of the linear system.
  /// @param x A pointer that will be overwritten with the result.
  void solve (const Eigen::MatrixXd& b, T* x) const;

  /// Compute the dot product of a ``celerite`` matrix and another arbitrary matrix
  ///
  /// This method computes ``A.b`` where ``A`` is defined by the parameters and
  /// ``b`` is an arbitrary matrix of the correct shape.
  ///
  /// @param alpha_real The coefficients of the real terms.
  /// @param beta_real The exponents of the real terms.
  /// @param alpha_complex_real The real part of the coefficients of the complex terms.
  /// @param alpha_complex_imag The imaginary part of the of the complex terms.
  /// @param beta_complex_real The real part of the exponents of the complex terms.
  /// @param beta_complex_imag The imaginary part of the exponents of the complex terms.
  /// @param x The _sorted_ array of input coordinates.
  /// @param b_in The matrix ``b`` described above.
  ///
  /// @return The matrix ``A.b``.
  Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> dot (
    const Eigen::Matrix<T, Eigen::Dynamic, 1>& alpha_real,
    const Eigen::Matrix<T, Eigen::Dynamic, 1>& beta_real,
    const Eigen::Matrix<T, Eigen::Dynamic, 1>& alpha_complex_real,
    const Eigen::Matrix<T, Eigen::Dynamic, 1>& alpha_complex_imag,
    const Eigen::Matrix<T, Eigen::Dynamic, 1>& beta_complex_real,
    const Eigen::Matrix<T, Eigen::Dynamic, 1>& beta_complex_imag,
    const Eigen::VectorXd& x,
    const Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>& b_in
  );

  size_t dim_ext () const {
    BLOCKSIZE
    return dim_ext;
  };

  // Needed for the Eigen-free interface.
  using Solver<T>::compute;
  using Solver<T>::solve;

protected:
  bool use_lapack_;
  Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> a_;
  Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> al_;
  Eigen::VectorXi ipiv_;

};

// Function for working with band matrices.
template <typename T>
inline T& get_band_element (Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>& m, int offset, int i, int j) {
  return m(offset - i, std::max(0, i) + j);
}

template <typename T>
void BandSolver<T>::build_matrix (
  const Eigen::Matrix<T, Eigen::Dynamic, 1>& alpha_real,
  const Eigen::Matrix<T, Eigen::Dynamic, 1>& beta_real,
  const Eigen::Matrix<T, Eigen::Dynamic, 1>& alpha_complex_real,
  const Eigen::Matrix<T, Eigen::Dynamic, 1>& alpha_complex_imag,
  const Eigen::Matrix<T, Eigen::Dynamic, 1>& beta_complex_real,
  const Eigen::Matrix<T, Eigen::Dynamic, 1>& beta_complex_imag,
  int offset_factor,
  const Eigen::VectorXd& x,
  Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>& A
) {
  // Dimensions.
  int j, k;
  int p_real = alpha_real.rows(),
      p_complex = alpha_complex_real.rows(),
      n = x.rows();
  BLOCKSIZE_BASE

  int offset = offset_factor * width;

  // Set up the extended matrix.
  A.resize(1+width+offset, dim_ext);
  A.setConstant(T(0.0));

  // Start with the diagonal.
  T sum_alpha = alpha_real.sum() + alpha_complex_real.sum();
  for (k = 0; k < n; ++k)
    get_band_element(A, offset, 0, k*block_size) = sum_alpha;

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
      get_band_element(A, offset, b-a, a) = value;
      get_band_element(A, offset, a-b, a) = value;
    }
    start_b += p_real;
    for (j = 0; j < p_complex; ++j) {
      a = block_id;
      b = start_b + 2*j;
      value = gamma_complex_real(j);
      get_band_element(A, offset, b-a, a) = value;
      get_band_element(A, offset, a-b, a) = value;

      b = start_b + 2*j + 1;
      value = gamma_complex_imag(j);
      get_band_element(A, offset, b-a, a) = value;
      get_band_element(A, offset, a-b, a) = value;
    }

    // Equations for the ls:
    start_a = block_id + 1;
    start_b += 2*p_complex;
    for (j = 0; j < p_real; ++j) {
      a = start_a + j;
      b = start_b + j;
      value = -1.0;
      get_band_element(A, offset, b-a, a) = value;
      get_band_element(A, offset, a-b, a) = value;
    }
    start_a += p_real;
    start_b += p_real;
    for (j = 0; j < p_complex; ++j) {
      a = start_a + 2*j;
      b = start_b + 2*j;
      value = -1.0;
      get_band_element(A, offset, b-a, a) = value;
      get_band_element(A, offset, a-b, a) = value;

      a += 1;
      b += 1;
      value = 1.0;
      get_band_element(A, offset, b-a, a) = value;
      get_band_element(A, offset, a-b, a) = value;
    }

    // Equations for the k+1 terms:
    start_a += 2*p_complex;
    start_b += 2*p_complex;
    for (j = 0; j < p_real; ++j) {
      a = start_a + j;
      b = start_b;
      value = alpha_real(j);
      get_band_element(A, offset, b-a, a) = value;
      get_band_element(A, offset, a-b, a) = value;

      if (k > 0) {
        a -= block_size;
        b = start_b + 1 + j - block_size;
        value = gamma_real(j);
        get_band_element(A, offset, b-a, a) = value;
        get_band_element(A, offset, a-b, a) = value;
      }
    }
    start_a += p_real;
    for (j = 0; j < p_complex; ++j) {
      a = start_a + 2*j;
      b = start_b;
      value = alpha_complex_real(j);
      get_band_element(A, offset, b-a, a) = value;
      get_band_element(A, offset, a-b, a) = value;

      a += 1;
      value = alpha_complex_imag(j);
      get_band_element(A, offset, b-a, a) = value;
      get_band_element(A, offset, a-b, a) = value;
      a -= 1;

      if (k > 0) {
        a -= block_size;
        b = start_b + 1 + p_real + 2*j - block_size;
        value = gamma_complex_real(j);
        get_band_element(A, offset, b-a, a) = value;
        get_band_element(A, offset, a-b, a) = value;

        b += 1;
        value = gamma_complex_imag(j);
        get_band_element(A, offset, b-a, a) = value;
        get_band_element(A, offset, a-b, a) = value;

        a += 1;
        b -= 1;
        get_band_element(A, offset, b-a, a) = value;
        get_band_element(A, offset, a-b, a) = value;

        b += 1;
        value = -gamma_complex_real(j);
        get_band_element(A, offset, b-a, a) = value;
        get_band_element(A, offset, a-b, a) = value;
      }
    }
  }
};

template <typename T>
int BandSolver<T>::compute (
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

  this->computed_ = false;
  if (x.rows() != diag.rows()) return SOLVER_DIMENSION_MISMATCH;
  if (alpha_real.rows() != beta_real.rows()) return SOLVER_DIMENSION_MISMATCH;
  if (alpha_complex_real.rows() != alpha_complex_imag.rows()) return SOLVER_DIMENSION_MISMATCH;
  if (alpha_complex_real.rows() != beta_complex_real.rows()) return SOLVER_DIMENSION_MISMATCH;
  if (alpha_complex_real.rows() != beta_complex_imag.rows()) return SOLVER_DIMENSION_MISMATCH;

  // Save the dimensions for later use
  this->p_real_ = alpha_real.rows();
  this->p_complex_ = alpha_complex_real.rows();
  this->n_ = x.rows();
  BLOCKSIZE

  int offset_factor = 1;
#ifdef WITH_LAPACK
  if (use_lapack_) offset_factor = 2;
#else
  if (use_lapack_) throw no_lapack();
#endif

  // Build the extended matrix.
  build_matrix(alpha_real, beta_real, alpha_complex_real, alpha_complex_imag,
               beta_complex_real, beta_complex_imag, offset_factor, x, a_);

  // Add the diagonal component
  for (int k = 0; k < this->n_; ++k)
    get_band_element(a_, offset_factor*width, 0, k*block_size) += diag(k);

  // Reshape the working arrays
#ifdef WITH_LAPACK
  if (!use_lapack_)
#endif
  al_.resize(width, dim_ext);
  ipiv_.resize(dim_ext);

  // Factorize the sparse matrix
  int nothing;
#ifdef WITH_LAPACK
  if (use_lapack_)
    band_factorize(dim_ext, width, width, a_, ipiv_);
  else
#endif
  bandec<T>(a_.data(), dim_ext, width, width, al_.data(), ipiv_.data(), &nothing);

// Compute the determinant
  T ld = T(0.0);
#ifdef WITH_LAPACK
  if (use_lapack_)
    for (int i = 0; i < dim_ext; ++i) ld += log(abs(a_(2*width, i)));
  else
#endif
  for (int i = 0; i < dim_ext; ++i) ld += log(abs(a_(0, i)));

  this->log_det_ = ld;
  this->computed_ = true;

  return 0;
}

template <typename T>
void BandSolver<T>::solve (const Eigen::MatrixXd& b, T* x) const {
  if (b.rows() != this->n_) throw dimension_mismatch();
  if (!(this->computed_)) throw compute_exception();
  int nrhs = b.cols();

  BLOCKSIZE

  // Pad the input vector to the extended size.
  Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> bex = Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>::Zero(dim_ext, nrhs);
  for (int j = 0; j < nrhs; ++j)
    for (int i = 0; i < this->n_; ++i)
      bex(i*block_size, j) = T(b(i, j));

  // Solve the extended system.
#ifdef WITH_LAPACK
  if (use_lapack_)
    band_solve(width, width, a_, ipiv_, bex);
  else
#endif
  for (int i = 0; i < nrhs; ++i)
    banbks<T>(a_.data(), dim_ext, width, width, al_.data(), ipiv_.data(), bex.col(i).data());

  // Copy the output.
  for (int j = 0; j < nrhs; ++j)
    for (int i = 0; i < this->n_; ++i)
      x[i+j*this->n_] = bex(i*block_size, j);
}

template <typename T>
Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> BandSolver<T>::dot (
  const Eigen::Matrix<T, Eigen::Dynamic, 1>& alpha_real,
  const Eigen::Matrix<T, Eigen::Dynamic, 1>& beta_real,
  const Eigen::Matrix<T, Eigen::Dynamic, 1>& alpha_complex_real,
  const Eigen::Matrix<T, Eigen::Dynamic, 1>& alpha_complex_imag,
  const Eigen::Matrix<T, Eigen::Dynamic, 1>& beta_complex_real,
  const Eigen::Matrix<T, Eigen::Dynamic, 1>& beta_complex_imag,
  const Eigen::VectorXd& t,
  const Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>& b_in
) {
  if (b_in.rows() != t.rows()) throw dimension_mismatch();
  int nrhs = b_in.cols();

  int p_real = alpha_real.rows(),
      p_complex = alpha_complex_real.rows(),
      n = t.rows();
  BLOCKSIZE_BASE

  // Build the extended matrix
  Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> A;
  build_matrix(alpha_real, beta_real, alpha_complex_real, alpha_complex_imag,
               beta_complex_real, beta_complex_imag, 1, t, A);

  // Pad the input vector to the extended size.
  Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> bex = Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>::Zero(dim_ext, nrhs);

  int ind, strt;
  T phi, psi, tau;
  Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>
    gm1_real(p_real, nrhs),
    up1_real(p_real, nrhs),
    gm1_comp(p_complex, nrhs),
    hm1_comp(p_complex, nrhs),
    up1_comp(p_complex, nrhs),
    vp1_comp(p_complex, nrhs);
  gm1_real.setConstant(T(0.0));
  up1_real.setConstant(T(0.0));
  gm1_comp.setConstant(T(0.0));
  hm1_comp.setConstant(T(0.0));
  up1_comp.setConstant(T(0.0));
  vp1_comp.setConstant(T(0.0));

  for (int m = 0; m < n - 1; ++m) {
    bex.row(m*block_size) = b_in.row(m);

    // g
    tau = t(m+1) - t(m);
    strt = m*block_size + 1 + p_real + 2*p_complex;
    for (int j = 0; j < p_real; ++j) {
      phi = exp(-beta_real(j) * tau);
      ind = strt + j;
      bex.row(ind) = (gm1_real.row(j) + b_in.row(m)) * phi;
      gm1_real.row(j) = bex.row(ind);
    }

    strt += p_real;
    for (int j = 0; j < p_complex; ++j) {
      phi = exp(-beta_complex_real(j) * tau) * cos(beta_complex_imag(j) * tau);
      psi = -exp(-beta_complex_real(j) * tau) * sin(beta_complex_imag(j) * tau);
      ind = strt + 2*j;
      bex.row(ind) = gm1_comp.row(j) * phi + b_in.row(m) * phi + psi * hm1_comp.row(j);
      bex.row(ind+1) = hm1_comp.row(j) * phi - b_in.row(m) * psi - psi * gm1_comp.row(j);
      gm1_comp.row(j) = bex.row(ind);
      hm1_comp.row(j) = bex.row(ind+1);
    }
  }

  // The final x
  bex.row((n-1)*block_size) = b_in.row(n-1);

  for (int m = n - 2; m >= 0; --m) {
    if (m < n - 2) tau = t(m+2) - t(m+1);
    else tau = T(0.0);

    // u
    strt = m*block_size + 1;
    for (int j = 0; j < p_real; ++j) {
      phi = exp(-beta_real(j) * tau);
      ind = strt + j;
      bex.row(ind) = up1_real.row(j) * phi + b_in.row(m + 1) * alpha_real(j);
      up1_real.row(j) = bex.row(ind);
    }

    strt += p_real;
    for (int j = 0; j < p_complex; ++j) {
      phi = exp(-beta_complex_real(j) * tau) * cos(beta_complex_imag(j) * tau);
      psi = -exp(-beta_complex_real(j) * tau) * sin(beta_complex_imag(j) * tau);
      ind = strt + 2*j;
      bex.row(ind) = up1_comp.row(j) * phi + b_in.row(m + 1) * alpha_complex_real(j) + psi * vp1_comp.row(j);
      bex.row(ind+1) = vp1_comp.row(j) * phi - b_in.row(m + 1) * alpha_complex_imag(j) - psi * up1_comp.row(j);
      up1_comp.row(j) = bex.row(ind);
      vp1_comp.row(j) = bex.row(ind+1);
    }
  }

  // Do the dot product - WARNING: this assumes symmetry!
  Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> b_out(b_in.rows(), b_in.cols());
  for (int j = 0; j < nrhs; ++j) {
    for (int i = 0; i < n; ++i) {
      int k = block_size * i;
      b_out(i, j) = 0.0;
      for (int kp = std::max(0, width - k); kp < std::min(2*width+1, dim_ext + width - k); ++kp)
        b_out(i, j) += A(kp, k) * bex(k + kp - width, j);
    }
  }

  return b_out;
}

#undef BLOCKSIZE
#undef BLOCKSIZE_BASE

};
};

#endif
