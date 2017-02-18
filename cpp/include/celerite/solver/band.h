#ifndef _CELERITE_SOLVER_BAND_H_
#define _CELERITE_SOLVER_BAND_H_

#include <cmath>
#include <Eigen/Core>

#include "celerite/utils.h"
#include "celerite/extended.h"
#include "celerite/exceptions.h"

#include "celerite/lapack.h"
#include "celerite/banded.h"
#include "celerite/solver/solver.h"

namespace celerite {
namespace solver {

/// The class provides all of the computation power for the fast GP solver
///
/// The `celerite::solver::BandSolver::compute` method must be called before
/// most of the other methods.
template <typename T>
class BandSolver : public Solver<T> {
public:
  /// You can decide to use LAPACK for solving if it is available
  ///
  /// @param use_lapack If true, LAPACK will be used for solving the band
  //                    system.
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
  WIDTH
  int offset = offset_factor * width;

  // Set up the extended matrix.
  A.resize(1+width+offset, dim_ext);
  A.setConstant(T(0.0));

  // Compute the diagonal element
  T sum_alpha = alpha_real.sum() + alpha_complex_real.sum();

  // Pre-compute the phis and psis
  double dt;
  T amp, arg;
  Eigen::Array<T, Eigen::Dynamic, 1> phi_real(p_real), phi_complex(p_complex), psi_complex(p_complex);

  dt = x(1) - x(0);
  for (j = 0; j < p_real; ++j) {
    phi_real(j) = exp(-beta_real(j) * dt);
  }
  for (j = 0; j < p_complex; ++j) {
    amp = exp(-beta_complex_real(j) * dt);
    arg = beta_complex_imag(j) * dt;
    phi_complex(j) = amp * cos(arg);
    psi_complex(j) = -amp * sin(arg);
  }

  int row, col, row2, row3;
  for (k = 0; k < n - 1; ++k) {
    // First column
    col = block_size * k;
    row = offset;
    A(row, col) = sum_alpha;
    row++;
    for (j = 0; j < p_real; ++j, ++row) {
      A(row, col) = phi_real(j);
    }
    for (j = 0; j < p_complex; ++j, row += 2) {
      A(row, col) = phi_complex(j);
      A(row+1, col) = psi_complex(j);
    }

    // Block 1
    col++;
    row2 = row - 1;
    row = offset - 1;
    if (k > 0) {
      row3 = offset - p_real - 2*p_complex - 1;
      for (j = 0; j < p_real; ++j, ++col, --row) {
        A(row3, col) = phi_real(j);
        A(row, col) = phi_real(j);
        A(row2, col) = T(-1.0);
      }
      for (j = 0; j < p_complex; ++j, col += 2, row -= 2) {
        A(row3, col) = phi_complex(j);
        A(row3+1, col) = psi_complex(j);
        A(row, col) = phi_complex(j);
        A(row2, col) = T(-1.0);

        A(row3-1, col+1) = psi_complex(j);
        A(row3, col+1) = -phi_complex(j);
        A(row-1, col+1) = psi_complex(j);
        A(row2, col+1) = T(1.0);
      }
    } else {
      for (j = 0; j < p_real; ++j, ++col, --row) {
        A(row, col) = phi_real(j);
        A(row2, col) = T(-1.0);
      }
      for (j = 0; j < p_complex; ++j, col += 2, row -= 2) {
        A(row, col) = phi_complex(j);
        A(row2, col) = T(-1.0);
        A(row-1, col+1) = psi_complex(j);
        A(row2, col+1) = T(1.0);
      }
    }

    // Block 3
    row = offset - p_real - 2*p_complex;
    row3 = row2 - p_real;
    if (k < n-2) {
      // Update the phis and psis
      dt = x(k+2) - x(k+1);
      for (j = 0; j < p_real; ++j) {
        phi_real(j) = exp(-beta_real(j) * dt);
      }
      for (j = 0; j < p_complex; ++j) {
        amp = exp(-beta_complex_real(j) * dt);
        arg = beta_complex_imag(j) * dt;
        phi_complex(j) = amp * cos(arg);
        psi_complex(j) = -amp * sin(arg);
      }

      for (j = 0; j < p_real; ++j, ++col) {
        A(row, col) = T(-1.0);
        A(row2 - j, col) = alpha_real(j);
        A(row2 + 1, col) = phi_real(j);
      }
      for (j = 0; j < p_complex; ++j, col += 2) {
        A(row, col) = T(-1.0);
        A(row3 - 2*j, col) = alpha_complex_real(j);
        A(row2 + 1, col) = phi_complex(j);
        A(row2 + 2, col) = psi_complex(j);

        A(row, col+1) = T(1.0);
        A(row3 - 2*j - 1, col+1) = alpha_complex_imag(j);
        A(row2, col+1) = psi_complex(j);
        A(row2 + 1, col+1) = -phi_complex(j);
      }
    } else {
      for (j = 0; j < p_real; ++j, ++col) {
        A(row, col) = T(-1.0);
        A(row2 - j, col) = alpha_real(j);
      }
      for (j = 0; j < p_complex; ++j, col += 2) {
        A(row, col) = T(-1.0);
        A(row3 - 2*j, col) = alpha_complex_real(j);
        A(row, col+1) = T(1.0);
        A(row3 - 2*j - 1, col+1) = alpha_complex_imag(j);
      }
    }

    for (j = 0; j < p_real; ++j, ++row) {
      A(row, col) = alpha_real(j);
    }
    for (j = 0; j < p_complex; ++j, row+=2) {
      A(row, col) = alpha_complex_real(j);
      A(row+1, col) = alpha_complex_imag(j);
    }
  }
  A(offset, col) = sum_alpha;
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
  WIDTH

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
  int offset = offset_factor*width;
  for (int k = 0; k < this->n_; ++k)
    a_(offset, k*block_size) += diag(k);

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
  WIDTH

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
  Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> bex =
    build_b_ext(alpha_real, beta_real, alpha_complex_real, alpha_complex_imag,
                beta_complex_real, beta_complex_imag, t, b_in);

  int p_real = alpha_real.rows(),
      p_complex = alpha_complex_real.rows(),
      n = t.rows(),
      nrhs = b_in.cols();
  BLOCKSIZE_BASE
  WIDTH

  // Build the extended matrix
  Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> A;
  build_matrix(alpha_real, beta_real, alpha_complex_real, alpha_complex_imag,
               beta_complex_real, beta_complex_imag, 1, t, A);

  Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> b_out(b_in.rows(), b_in.cols());
  // Do the dot product - WARNING: this assumes symmetry!
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

};
};

#endif
