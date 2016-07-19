#ifndef _GEORGE_GRP_
#define _GEORGE_GRP_

#include <cmath>
#include <vector>
#include <complex>
#include <Eigen/Dense>
#include <Eigen/Sparse>

namespace genrp {

#define GRP_DIMENSION_MISMATCH 1

class GRPSolver {
  typedef std::complex<double> entry_t;

public:
  GRPSolver (const Eigen::VectorXd alpha, const Eigen::VectorXcd beta);
  void compute (const Eigen::VectorXd x, const Eigen::VectorXd diag);
  void solve (const Eigen::VectorXd& b, double* x) const;
  double get_log_determinant () const;

  // Eigen-free interface.
  GRPSolver (size_t p, const double* alpha, const entry_t* beta);
  void solve (const double* b, double* x) const;
  void compute (size_t n, const double* t, const double* diag);

private:
  Eigen::VectorXd alpha_;
  Eigen::VectorXcd beta_;
  size_t n_, p_, block_size_, dim_ext_;
  Eigen::SparseLU<Eigen::SparseMatrix<entry_t>, Eigen::COLAMDOrdering<int> > factor_;

};

GRPSolver::GRPSolver (const Eigen::VectorXd alpha, const Eigen::VectorXcd beta)
  : alpha_(alpha),
    beta_(beta),
    p_(alpha.rows())
{
  if (alpha_.rows() != beta_.rows()) throw GRP_DIMENSION_MISMATCH;
}

void GRPSolver::compute (const Eigen::VectorXd x, const Eigen::VectorXd diag) {
  typedef Eigen::Triplet<entry_t> triplet_t;

  // Check dimensions.
  if (x.rows() != diag.rows()) throw GRP_DIMENSION_MISMATCH;
  n_ = x.rows();

  // Pre-compute gamma: Equation (58)
  Eigen::MatrixXcd gamma(n_ - 1, p_);
  for (size_t i = 0; i < n_ - 1; ++i) {
    double delta = fabs(x(i+1) - x(i));
    for (size_t k = 0; k < p_; ++k)
      gamma(i, k) = exp(-beta_(k) * delta);
  }

  // Compute the block sizes: Algorithm 1
  block_size_ = 2 * p_ + 1;
  dim_ext_ = block_size_ * n_ - 2 * p_;

  // Find the non-zero entries in the sparse representation using Algorithm 1.
  size_t nnz = (n_-1)*(6*p_+1) + 2*(n_-2)*p_ + 1, count = 0;
  std::vector<triplet_t> triplets(nnz);

  for (size_t i = 0; i < n_; ++i)  // Line 3
    triplets[count++] = triplet_t(i * block_size_, i * block_size_, diag(i));

  size_t a, b;
  entry_t value;
  for (size_t i = 0; i < n_ - 1; ++i) {  // Line 5
    size_t im1n = i * block_size_,        // (i - 1) * n
           in = (i + 1) * block_size_;    // i * n
    for (size_t k = 0; k < p_; ++k) {
      // Lines 6-7:
      a = im1n;
      b = im1n+k+1;
      value = gamma(i, k);
      triplets[count++] = triplet_t(a, b, value);
      triplets[count++] = triplet_t(b, a, value);

      // Lines 8-9:
      a = in;
      b = im1n+p_+k+1;
      value = alpha_(k);
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
      value = gamma(i+1, k);
      triplets[count++] = triplet_t(a, b, value);
      triplets[count++] = triplet_t(b, a, value);
    }
  }

  // Build and factorize the sparse matrix.
  Eigen::SparseMatrix<entry_t> A_ex(dim_ext_, dim_ext_);
  A_ex.setFromTriplets(triplets.begin(), triplets.end());
  factor_.compute(A_ex);
}

void GRPSolver::solve (const Eigen::VectorXd& b, double* x) const {
  if (b.rows() != n_) throw GRP_DIMENSION_MISMATCH;

  // Pad the input vector to the extended size.
  Eigen::VectorXcd bex = Eigen::VectorXcd::Zero(dim_ext_);
  for (size_t i = 0; i < n_; ++i) bex(i*block_size_) = b(i);

  // Solve the extended system.
  Eigen::VectorXcd xex = factor_.solve(bex);

  // Copy the output.
  for (size_t i = 0; i < n_; ++i) x[i] = xex(i*block_size_).real();
}

double GRPSolver::get_log_determinant () const {
  return factor_.logAbsDeterminant().real();
}

// Eigen-free interface:
GRPSolver::GRPSolver (size_t p, const double* alpha, const entry_t* beta) {
  p_ = p;
  alpha_ = Eigen::Map<const Eigen::Matrix<double, Eigen::Dynamic, 1> >(alpha, p, 1);
  beta_ = Eigen::Map<const Eigen::Matrix<entry_t, Eigen::Dynamic, 1> >(beta, p, 1);
}

void GRPSolver::compute (size_t n, const double* t, const double* diag) {
  typedef Eigen::Map<const Eigen::Matrix<double, Eigen::Dynamic, 1> > vector_t;
  compute(vector_t(t, n, 1), vector_t(diag, n, 1));
}

void GRPSolver::solve (const double* b, double* x) const {
  Eigen::Map<const Eigen::Matrix<double, Eigen::Dynamic, 1> > bin(b, n_, 1);
  solve(bin, x);
}

};

#endif
