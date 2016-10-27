#ifndef _GENRP_SOLVER_DIRECT_
#define _GENRP_SOLVER_DIRECT_

#include <cmath>
#include <vector>
#include <complex>
#include <Eigen/Dense>
#include <Eigen/Sparse>

#include "genrp/utils.h"

namespace genrp {

class DirectSolver {
  typedef Eigen::Matrix<double, Eigen::Dynamic, 1> real_vector_t;
  typedef Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> real_matrix_t;
  typedef Eigen::Matrix<std::complex<double>, Eigen::Dynamic, 1> complex_vector_t;
  typedef Eigen::Matrix<std::complex<double>, Eigen::Dynamic, Eigen::Dynamic> complex_matrix_t;

protected:
  real_vector_t alpha_real_, alpha_complex_, beta_real_;
  complex_vector_t beta_complex_;
  size_t n_, p_;
  double log_det_;

public:
  DirectSolver () {};
  DirectSolver (const real_vector_t alpha, const real_vector_t beta);
  DirectSolver (const real_vector_t alpha_real, const real_vector_t beta_real,
                const real_vector_t alpha_complex, const complex_vector_t beta_complex);

  virtual ~DirectSolver () {};
  // void alpha_and_beta (const Eigen::VectorXd alpha, const vector_t beta);
  virtual void compute (const real_vector_t& x, const real_vector_t& diag);
  virtual void solve (const real_matrix_t& b, double* x) const;
  double dot_solve (const real_vector_t& b) const;
  double log_determinant () const;

  // Eigen-free interface.
  // DirectSolver (size_t p, const double* alpha, const entry_t* beta);
  // void compute (size_t n, const double* t, const double* diag);
  // void solve (const double* b, double* x) const;
  // void solve (size_t nrhs, const double* b, double* x) const;
  // double dot_solve (const double* b) const;

private:
  Eigen::LDLT<complex_matrix_t> factor_;

};

DirectSolver::DirectSolver (const Eigen::VectorXd alpha, const Eigen::VectorXd beta)
  : alpha_(alpha),
    beta_(beta),
    p_(alpha.rows())
{
  assert ((alpha_.rows() == beta_.rows()) && "DIMENSION_MISMATCH");
}

template <typename entry_t>
void DirectSolver<entry_t>::alpha_and_beta (const Eigen::VectorXd alpha, const Eigen::Matrix<entry_t, Eigen::Dynamic, 1> beta) {
  p_ = alpha.rows();
  alpha_ = alpha;
  beta_ = beta;
}

template <typename entry_t>
void DirectSolver<entry_t>::compute (const Eigen::VectorXd& x, const Eigen::VectorXd& diag) {
  assert ((x.rows() == diag.rows()) && "DIMENSION_MISMATCH");
  n_ = x.rows();

  // Build the matrix.
  entry_t v, asum = alpha_.sum();
  matrix_t K(n_, n_);
  for (size_t i = 0; i < n_; ++i) {
    K(i, i) = asum + diag(i);

    for (size_t j = i+1; j < n_; ++j) {
      v = entry_t(0.0);
      for (size_t p = 0; p < p_; ++p)
        v += alpha_(p) * exp(-beta_(p) * fabs(x(j) - x(i)));
      K(i, j) = v;
      K(j, i) = get_conj(v);
    }
  }

  // Factorize the matrix.
  factor_ = K.ldlt();
  log_det_ = get_real(log(factor_.vectorD().array()).sum());
}

template <typename entry_t>
void DirectSolver<entry_t>::solve (const Eigen::MatrixXd& b, double* x) const {
  assert ((b.rows() == n_) && "DIMENSION_MISMATCH");
  size_t nrhs = b.cols();

  matrix_t result = factor_.solve(b.cast<entry_t>());

  // Copy the output.
  for (size_t j = 0; j < nrhs; ++j)
    for (size_t i = 0; i < n_; ++i)
      x[i+j*nrhs] = get_real(result(i, j));
}

template <typename entry_t>
double DirectSolver<entry_t>::dot_solve (const Eigen::VectorXd& b) const {
  Eigen::VectorXd out(n_);
  solve(b, &(out(0)));
  return b.transpose() * out;
}

template <typename entry_t>
double DirectSolver<entry_t>::log_determinant () const {
  return log_det_;
}

// Eigen-free interface:
template <typename entry_t>
DirectSolver<entry_t>::DirectSolver (size_t p, const double* alpha, const entry_t* beta) {
  p_ = p;
  alpha_ = Eigen::Map<const Eigen::Matrix<double, Eigen::Dynamic, 1> >(alpha, p, 1);
  beta_ = Eigen::Map<const Eigen::Matrix<entry_t, Eigen::Dynamic, 1> >(beta, p, 1);
}

template <typename entry_t>
void DirectSolver<entry_t>::compute (size_t n, const double* t, const double* diag) {
  typedef Eigen::Map<const Eigen::Matrix<double, Eigen::Dynamic, 1> > vector_t;
  compute(vector_t(t, n, 1), vector_t(diag, n, 1));
}

template <typename entry_t>
void DirectSolver<entry_t>::solve (const double* b, double* x) const {
  Eigen::Map<const Eigen::Matrix<double, Eigen::Dynamic, 1> > bin(b, n_, 1);
  solve(bin, x);
}

template <typename entry_t>
void DirectSolver<entry_t>::solve (size_t nrhs, const double* b, double* x) const {
  Eigen::Map<const Eigen::Matrix<double, Eigen::Dynamic, 1> > bin(b, n_, nrhs);
  solve(bin, x);
}

template <typename entry_t>
double DirectSolver<entry_t>::dot_solve (const double* b) const {
  Eigen::Map<const Eigen::Matrix<double, Eigen::Dynamic, 1> > bin(b, n_, 1);
  return dot_solve(bin);
}

};

#endif
