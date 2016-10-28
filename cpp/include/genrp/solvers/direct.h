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
  size_t n_, p_real_, p_complex_;
  double log_det_;

public:
  DirectSolver () {};
  DirectSolver (const real_vector_t alpha, const real_vector_t beta);
  DirectSolver (const real_vector_t alpha, const complex_vector_t beta);
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
  Eigen::LDLT<real_matrix_t> factor_;

};

DirectSolver::DirectSolver (const Eigen::VectorXd alpha, const Eigen::VectorXd beta)
  : alpha_real_(alpha),
    beta_real_(beta),
    p_real_(alpha.rows()),
    p_complex_(0)
{
  assert ((alpha_real_.rows() == beta_real_.rows()) && "DIMENSION_MISMATCH");
}

DirectSolver::DirectSolver (const Eigen::VectorXd alpha, const Eigen::VectorXcd beta)
  : alpha_complex_(alpha),
    beta_complex_(beta),
    p_real_(0),
    p_complex_(alpha.rows())
{
  assert ((alpha_complex_.rows() == beta_complex_.rows()) && "DIMENSION_MISMATCH");
}

DirectSolver::DirectSolver (const Eigen::VectorXd alpha_real, const Eigen::VectorXd beta_real,
                            const Eigen::VectorXd alpha_complex, const Eigen::VectorXcd beta_complex)
  : alpha_real_(alpha_real),
    beta_real_(beta_real),
    alpha_complex_(alpha_complex),
    beta_complex_(beta_complex),
    p_real_(alpha_real.rows()),
    p_complex_(alpha_complex.rows())
{
  assert ((alpha_real_.rows() == beta_real_.rows()) && "DIMENSION_MISMATCH");
  assert ((alpha_complex_.rows() == beta_complex_.rows()) && "DIMENSION_MISMATCH");
}

// template <typename entry_t>
// void DirectSolver<entry_t>::alpha_and_beta (const Eigen::VectorXd alpha, const Eigen::Matrix<entry_t, Eigen::Dynamic, 1> beta) {
//   p_ = alpha.rows();
//   alpha_ = alpha;
//   beta_ = beta;
// }

void DirectSolver::compute (const Eigen::VectorXd& x, const Eigen::VectorXd& diag) {
  assert ((x.rows() == diag.rows()) && "DIMENSION_MISMATCH");
  n_ = x.rows();

  // Build the matrix.
  double v, dx, asum = alpha_real_.sum() + 2.0 * alpha_complex_.sum();
  real_matrix_t K(n_, n_);
  for (size_t i = 0; i < n_; ++i) {
    K(i, i) = asum + diag(i);

    for (size_t j = i+1; j < n_; ++j) {
      v = 0.0;
      dx = fabs(x(j) - x(i));
      v += (alpha_real_.array() * exp(-beta_real_.array() * dx)).sum();
      v += 2.0 * (alpha_complex_.array() * exp(-beta_complex_.real().array() * dx) * cos(beta_complex_.imag().array() * dx)).sum();
      K(i, j) = v;
      K(j, i) = v;
    }
  }

  // Factorize the matrix.
  factor_ = K.ldlt();
  log_det_ = log(factor_.vectorD().array()).sum();
}

void DirectSolver::solve (const Eigen::MatrixXd& b, double* x) const {
  assert ((b.rows() == n_) && "DIMENSION_MISMATCH");
  size_t nrhs = b.cols();

  real_matrix_t result = factor_.solve(b);

  // Copy the output.
  for (size_t j = 0; j < nrhs; ++j)
    for (size_t i = 0; i < n_; ++i)
      x[i+j*nrhs] = result(i, j);
}

double DirectSolver::dot_solve (const Eigen::VectorXd& b) const {
  real_vector_t out(n_);
  solve(b, &(out(0)));
  return b.transpose() * out;
}

double DirectSolver::log_determinant () const {
  return log_det_;
}

// // Eigen-free interface:
// template <typename entry_t>
// DirectSolver<entry_t>::DirectSolver (size_t p, const double* alpha, const entry_t* beta) {
//   p_ = p;
//   alpha_ = Eigen::Map<const Eigen::Matrix<double, Eigen::Dynamic, 1> >(alpha, p, 1);
//   beta_ = Eigen::Map<const Eigen::Matrix<entry_t, Eigen::Dynamic, 1> >(beta, p, 1);
// }
//
// template <typename entry_t>
// void DirectSolver<entry_t>::compute (size_t n, const double* t, const double* diag) {
//   typedef Eigen::Map<const Eigen::Matrix<double, Eigen::Dynamic, 1> > vector_t;
//   compute(vector_t(t, n, 1), vector_t(diag, n, 1));
// }
//
// template <typename entry_t>
// void DirectSolver<entry_t>::solve (const double* b, double* x) const {
//   Eigen::Map<const Eigen::Matrix<double, Eigen::Dynamic, 1> > bin(b, n_, 1);
//   solve(bin, x);
// }
//
// template <typename entry_t>
// void DirectSolver<entry_t>::solve (size_t nrhs, const double* b, double* x) const {
//   Eigen::Map<const Eigen::Matrix<double, Eigen::Dynamic, 1> > bin(b, n_, nrhs);
//   solve(bin, x);
// }
//
// template <typename entry_t>
// double DirectSolver<entry_t>::dot_solve (const double* b) const {
//   Eigen::Map<const Eigen::Matrix<double, Eigen::Dynamic, 1> > bin(b, n_, 1);
//   return dot_solve(bin);
// }

};

#endif
