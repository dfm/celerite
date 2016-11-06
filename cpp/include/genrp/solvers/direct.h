#ifndef _GENRP_SOLVER_DIRECT_H_
#define _GENRP_SOLVER_DIRECT_H_

#include <cmath>
#include <vector>
#include <complex>
#include <Eigen/Dense>

namespace genrp {

template <typename T>
class DirectSolver {
protected:
  size_t n_, p_real_, p_complex_;
  T log_det_;

public:
  DirectSolver () {};
  virtual ~DirectSolver () {};

  virtual void compute (
    const Eigen::Matrix<T, Eigen::Dynamic, 1>& alpha_real,
    const Eigen::Matrix<T, Eigen::Dynamic, 1>& beta_real,
    const Eigen::Matrix<T, Eigen::Dynamic, 1>& alpha_complex,
    const Eigen::Matrix<T, Eigen::Dynamic, 1>& beta_complex_real,
    const Eigen::Matrix<T, Eigen::Dynamic, 1>& beta_complex_imag,
    const Eigen::VectorXd& x,
    const Eigen::Matrix<T, Eigen::Dynamic, 1>& diag
  );
  virtual void solve (const Eigen::MatrixXd& b, T* x) const;
  T dot_solve (const Eigen::VectorXd& b) const;
  T log_determinant () const;

  // Eigen-free interface.
  void compute (size_t p_real, const T* const alpha_real, const T* const beta_real,
                size_t p_complex, const T* const alpha_complex, const T* const beta_complex_real, const T* const beta_complex_imag,
                size_t n, const double* t, const T* const diag);
  void solve (const double* const b, T* x) const;
  void solve (size_t nrhs, const double* const b, T* x) const;
  T dot_solve (const double* const b) const;

private:
  Eigen::LDLT<Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> > factor_;

};

template <typename T>
void DirectSolver<T>::compute (
  const Eigen::Matrix<T, Eigen::Dynamic, 1>& alpha_real,
  const Eigen::Matrix<T, Eigen::Dynamic, 1>& beta_real,
  const Eigen::Matrix<T, Eigen::Dynamic, 1>& alpha_complex,
  const Eigen::Matrix<T, Eigen::Dynamic, 1>& beta_complex_real,
  const Eigen::Matrix<T, Eigen::Dynamic, 1>& beta_complex_imag,
  const Eigen::VectorXd& x,
  const Eigen::Matrix<T, Eigen::Dynamic, 1>& diag
)
{
  // Check the dimensions
  assert ((alpha_real.rows() == beta_real.rows()) && "DIMENSION_MISMATCH");
  assert ((alpha_complex.rows() == beta_complex_real.rows()) && "DIMENSION_MISMATCH");
  assert ((alpha_complex.rows() == beta_complex_imag.rows()) && "DIMENSION_MISMATCH");
  assert ((x.rows() == diag.rows()) && "DIMENSION_MISMATCH");

  // Save the dimensions for later use
  p_real_ = alpha_real.rows();
  p_complex_ = alpha_complex.rows();
  n_ = x.rows();

  // Build the matrix.
  double dx;
  T v, asum = alpha_real.sum() + alpha_complex.sum();
  Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> K(n_, n_);
  for (size_t i = 0; i < n_; ++i) {
    K(i, i) = asum + diag(i);

    for (size_t j = i+1; j < n_; ++j) {
      v = 0.0;
      dx = fabs(x(j) - x(i));
      v += (alpha_real.array() * exp(-beta_real.array() * dx)).sum();
      v += (alpha_complex.array() * exp(-beta_complex_real.array() * dx) * cos(beta_complex_imag.array() * dx)).sum();
      K(i, j) = v;
      K(j, i) = v;
    }
  }

  // Factorize the matrix.
  factor_ = K.ldlt();
  log_det_ = log(factor_.vectorD().array()).sum();
}

template <typename T>
void DirectSolver<T>::solve (const Eigen::MatrixXd& b, T* x) const {
  assert ((b.rows() == n_) && "DIMENSION_MISMATCH");
  size_t nrhs = b.cols();

  Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> result = factor_.solve(b);

  // Copy the output.
  for (size_t j = 0; j < nrhs; ++j)
    for (size_t i = 0; i < n_; ++i)
      x[i+j*nrhs] = result(i, j);
}

template <typename T>
T DirectSolver<T>::dot_solve (const Eigen::VectorXd& b) const {
  Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> out(n_);
  solve(b, out.data());
  return b.transpose() * out;
}

template <typename T>
T DirectSolver<T>::log_determinant () const {
  return log_det_;
}

// // Eigen-free interface:
// DirectSolver::DirectSolver (size_t p, const double* alpha, const double* beta) {
//   p_real_ = p;
//   p_complex_ = 0;
//   alpha_real = Eigen::Map<const Eigen::VectorXd>(alpha, p, 1);
//   beta_real = Eigen::Map<const Eigen::VectorXd>(beta, p, 1);
// }
//
// DirectSolver::DirectSolver (size_t p, const double* alpha, const double* beta_real, const double* beta_imag) {
//   p_real_ = 0;
//   p_complex_ = p;
//   alpha_complex = Eigen::Map<const Eigen::VectorXd>(alpha, p, 1);
//   beta_complex_real = Eigen::Map<const Eigen::VectorXd>(beta_real, p, 1);
//   beta_complex_imag = Eigen::Map<const Eigen::VectorXd>(beta_imag, p, 1);
// }
//
// DirectSolver::DirectSolver (size_t p_real, const double* alpha_real, const double* beta_real,
//                             size_t p_complex, const double* alpha_complex, const double* beta_complex_real, const double* beta_complex_imag) {
//   p_real_ = p_real;
//   p_complex_ = p_complex;
//   alpha_real = Eigen::Map<const Eigen::VectorXd>(alpha_real, p_real, 1);
//   beta_real = Eigen::Map<const Eigen::VectorXd>(beta_real, p_real, 1);
//   alpha_complex = Eigen::Map<const Eigen::VectorXd>(alpha_complex, p_complex, 1);
//   beta_complex_real = Eigen::Map<const Eigen::VectorXd>(beta_complex_real, p_complex, 1);
//   beta_complex_imag = Eigen::Map<const Eigen::VectorXd>(beta_complex_imag, p_complex, 1);
// }
//
// void DirectSolver::compute (size_t n, const double* t, const double* diag) {
//   typedef Eigen::Map<const Eigen::Matrix<double, Eigen::Dynamic, 1> > vector_t;
//   compute(vector_t(t, n, 1), vector_t(diag, n, 1));
// }
//
// void DirectSolver::solve (const double* b, double* x) const {
//   Eigen::Map<const Eigen::MatrixXd> bin(b, n_, 1);
//   solve(bin, x);
// }
//
// void DirectSolver::solve (size_t nrhs, const double* b, double* x) const {
//   Eigen::Map<const Eigen::MatrixXd> bin(b, n_, nrhs);
//   solve(bin, x);
// }
//
// double DirectSolver::dot_solve (const double* b) const {
//   Eigen::Map<const Eigen::VectorXd> bin(b, n_);
//   return dot_solve(bin);
// }

};

#endif
