#ifndef _GENRP_SOLVER_DIRECT_H_
#define _GENRP_SOLVER_DIRECT_H_

#include <cmath>
#include <vector>
#include <complex>
#include <Eigen/Dense>

namespace genrp {

class DirectSolver {
protected:
  Eigen::VectorXd alpha_real_, alpha_complex_, beta_real_;
  Eigen::VectorXcd beta_complex_;
  size_t n_, p_real_, p_complex_;
  double log_det_;

public:
  DirectSolver () {};
  DirectSolver (const Eigen::VectorXd alpha, const Eigen::VectorXd beta);
  DirectSolver (const Eigen::VectorXd alpha, const Eigen::VectorXcd beta);
  DirectSolver (const Eigen::VectorXd alpha_real, const Eigen::VectorXd beta_real,
                const Eigen::VectorXd alpha_complex, const Eigen::VectorXcd beta_complex);

  virtual ~DirectSolver () {};
  void alpha_and_beta (const Eigen::VectorXd alpha, const Eigen::VectorXd beta);
  void alpha_and_beta (const Eigen::VectorXd alpha, const Eigen::VectorXcd beta);
  void alpha_and_beta (const Eigen::VectorXd alpha_real, const Eigen::VectorXd beta_real,
                       const Eigen::VectorXd alpha_complex, const Eigen::VectorXcd beta_complex);
  virtual void compute (const Eigen::VectorXd& x, const Eigen::VectorXd& diag);
  virtual void solve (const Eigen::MatrixXd& b, double* x) const;
  double dot_solve (const Eigen::VectorXd& b) const;
  double log_determinant () const;

  // Eigen-free interface.
  DirectSolver (size_t p, const double* alpha, const double* beta);
  DirectSolver (size_t p, const double* alpha, const std::complex<double>* beta);
  DirectSolver (size_t p_real, const double* alpha_real, const double* beta_real,
                size_t p_complex, const double* alpha_complex, const std::complex<double>* beta_complex);

  void compute (size_t n, const double* t, const double* diag);
  void solve (const double* b, double* x) const;
  void solve (size_t nrhs, const double* b, double* x) const;
  double dot_solve (const double* b) const;

private:
  Eigen::LDLT<Eigen::MatrixXd> factor_;

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
    alpha_complex_(alpha_complex),
    beta_real_(beta_real),
    beta_complex_(beta_complex),
    p_real_(alpha_real.rows()),
    p_complex_(alpha_complex.rows())
{
  assert ((alpha_real_.rows() == beta_real_.rows()) && "DIMENSION_MISMATCH");
  assert ((alpha_complex_.rows() == beta_complex_.rows()) && "DIMENSION_MISMATCH");
}

void DirectSolver::alpha_and_beta (const Eigen::VectorXd alpha, const Eigen::VectorXd beta)
{
  p_real_ = alpha.rows();
  p_complex_ = 0;
  alpha_real_ = alpha;
  beta_real_ = beta;
}

void DirectSolver::alpha_and_beta (const Eigen::VectorXd alpha, const Eigen::VectorXcd beta)
{
  p_real_ = 0;
  p_complex_ = alpha.rows();
  alpha_complex_ = alpha;
  beta_complex_ = beta;
}

void DirectSolver::alpha_and_beta (const Eigen::VectorXd alpha_real, const Eigen::VectorXd beta_real,
                                   const Eigen::VectorXd alpha_complex, const Eigen::VectorXcd beta_complex)
{
  p_real_ = alpha_real.rows();
  p_complex_ = alpha_complex.rows();
  alpha_real_ = alpha_real;
  beta_real_ = beta_real;
  alpha_complex_ = alpha_complex;
  beta_complex_ = beta_complex;
}

void DirectSolver::compute (const Eigen::VectorXd& x, const Eigen::VectorXd& diag) {
  assert ((x.rows() == diag.rows()) && "DIMENSION_MISMATCH");
  n_ = x.rows();

  // Build the matrix.
  double v, dx, asum = alpha_real_.sum() + alpha_complex_.sum();
  Eigen::MatrixXd K(n_, n_);
  for (size_t i = 0; i < n_; ++i) {
    K(i, i) = asum + diag(i);

    for (size_t j = i+1; j < n_; ++j) {
      v = 0.0;
      dx = fabs(x(j) - x(i));
      v += (alpha_real_.array() * exp(-beta_real_.array() * dx)).sum();
      v += (alpha_complex_.array() * exp(-beta_complex_.real().array() * dx) * cos(beta_complex_.imag().array() * dx)).sum();
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

  Eigen::MatrixXd result = factor_.solve(b);

  // Copy the output.
  for (size_t j = 0; j < nrhs; ++j)
    for (size_t i = 0; i < n_; ++i)
      x[i+j*nrhs] = result(i, j);
}

double DirectSolver::dot_solve (const Eigen::VectorXd& b) const {
  Eigen::VectorXd out(n_);
  solve(b, &(out(0)));
  return b.transpose() * out;
}

double DirectSolver::log_determinant () const {
  return log_det_;
}

// Eigen-free interface:
DirectSolver::DirectSolver (size_t p, const double* alpha, const double* beta) {
  p_real_ = p;
  p_complex_ = 0;
  alpha_real_ = Eigen::Map<const Eigen::VectorXd>(alpha, p, 1);
  beta_real_ = Eigen::Map<const Eigen::VectorXd>(beta, p, 1);
}

DirectSolver::DirectSolver (size_t p, const double* alpha, const std::complex<double>* beta) {
  p_real_ = 0;
  p_complex_ = p;
  alpha_complex_ = Eigen::Map<const Eigen::VectorXd>(alpha, p, 1);
  beta_complex_ = Eigen::Map<const Eigen::VectorXcd>(beta, p, 1);
}

DirectSolver::DirectSolver (size_t p_real, const double* alpha_real, const double* beta_real,
                            size_t p_complex, const double* alpha_complex, const std::complex<double>* beta_complex) {
  p_real_ = p_real;
  p_complex_ = p_real;
  alpha_real_ = Eigen::Map<const Eigen::VectorXd>(alpha_real, p_real, 1);
  beta_real_ = Eigen::Map<const Eigen::VectorXd>(beta_real, p_real, 1);
  alpha_complex_ = Eigen::Map<const Eigen::VectorXd>(alpha_complex, p_complex, 1);
  beta_complex_ = Eigen::Map<const Eigen::VectorXcd>(beta_complex, p_complex, 1);
}

void DirectSolver::compute (size_t n, const double* t, const double* diag) {
  typedef Eigen::Map<const Eigen::Matrix<double, Eigen::Dynamic, 1> > vector_t;
  compute(vector_t(t, n, 1), vector_t(diag, n, 1));
}

void DirectSolver::solve (const double* b, double* x) const {
  Eigen::Map<const Eigen::MatrixXd> bin(b, n_, 1);
  solve(bin, x);
}

void DirectSolver::solve (size_t nrhs, const double* b, double* x) const {
  Eigen::Map<const Eigen::MatrixXd> bin(b, n_, nrhs);
  solve(bin, x);
}

double DirectSolver::dot_solve (const double* b) const {
  Eigen::Map<const Eigen::VectorXd> bin(b, n_);
  return dot_solve(bin);
}

};

#endif
