#ifndef _GENRP_SOLVER_SOLVER_H_
#define _GENRP_SOLVER_SOLVER_H_

#include <Eigen/Core>

#include "genrp/exceptions.h"

namespace genrp {
namespace solver {

int SOLVER_DIMENSION_MISMATCH = 1;
int SOLVER_NOT_COMPUTED = 2;

template <typename T>
class Solver {
protected:
  bool computed_;
  int n_, p_real_, p_complex_;
  T log_det_;

public:
  Solver () : computed_(false) {};
  virtual ~Solver () {};

  virtual int compute (
    const Eigen::Matrix<T, Eigen::Dynamic, 1>& alpha_real,
    const Eigen::Matrix<T, Eigen::Dynamic, 1>& beta_real,
    const Eigen::Matrix<T, Eigen::Dynamic, 1>& alpha_complex_real,
    const Eigen::Matrix<T, Eigen::Dynamic, 1>& alpha_complex_imag,
    const Eigen::Matrix<T, Eigen::Dynamic, 1>& beta_complex_real,
    const Eigen::Matrix<T, Eigen::Dynamic, 1>& beta_complex_imag,
    const Eigen::VectorXd& x,
    const Eigen::Matrix<T, Eigen::Dynamic, 1>& diag
  ) = 0;
  virtual void solve (const Eigen::MatrixXd& b, T* x) const = 0;

  Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> solve (const Eigen::MatrixXd& b) const;
  T dot_solve (const Eigen::VectorXd& b) const;
  T log_determinant () const {
    if (!(this->computed_)) throw compute_exception();
    return log_det_;
  };
  bool computed () const { return computed_; };

  // Helpers
  int compute (
    const Eigen::Matrix<T, Eigen::Dynamic, 1>& alpha_real,
    const Eigen::Matrix<T, Eigen::Dynamic, 1>& beta_real,
    const Eigen::Matrix<T, Eigen::Dynamic, 1>& alpha_complex_real,
    const Eigen::Matrix<T, Eigen::Dynamic, 1>& beta_complex_real,
    const Eigen::Matrix<T, Eigen::Dynamic, 1>& beta_complex_imag,
    const Eigen::VectorXd& x,
    const Eigen::Matrix<T, Eigen::Dynamic, 1>& diag
  );
  int compute (
    const Eigen::Matrix<T, Eigen::Dynamic, 1>& alpha_real,
    const Eigen::Matrix<T, Eigen::Dynamic, 1>& beta_real,
    const Eigen::VectorXd& x,
    const Eigen::Matrix<T, Eigen::Dynamic, 1>& diag
  );
  int compute (
    const Eigen::Matrix<T, Eigen::Dynamic, 1>& alpha_complex_real,
    const Eigen::Matrix<T, Eigen::Dynamic, 1>& beta_complex_real,
    const Eigen::Matrix<T, Eigen::Dynamic, 1>& beta_complex_imag,
    const Eigen::VectorXd& x,
    const Eigen::Matrix<T, Eigen::Dynamic, 1>& diag
  );
  int compute (
    const Eigen::Matrix<T, Eigen::Dynamic, 1>& alpha_complex_real,
    const Eigen::Matrix<T, Eigen::Dynamic, 1>& alpha_complex_imag,
    const Eigen::Matrix<T, Eigen::Dynamic, 1>& beta_complex_real,
    const Eigen::Matrix<T, Eigen::Dynamic, 1>& beta_complex_imag,
    const Eigen::VectorXd& x,
    const Eigen::Matrix<T, Eigen::Dynamic, 1>& diag
  );

};

template <typename T>
T Solver<T>::dot_solve (const Eigen::VectorXd& b) const {
  if (!(this->computed_)) throw compute_exception();
  Eigen::Matrix<T, Eigen::Dynamic, 1> out(n_);
  solve(b, out.data());
  return b.transpose().cast<T>() * out;
}

template <typename T>
Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> Solver<T>::solve (const Eigen::MatrixXd& b) const {
  Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> x(b.rows(), b.cols());
  solve(b, x.data());
  return x;
}

// Helpers
template <typename T>
int Solver<T>::compute (
  const Eigen::Matrix<T, Eigen::Dynamic, 1>& alpha_real,
  const Eigen::Matrix<T, Eigen::Dynamic, 1>& beta_real,
  const Eigen::Matrix<T, Eigen::Dynamic, 1>& alpha_complex_real,
  const Eigen::Matrix<T, Eigen::Dynamic, 1>& beta_complex_real,
  const Eigen::Matrix<T, Eigen::Dynamic, 1>& beta_complex_imag,
  const Eigen::VectorXd& x,
  const Eigen::Matrix<T, Eigen::Dynamic, 1>& diag
)
{
  Eigen::Matrix<T, Eigen::Dynamic, 1> alpha_complex_imag = Eigen::Matrix<T, Eigen::Dynamic, 1>::Zero(alpha_complex_real.rows());
  return this->compute(alpha_real, beta_real, alpha_complex_real, alpha_complex_imag, beta_complex_real, beta_complex_imag, x, diag);
}

template <typename T>
int Solver<T>::compute (
  const Eigen::Matrix<T, Eigen::Dynamic, 1>& alpha_real,
  const Eigen::Matrix<T, Eigen::Dynamic, 1>& beta_real,
  const Eigen::VectorXd& x,
  const Eigen::Matrix<T, Eigen::Dynamic, 1>& diag
)
{
  Eigen::Matrix<T, Eigen::Dynamic, 1> nothing;
  return this->compute(alpha_real, beta_real, nothing, nothing, nothing, nothing, x, diag);
}

template <typename T>
int Solver<T>::compute (
  const Eigen::Matrix<T, Eigen::Dynamic, 1>& alpha_complex_real,
  const Eigen::Matrix<T, Eigen::Dynamic, 1>& beta_complex_real,
  const Eigen::Matrix<T, Eigen::Dynamic, 1>& beta_complex_imag,
  const Eigen::VectorXd& x,
  const Eigen::Matrix<T, Eigen::Dynamic, 1>& diag
)
{
  Eigen::Matrix<T, Eigen::Dynamic, 1> nothing;
  Eigen::Matrix<T, Eigen::Dynamic, 1> alpha_complex_imag = Eigen::Matrix<T, Eigen::Dynamic, 1>::Zero(alpha_complex_real.rows());
  return this->compute(nothing, nothing, alpha_complex_real, alpha_complex_imag, beta_complex_real, beta_complex_imag, x, diag);
}

template <typename T>
int Solver<T>::compute (
  const Eigen::Matrix<T, Eigen::Dynamic, 1>& alpha_complex_real,
  const Eigen::Matrix<T, Eigen::Dynamic, 1>& alpha_complex_imag,
  const Eigen::Matrix<T, Eigen::Dynamic, 1>& beta_complex_real,
  const Eigen::Matrix<T, Eigen::Dynamic, 1>& beta_complex_imag,
  const Eigen::VectorXd& x,
  const Eigen::Matrix<T, Eigen::Dynamic, 1>& diag
)
{
  Eigen::Matrix<T, Eigen::Dynamic, 1> nothing;
  return this->compute(nothing, nothing, alpha_complex_real, alpha_complex_imag, beta_complex_real, beta_complex_imag, x, diag);
}

};
};

#endif
