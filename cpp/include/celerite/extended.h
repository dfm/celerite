#ifndef _CELERITE_EXTENDED_H_
#define _CELERITE_EXTENDED_H_

#include <cmath>
#include <Eigen/Core>

namespace celerite {

#define BLOCKSIZE_BASE                              \
  int block_size = 2 * p_real + 4 * p_complex + 1,  \
      dim_ext = block_size * (n - 1) + 1;           \

#define BLOCKSIZE                                   \
  int p_real = this->p_real_,                       \
      p_complex = this->p_complex_,                 \
      n = this->n_;                                 \
  BLOCKSIZE_BASE

#define WIDTH  \
  int width = p_real + 2 * p_complex + 2;           \
  if (p_complex == 0) width = p_real + 1;


template <typename T>
Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> build_b_ext (
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

  return bex;
}

};

#endif
