#ifndef _CELERITE_CARMA_H_
#define _CELERITE_CARMA_H_

#include <cmath>
#include <cfloat>
#include <complex>
#include <Eigen/Dense>

#include "celerite/utils.h"
#include "celerite/exceptions.h"

namespace celerite {
namespace carma {

inline Eigen::VectorXcd roots_from_params (const Eigen::VectorXd& params) {
  int n = params.rows();
  std::complex<double> b, c, arg;
  Eigen::VectorXcd roots(n);
  if (n == 0) return roots;
  if (n % 2 == 1) roots(n - 1) = -exp(params(n - 1));
  for (int i = 0; i < n-1; i += 2) {
    b = exp(params(i+1));
    c = exp(params(i));
    arg = sqrt(b*b-4.0*c);
    roots(i) = 0.5 * (-b + arg);
    roots(i+1) = 0.5 * (-b - arg);
  }
  return roots;
}

inline Eigen::VectorXcd poly_from_roots (const Eigen::VectorXcd& roots) {
  int n = roots.rows() + 1;
  if (n == 1) return Eigen::VectorXcd::Ones(1);
  Eigen::VectorXcd poly = Eigen::VectorXcd::Zero(n);
  poly(0) = -roots(0);
  poly(1) = 1.0;
  for (int i = 1; i < n-1; ++i) {
    for (int j = n-1; j >= 1; --j)
      poly(j) = poly(j - 1) - roots(i) * poly(j);
    poly(0) *= -roots(i);
  }
  return poly;
}

struct State {
  double time;
  Eigen::VectorXcd x;
  Eigen::MatrixXcd P;
};

class CARMASolver {
public:
CARMASolver (double log_sigma, Eigen::VectorXd arpars, Eigen::VectorXd mapars)
  : sigma_(exp(log_sigma)), p_(arpars.rows()), q_(mapars.rows()),
    arroots_(roots_from_params(arpars)), maroots_(roots_from_params(mapars)),
    b_(Eigen::MatrixXcd::Zero(1, p_)), lambda_base_(p_)
{
  if (q_ >= p_) throw dimension_mismatch();

  // Pre-compute the base lambda vector.
  for (int i = 0; i < p_; ++i)
    lambda_base_(i) = exp(arroots_(i));

  // Compute the polynomial coefficients and rotate into the diagonalized space.
  alpha_ = poly_from_roots(arroots_);
  beta_ = poly_from_roots(maroots_);
  beta_ /= beta_(0);

  setup();
};

void get_celerite_coeffs (
    Eigen::VectorXd& alpha_real, Eigen::VectorXd& beta_real,
    Eigen::VectorXd& alpha_complex_real, Eigen::VectorXd& alpha_complex_imag,
    Eigen::VectorXd& beta_complex_real, Eigen::VectorXd& beta_complex_imag
) const {
  Eigen::VectorXd ar(p_), cr(p_),
                  a(p_), b(p_), c(p_), d(p_);
  int p_real = 0, p_complex = 0;
  bool is_conj;
  std::complex<double> term1, term2, full_term;
  for (int k = 0; k < p_; ++k) {
    term1 = log(beta_[0]);
    term2 = log(beta_[0]);
    for (int l = 1; l < q_ + 1; ++l) {
      term1 = _logsumexp(term1, log(beta_[l]) + std::complex<double>(l) * log(arroots_[k]));
      term2 = _logsumexp(term2, log(beta_[l]) + std::complex<double>(l) * log(-arroots_[k]));
    }
    full_term = 2.0 * log(sigma_) + term1 + term2 - log(-arroots_[k].real());
    for (int l = 0; l < p_; ++l) {
      if (l != k)
        full_term -= log(arroots_[l] - arroots_[k]) + log(std::conj(arroots_[l]) + arroots_[k]);
    }
    full_term = exp(full_term);

    // Check for and discard conjugate pairs.
    if (isclose(full_term.imag(), 0.0) && isclose(arroots_[k].imag(), 0.0)) {
      ar[p_real] = 0.5 * full_term.real();
      cr[p_real] = -arroots_[k].real();
      p_real ++;
    } else {
      is_conj = false;
      for (int l = 0; l < p_complex; ++l) {
        if (isclose(a[l], full_term.real()) &&
            isclose(b[l], -full_term.imag()) &&
            isclose(c[l], -arroots_[k].real()) &&
            isclose(d[l], arroots_[k].imag())) {
          is_conj = true;
          break;
        }
      }
      if (!is_conj) {
        a[p_complex] = full_term.real();
        b[p_complex] = full_term.imag();
        c[p_complex] = -arroots_[k].real();
        d[p_complex] = -arroots_[k].imag();
        p_complex ++;
      }
    }
  }

  // Copy the results
  alpha_real.resize(p_real);
  beta_real.resize(p_real);
  alpha_complex_real.resize(p_complex);
  alpha_complex_imag.resize(p_complex);
  beta_complex_real.resize(p_complex);
  beta_complex_imag.resize(p_complex);
  for (int i = 0; i < p_real; ++i) {
    alpha_real(i) = ar(i);
    beta_real(i) = cr(i);
  }
  for (int i = 0; i < p_complex; ++i) {
    alpha_complex_real(i) = a(i);
    alpha_complex_imag(i) = b(i);
    beta_complex_real(i) = c(i);
    beta_complex_imag(i) = d(i);
  }
};

void setup () {
  // Construct the rotation matrix for the diagonalized space.
  Eigen::MatrixXcd U(p_, p_);
  for (int i = 0; i < p_; ++i)
    for (int j = 0; j < p_; ++j)
      U(i, j) = pow(arroots_(j), i);
  b_.head(q_ + 1) = beta_;
  b_ = b_ * U;

  // Compute V.
  Eigen::VectorXcd e = Eigen::VectorXcd::Zero(p_);
  e(p_ - 1) = sigma_;

  // J = U \ e
  Eigen::FullPivLU<Eigen::MatrixXcd> lu(U);
  Eigen::VectorXcd J = lu.solve(e);

  // V_ij = -J_i J_j^* / (r_i + r_j^*)
  V_ = -J * J.adjoint();
  for (int i = 0; i < p_; ++i)
    for (int j = 0; j < p_; ++j)
      V_(i, j) /= arroots_(i) + std::conj(arroots_(j));
};

void reset (double t) {
  // Step 2 from Kelly et al.
  state_.time = t;
  state_.x.resize(p_);
  state_.x.setConstant(0.0);
  state_.P = V_;
};

void predict (double yerr) {
  // Steps 3 and 9 from Kelly et al.
  expectation_ = 0.0;
  variance_ = yerr * yerr;
  for (int i = 0; i < p_; ++i) {
    expectation_ += (b_(i) * state_.x(i)).real();
    for (int j = 0; j < p_; ++j) {
      variance_ += (b_(i) * state_.P(i, j) * std::conj(b_(j))).real();
    }
  }

  // Check the variance value for instability.
  if (variance_ < 0.0)
    throw carma_exception();
};

void update_state (double y) {
  // Steps 4-6 and 10-12 from Kelly et al.
  Eigen::VectorXcd K(p_, p_);
  K.setConstant(0.0);
  for (int i = 0; i < p_; ++i) {
    for (int j = 0; j < p_; ++j)
      K(i) += state_.P(i, j) * std::conj(b_(j)) / variance_;
    state_.x(i) += (y - expectation_) * K(i);
  }

  for (int i = 0; i < p_; ++i)
    for (int j = 0; j < p_; ++j)
      state_.P(i, j) -= variance_ * K(i) * std::conj(K(j));
};

void advance_time (double dt) {
  // Steps 7 and 8 from Kelly et al.
  Eigen::VectorXcd lam(p_);
  state_.time += dt;
  for (int i = 0; i < p_; ++i) {
    lam(i) = pow(lambda_base_(i), dt);
    state_.x(i) *= lam(i);
  }
  Eigen::MatrixXcd P = state_.P;
  state_.P = V_;
  for (int i = 0; i < p_; ++i) {
    for (int j = 0; j < p_; ++j) {
      state_.P(i, j) += lam(i) * (P(i, j) - V_(i, j)) * std::conj(lam(j));
    }
  }
};

double log_likelihood (const Eigen::VectorXd& t, const Eigen::VectorXd& y, const Eigen::VectorXd& yerr) {
  int n = t.rows();
  if (y.rows() != n || yerr.rows() != n) throw dimension_mismatch();
  double r, ll = n * log(2.0 * M_PI);

  reset(t(0));
  for (int i = 0; i < n; ++i) {
    // Integrate the Kalman filter.
    predict(yerr(i));
    update_state(y(i));
    if (i < n - 1) advance_time(t(i+1) - t(i));

    // Update the likelihood evaluation.
    r = y(i) - expectation_;
    ll += r * r / variance_ + log(variance_);
  }

  return -0.5 * ll;
};

double psd (double f) const {
  std::complex<double> w(0.0, 2.0 * M_PI * f), num = 0.0, denom = 0.0;
  for (int i = 0; i < q_+1; ++i)
    num += beta_(i) * pow(w, i);
  for (int i = 0; i < p_+1; ++i)
    denom += alpha_(i) * pow(w, i);
  return sigma_*sigma_ * std::norm(num) / std::norm(denom);
};

double covariance (double tau) const {
  std::complex<double> n1, n2, norm, value = 0.0;

  for (int k = 0; k < p_; ++k) {
    n1 = 0.0;
    n2 = 0.0;
    for (int l = 0; l < q_+1; ++l) {
      n1 += beta_(l) * pow(arroots_(k), l);
      n2 += beta_(l) * pow(-arroots_(k), l);
    }
    norm = n1 * n2 / arroots_(k).real();
    for (int l = 0; l < p_; ++l) {
      if (l != k)
        norm /= (arroots_(l) - arroots_(k)) * (std::conj(arroots_(l)) + arroots_(k));
    }
    value += norm * exp(arroots_(k) * tau);
  }

  return -0.5 * sigma_*sigma_ * value.real();
};

private:

  double sigma_;
  int p_, q_;
  Eigen::VectorXcd arroots_, maroots_;
  Eigen::VectorXcd alpha_, beta_;
  Eigen::RowVectorXcd b_;

  Eigen::MatrixXcd V_;
  Eigen::ArrayXcd lambda_base_;
  State state_;

  // Prediction
  double expectation_, variance_;

}; // class CARMASolver

}; // namespace carma
}; // namespace celerite

#endif // _CELERITE_CARMA_H_
