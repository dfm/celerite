#ifndef _GENRP_CARMA_H_
#define _GENRP_CARMA_H_

#include <cmath>
#include <cfloat>
#include <complex>
#include <Eigen/Dense>

namespace genrp {
namespace carma {

template <typename T1, typename T2>
inline bool isclose (const T1& a, const T2& b) {
  using std::abs;
  return (abs(a - b) <= 1e-6);
}

Eigen::VectorXcd roots_from_params (const Eigen::VectorXd& params) {
  unsigned n = params.rows();
  std::complex<double> b, c, arg;
  Eigen::VectorXcd roots(n);
  if (n == 0) return roots;
  if (n % 2 == 1) roots(n - 1) = -exp(params(n - 1));
  for (unsigned i = 0; i < n-1; i += 2) {
    b = exp(params(i+1));
    c = exp(params(i));
    arg = sqrt(b*b-4.0*c);
    roots(i) = 0.5 * (-b + arg);
    roots(i+1) = 0.5 * (-b - arg);
  }
  return roots;
}

Eigen::VectorXcd poly_from_roots (const Eigen::VectorXcd& roots) {
  unsigned n = roots.rows() + 1;
  if (n == 1) return Eigen::VectorXcd::Ones(1);
  Eigen::VectorXcd poly = Eigen::VectorXcd::Zero(n);
  poly(0) = -roots(0);
  poly(1) = 1.0;
  for (unsigned i = 1; i < n-1; ++i) {
    for (unsigned j = n-1; j >= 1; --j)
      poly(j) = poly(j - 1) - roots(i) * poly(j);
    poly(0) *= -roots(i);
  }
  return poly;
}

struct Prediction {
  double expectation;
  double variance;
};

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
  // Pre-compute the base lambda vector.
  for (unsigned i = 0; i < p_; ++i)
    lambda_base_(i) = exp(arroots_(i));

  // Compute the polynomial coefficients and rotate into the diagonalized space.
  alpha_ = poly_from_roots(arroots_);
  beta_ = poly_from_roots(maroots_);
  beta_ /= beta_(0);
};

void get_genrp_coeffs (
    Eigen::VectorXd& alpha_real, Eigen::VectorXd& beta_real,
    Eigen::VectorXd& alpha_complex_real, Eigen::VectorXd& alpha_complex_imag,
    Eigen::VectorXd& beta_complex_real, Eigen::VectorXd& beta_complex_imag
) const {
  alpha_real.resize(p_);
  beta_real.resize(p_);
  alpha_complex_real.resize(p_);
  alpha_complex_imag.resize(p_);
  beta_complex_real.resize(p_);
  beta_complex_imag.resize(p_);
  size_t p_real = 0, p_complex = 0;
  bool is_conj;
  std::complex<double> term1, term2, full_term;
  for (size_t k = 0; k < p_; ++k) {
    term1 = 0.0;
    term2 = 0.0;
    for (size_t l = 0; l < q_ + 1; ++l) {
      term1 += beta_[l] * pow(arroots_[k], l);
      term2 += beta_[l] * pow(-arroots_[k], l);
    }
    full_term = -sigma_ * sigma_ * term1 * term2 / arroots_[k].real();
    for (size_t l = 0; l < p_; ++l) {
      if (l != k)
        full_term /= (arroots_[l] - arroots_[k]) * (std::conj(arroots_[l]) + arroots_[k]);
    }

    // Check for and discard conjugate pairs.
    if (isclose(full_term.imag(), 0.0) && isclose(arroots_[k].imag(), 0.0)) {
      alpha_real[p_real] = 0.5 * full_term.real();
      beta_real[p_real] = -arroots_[k].real();
      p_real ++;
    } else {
      is_conj = false;
      for (size_t l = 0; l < p_complex; ++l) {
        if (isclose(alpha_complex_real[l], full_term.real()) &&
            isclose(alpha_complex_imag[l], -full_term.imag()) &&
            isclose(beta_complex_real[l], -arroots_[k].real()) &&
            isclose(beta_complex_imag[l], arroots_[k].imag())) {
          is_conj = true;
          break;
        }
      }
      if (!is_conj) {
        alpha_complex_real[p_complex] = full_term.real();
        alpha_complex_imag[p_complex] = full_term.imag();
        beta_complex_real[p_complex] = -arroots_[k].real();
        beta_complex_imag[p_complex] = -arroots_[k].imag();
        p_complex ++;
      }
    }
  }

  alpha_real = alpha_real.head(p_real);
  beta_real = beta_real.head(p_real);
  alpha_complex_real = alpha_complex_real.head(p_complex);
  alpha_complex_imag = alpha_complex_imag.head(p_complex);
  beta_complex_real = beta_complex_real.head(p_complex);
  beta_complex_imag = beta_complex_imag.head(p_complex);
};

void setup () {
  // Construct the rotation matrix for the diagonalized space.
  Eigen::MatrixXcd U(p_, p_);
  for (unsigned i = 0; i < p_; ++i)
    for (unsigned j = 0; j < p_; ++j)
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
  for (unsigned i = 0; i < p_; ++i)
    for (unsigned j = 0; j < p_; ++j)
      V_(i, j) /= arroots_(i) + std::conj(arroots_(j));
};

void reset (double t) {
  // Step 2 from Kelly et al.
  state_.time = t;
  state_.x = Eigen::VectorXcd::Zero(p_);
  state_.P = V_;
};

Prediction predict (double yerr) const {
  // Steps 3 and 9 from Kelly et al.
  Prediction pred;
  std::complex<double> tmp = b_ * state_.x;
  pred.expectation = tmp.real();
  tmp = b_ * state_.P * b_.adjoint();
  pred.variance = yerr * yerr + tmp.real();

  // Check the variance value for instability.
  if (pred.variance < 0.0) throw 1;
  return pred;
};

void update_state (const Prediction& pred, double y) {
  // Steps 4-6 and 10-12 from Kelly et al.
  Eigen::VectorXcd K = state_.P * b_.adjoint() / pred.variance;
  state_.x += (y - pred.expectation) * K;
  state_.P -= pred.variance * K * K.adjoint();
};

void advance_time (double dt) {
  // Steps 7 and 8 from Kelly et al.
  Eigen::VectorXcd lam = pow(lambda_base_, dt).matrix();
  state_.time += dt;
  for (unsigned i = 0; i < p_; ++i) state_.x(i) *= lam(i);
  state_.P = lam.asDiagonal() * (state_.P - V_) * lam.conjugate().asDiagonal() + V_;
};

double log_likelihood (const Eigen::VectorXd& t, const Eigen::VectorXd& y, const Eigen::VectorXd& yerr) {
  unsigned n = t.rows();
  double r, ll = n * log(2.0 * M_PI);
  Prediction pred;

  reset(t(0));
  for (unsigned i = 0; i < n; ++i) {
    // Integrate the Kalman filter.
    pred = predict(yerr(i));
    update_state(pred, y(i));
    if (i < n - 1) advance_time(t(i+1) - t(i));

    // Update the likelihood evaluation.
    r = y(i) - pred.expectation;
    ll += r * r / pred.variance + log(pred.variance);
  }

  return -0.5 * ll;
};

double psd (double f) const {
  std::complex<double> w(0.0, 2.0 * M_PI * f), num = 0.0, denom = 0.0;
  for (unsigned i = 0; i < q_+1; ++i)
    num += beta_(i) * pow(w, i);
  for (unsigned i = 0; i < p_+1; ++i)
    denom += alpha_(i) * pow(w, i);
  return sigma_*sigma_ * std::norm(num) / std::norm(denom);
};

double covariance (double tau) const {
  std::complex<double> n1, n2, norm, value = 0.0;

  for (unsigned k = 0; k < p_; ++k) {
    n1 = 0.0;
    n2 = 0.0;
    for (unsigned l = 0; l < q_+1; ++l) {
      n1 += beta_(l) * pow(arroots_(k), l);
      n2 += beta_(l) * pow(-arroots_(k), l);
    }
    norm = n1 * n2 / arroots_(k).real();
    for (unsigned l = 0; l < p_; ++l) {
      if (l != k)
        norm /= (arroots_(l) - arroots_(k)) * (std::conj(arroots_(l)) + arroots_(k));
    }
    value += norm * exp(arroots_(k) * tau);
  }

  return -0.5 * sigma_*sigma_ * value.real();
};

private:

  double sigma_;
  unsigned p_, q_;
  Eigen::VectorXcd arroots_, maroots_;
  Eigen::VectorXcd alpha_, beta_;
  Eigen::RowVectorXcd b_;

  Eigen::MatrixXcd V_;
  Eigen::ArrayXcd lambda_base_;
  State state_;
}; // class CARMASolver

}; // namespace carma
}; // namespace genrp

#endif // _GENRP_CARMA_H_
