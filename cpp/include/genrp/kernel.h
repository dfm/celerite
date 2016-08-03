#ifndef _GENRP_KERNEL_
#define _GENRP_KERNEL_

#include <cmath>
#include <vector>
#include <complex>
#include <Eigen/Dense>

namespace genrp {


class Term {
  friend class Kernel;
public:
  Term (double log_a, double log_q) : periodic(false) {
    this->log_a(log_a);
    this->log_q(log_q);
  };

  Term (double log_a, double log_q, double log_f) : periodic(true) {
    this->log_a(log_a);
    this->log_q(log_q);
    this->log_f(log_f);
  };

  size_t size () const { if (periodic) return 3; return 2; };
  size_t num_coeffs () const { if (periodic) return 2; return 1; };

  void set_params (const double* params) {
    log_a(params[0]);
    log_q(params[1]);
    if (periodic) log_f(params[2]);
  };

  void get_params (double* params) const {
    params[0] = log(a);
    params[1] = log(q);
    if (periodic) params[2] = log(f);
  };

  void get_alpha (double* alpha) const {
    alpha[0] = fp2a * q;
    if (periodic) {
      alpha[0] *= 0.5;
      alpha[1] = 0.5 * fp2a * q;
    }
  };

  void get_beta (std::complex<double>* beta) const {
    if (periodic) {
      beta[0] = std::complex<double>(tpq, tpf);
      beta[1] = std::complex<double>(tpq, -tpf);
      return;
    }
    beta[0] = tpq;
  };

  void carma_sigma2 (double* sigma2) const {
    if (periodic) {
      sigma2[0] = 2*M_PI*a * tpq * tpq;
      sigma2[1] = 2*M_PI*a * tpq * tpq;
      return;
    }
    sigma2[0] = 2.0 * 2*M_PI*a * tpq * tpq;
  };

  void carma_alpha (std::complex<double>* alpha) const {
    if (periodic) {
      alpha[0] = std::complex<double>(tpq, tpf);
      alpha[1] = std::complex<double>(tpq, -tpf);
      return;
    }
    alpha[0] = tpq;
  };

  double psd (double f) const {
    double df, psd = 0.0;
    if (periodic) {
      df = (f - this->f) / q;
      psd += a / (1.0 + df * df);
      df = (f + this->f) / q;
      psd += a / (1.0 + df * df);
    } else {
      df = f / q;
      psd += 2.0 * a / (1.0 + df * df);
    }
    return psd;
  };

  double value (double dt) const {
    double value = fp2a * q * exp(-tpq * fabs(dt));
    if (periodic) value *= cos(tpf * dt);
    return value;
  };

  void log_a (double log_a) {
    a = exp(log_a);
    fp2a = 4.0 * M_PI * M_PI * a;
  };

  void log_q (double log_q) {
    q = exp(log_q);
    tpq = 2.0 * M_PI * q;
  };

  void log_f (double log_f) {
    f = exp(log_f);
    tpf = 2.0 * M_PI * f;
  }

private:
  bool periodic;
  double a, q, f, fp2a, tpf, tpq;
};


class Kernel {
public:
  Kernel () {};

  size_t size () const {
    size_t size = 0;
    for (size_t i = 0; i < terms_.size(); ++i) size += terms_[i].size();
    return size;
  };
  size_t num_terms () const { return terms_.size(); };
  size_t num_coeffs () const {
    size_t size = 0;
    for (size_t i = 0; i < terms_.size(); ++i) size += terms_[i].num_coeffs();
    return size;
  };

  void add_term (double log_amp, double log_q) {
    Term term(log_amp, log_q);
    terms_.push_back(term);
  };

  void add_term (double log_amp, double log_q, double log_freq) {
    Term term(log_amp, log_q, log_freq);
    terms_.push_back(term);
  };

  Eigen::VectorXd alpha () const {
    size_t count = 0;
    Eigen::VectorXd alpha(num_coeffs());
    for (size_t i = 0; i < terms_.size(); ++i) {
      terms_[i].get_alpha(&(alpha(count)));
      count += terms_[i].num_coeffs();
    }
    return alpha;
  };

  Eigen::VectorXcd beta () const {
    size_t count = 0;
    Eigen::VectorXcd beta(num_coeffs());
    for (size_t i = 0; i < terms_.size(); ++i) {
      terms_[i].get_beta(&(beta(count)));
      count += terms_[i].num_coeffs();
    }
    return beta;
  };

  Eigen::VectorXd params () const {
    size_t count = 0;
    Eigen::VectorXd pars(size());
    for (size_t i = 0; i < terms_.size(); ++i) {
      terms_[i].get_params(&(pars(count)));
      count += terms_[i].size();
    }
    return pars;
  };

  void params (const Eigen::VectorXd& pars) {
    size_t count = 0;
    for (size_t i = 0; i < terms_.size(); ++i) {
      terms_[i].set_params(&(pars(count)));
      count += terms_[i].size();
    }
  };

  double value (double dt) const {
    double result = 0.0;
    for (size_t i = 0; i < terms_.size(); ++i) result += terms_[i].value(dt);
    return result;
  };

  double psd (double f) const {
    double result = 0.0;
    for (size_t i = 0; i < terms_.size(); ++i) result += terms_[i].psd(f);
    return result;
  };

  Eigen::VectorXd carma_sigma2s () const {
    size_t count = 0;
    Eigen::VectorXd sig2(num_coeffs());
    for (size_t i = 0; i < terms_.size(); ++i) {
      terms_[i].carma_sigma2(&(sig2(count)));
      count += terms_[i].num_coeffs();
    }
    return sig2;
  };

  Eigen::VectorXcd carma_alphas () const {
    size_t count = 0;
    Eigen::VectorXcd alpha(num_coeffs());
    for (size_t i = 0; i < terms_.size(); ++i) {
      terms_[i].carma_alpha(&(alpha(count)));
      count += terms_[i].num_coeffs();
    }
    return alpha;
  };

private:
  std::vector<Term> terms_;

};

};

#endif
