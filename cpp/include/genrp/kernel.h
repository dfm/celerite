#ifndef _GENRP_KERNEL_
#define _GENRP_KERNEL_

#include <cmath>
#include <vector>
#include <complex>
#include <Eigen/Dense>

namespace genrp {

struct Term {
  double factor;
  double amp;
  double q;
};

struct PeriodicTerm {
  double factor;
  double amp;
  double q;
  double freq;
};

class Kernel {
public:
  Kernel () {};

  size_t size () const { return 2 * terms_.size() + 3 * periodic_terms_.size(); };
  size_t num_terms () const { return terms_.size() + periodic_terms_.size(); };
  size_t num_coeffs () const { return terms_.size() + 2 * periodic_terms_.size(); };

  void add_term (double log_amp, double log_q) {
    Term term;
    term.amp = exp(log_amp);
    term.q = exp(-log_q);
    term.factor = 2.0 * M_PI * term.q * term.amp;
    terms_.push_back(term);
  };

  void add_term (double log_amp, double log_q, double log_freq) {
    PeriodicTerm term;
    term.amp = exp(log_amp);
    term.q = exp(-log_q);
    term.factor = 2.0*M_PI * term.q * term.amp;
    term.freq = 2.0*M_PI * exp(log_freq);
    periodic_terms_.push_back(term);
  };

  Eigen::VectorXd alpha () const {
    size_t count = 0;
    Eigen::VectorXd alpha(terms_.size() + 2*periodic_terms_.size());
    for (size_t i = 0; i < terms_.size(); ++i) alpha(count++) = terms_[i].factor;
    for (size_t i = 0; i < periodic_terms_.size(); ++i) {
      double value = periodic_terms_[i].factor;
      alpha(count++) = 0.5 * value;
      alpha(count++) = 0.5 * value;
    }
    return alpha;
  };

  Eigen::VectorXcd beta () const {
    size_t count = 0;
    Eigen::VectorXcd beta(terms_.size() + 2*periodic_terms_.size());
    for (size_t i = 0; i < terms_.size(); ++i) beta(count++) = terms_[i].q;
    for (size_t i = 0; i < periodic_terms_.size(); ++i) {
      double re = periodic_terms_[i].q,
             im = periodic_terms_[i].freq;
      beta(count++) = std::complex<double>(re, im);
      beta(count++) = std::complex<double>(re, -im);
    }
    return beta;
  };

  Eigen::VectorXd params () const {
    size_t count = 0;
    Eigen::VectorXd pars(size());
    for (size_t i = 0; i < terms_.size(); ++i) {
      pars(count++) = log(terms_[i].amp);
      pars(count++) = -log(terms_[i].q);
    }
    for (size_t i = 0; i < periodic_terms_.size(); ++i) {
      pars(count++) = log(periodic_terms_[i].amp);
      pars(count++) = -log(periodic_terms_[i].q);
      pars(count++) = log(periodic_terms_[i].freq / (2*M_PI));
    }
    return pars;
  };

  void params (const Eigen::VectorXd& pars) {
    size_t count = 0;
    for (size_t i = 0; i < terms_.size(); ++i) {
      terms_[i].amp = exp(pars(count++));
      terms_[i].q = exp(-pars(count++));
      terms_[i].factor = 2.0*M_PI * terms_[i].q * terms_[i].amp;
    }
    for (size_t i = 0; i < periodic_terms_.size(); ++i) {
      periodic_terms_[i].amp = exp(pars(count++));
      periodic_terms_[i].q = exp(-pars(count++));
      periodic_terms_[i].factor = 2.0*M_PI * periodic_terms_[i].q * periodic_terms_[i].amp;
      periodic_terms_[i].freq = 2.0*M_PI*exp(pars(count++));
    }
  };

  double value (double dt) const {
    double result = 0.0;
    dt = fabs(dt);
    for (size_t i = 0; i < terms_.size(); ++i) {
      Term t = terms_[i];
      result += t.factor * exp(-t.q*dt);
    }
    for (size_t i = 0; i < periodic_terms_.size(); ++i) {
      PeriodicTerm t = periodic_terms_[i];
      result += t.factor * exp(-t.q*dt) * cos(t.freq*dt);
    }
    return result;
  };

  double psd (double f) const {
    double result = 0.0,
           w = 2*M_PI*f;
    for (size_t i = 0; i < terms_.size(); ++i) {
      Term t = terms_[i];
      double dw = w / t.q;
      result += 2.0 * t.amp / (1.0 + dw*dw);
    }
    for (size_t i = 0; i < periodic_terms_.size(); ++i) {
      PeriodicTerm t = periodic_terms_[i];
      double dw = (w - t.freq) / t.q;
      result += t.amp / (1.0 + dw*dw);
      dw = (w + t.freq) / t.q;
      result += t.amp / (1.0 + dw*dw);
    }
    return result;
  };

private:
  std::vector<Term> terms_;
  std::vector<PeriodicTerm> periodic_terms_;

};

};

#endif
