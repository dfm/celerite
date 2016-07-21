#ifndef _GENRP_KERNEL_
#define _GENRP_KERNEL_

#include <cmath>
#include <vector>
#include <complex>
#include <Eigen/Dense>

namespace genrp {

struct Term {
  double log_amp;
  double log_q;
};

struct PeriodicTerm {
  double log_amp;
  double log_q;
  double log_freq;
};

class Kernel {
public:
  Kernel () {};

  size_t size () const { return 2 * terms_.size() + 3 * periodic_terms_.size(); };

  void add_term (double log_amp, double log_q) {
    Term term;
    term.amp = exp(log_amp);
    term.q = exp(-log_q);
    terms_.push_back(term);
  };

  void add_term (double log_amp, double log_q, double log_freq) {
    PeriodicTerm term;
    term.amp = exp(log_amp);
    term.q = exp(-log_q);
    term.freq = 2*M_PI*exp(log_freq);
    periodic_terms_.push_back(term);
  };

  Eigen::VectorXd alpha () const {
    size_t count = 0;
    Eigen::VectorXd alpha(terms_.size() + 2*periodic_terms_.size());
    for (size_t i = 0; i < terms_.size(); ++i) alpha(count++) = exp(terms_[i].log_amp);
    for (size_t i = 0; i < periodic_terms_.size(); ++i) {
      double value = exp(periodic_terms_[i].log_amp);
      alpha(count++) = value;
      alpha(count++) = value;
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
      pars(count++) = log(terms_[i].log_amp);
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
    }
    for (size_t i = 0; i < periodic_terms_.size(); ++i) {
      periodic_terms_[i].amp = exp(pars(count++));
      periodic_terms_[i].q = exp(-pars(count++));
      periodic_terms_[i].freq = 2*M_PI*exp(pars(count++));
    }
  };

  double value (double dt) const {
    double result = 0.0;
    dt = fabs(dt);
    for (size_t i = 0; i < terms_.size(); ++i) {
      Term t = terms_[i];
      result += t.amp * exp(-t.q*dt);
    }
    for (size_t i = 0; i < periodic_terms_.size(); ++i) {
      PeriodicTerm t = periodic_terms_[i];
      result += t.amp*exp(-t.q*dt)*cos(t.freq*dt);
    }
    return result;
  };

  double grad (double dt, double* grad) const {
    size_t ind = 0;
    double result = 0.0, arg1, arg2, k0, k;
    dt = fabs(dt);
    for (size_t i = 0; i < terms_.size(); ++i) {
      Term t = terms_[i];
      arg1 = t.q*dt;
      k = t.amp * exp(-arg1);
      result += k;
      grad[ind++] = k;
      grad[ind++] = k * arg1;
    }
    for (size_t i = 0; i < periodic_terms_.size(); ++i) {
      PeriodicTerm t = periodic_terms_[i];
      arg1 = t.q*dt;
      arg2 = t.freq*dt;
      k0 = t.amp*exp(-arg1);
      k = k0*cos(arg2);
      result += k;
      grad[ind++] = k;
      grad[ind++] = k * arg1;
      grad[ind++] = -k0 * arg * sin(arg2);
    }
    return result;
  };

private:
  std::vector<Term> terms_;
  std::vector<PeriodicTerm> periodic_terms_;

};

};

#endif
