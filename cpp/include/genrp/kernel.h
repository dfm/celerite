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
    term.log_amp = log_amp;
    term.log_q = log_q;
    terms_.push_back(term);
  };

  void add_term (double log_amp, double log_q, double log_freq) {
    PeriodicTerm term;
    term.log_amp = log_amp;
    term.log_q = log_q;
    term.log_freq = log_freq;
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
    for (size_t i = 0; i < terms_.size(); ++i) beta(count++) = exp(-terms_[i].log_q);
    for (size_t i = 0; i < periodic_terms_.size(); ++i) {
      double re = exp(-periodic_terms_[i].log_q),
             im = 2*M_PI*exp(periodic_terms_[i].log_freq);
      beta(count++) = std::complex<double>(re, im);
      beta(count++) = std::complex<double>(re, -im);
    }
    return beta;
  };

  Eigen::VectorXd params () const {
    size_t count = 0;
    Eigen::VectorXd pars(size());
    for (size_t i = 0; i < terms_.size(); ++i) {
      pars(count++) = terms_[i].log_amp;
      pars(count++) = terms_[i].log_q;
    }
    for (size_t i = 0; i < periodic_terms_.size(); ++i) {
      pars(count++) = periodic_terms_[i].log_amp;
      pars(count++) = periodic_terms_[i].log_q;
      pars(count++) = periodic_terms_[i].log_freq;
    }
    return pars;
  };

  void params (const Eigen::VectorXd& pars) {
    size_t count = 0;
    for (size_t i = 0; i < terms_.size(); ++i) {
      terms_[i].log_amp = pars(count++);
      terms_[i].log_q = pars(count++);
    }
    for (size_t i = 0; i < periodic_terms_.size(); ++i) {
      periodic_terms_[i].log_amp = pars(count++);
      periodic_terms_[i].log_q = pars(count++);
      periodic_terms_[i].log_freq =pars(count++);
    }
  };

  double value (double dt) const {
    double result = 0.0;
    dt = fabs(dt);
    for (size_t i = 0; i < terms_.size(); ++i) {
      Term t = terms_[i];
      result += exp(t.log_amp) * exp(-exp(-t.log_q)*dt);
    }
    for (size_t i = 0; i < periodic_terms_.size(); ++i) {
      PeriodicTerm t = periodic_terms_[i];
      result += exp(t.log_amp)*exp(-exp(-t.log_q)*dt)*cos(2*M_PI*exp(t.log_freq)*dt);
    }
    return result;
  };

private:
  std::vector<Term> terms_;
  std::vector<PeriodicTerm> periodic_terms_;

};

};

#endif
