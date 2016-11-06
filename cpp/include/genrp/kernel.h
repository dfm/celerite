#ifndef _GENRP_KERNEL_
#define _GENRP_KERNEL_

#include <cmath>
#include <vector>
#include <Eigen/Core>

namespace genrp {


class Term {
  friend class Kernel;
public:
  Term (double log_a, double log_q) {
    this->log_a(log_a);
    this->log_q(log_q);
  };

  size_t size () const { return 2; };

  void set_params (const double* params) {
    log_a(params[0]);
    log_q(params[1]);
  };

  void get_params (double* params) const {
    params[0] = log(a);
    params[1] = log(q);
  };

  double alpha () const {
    return fp2a * q;
  };

  double beta () const {
    return tpq;
  };

  virtual double psd (double f) const {
    double df = f / q;
    return 2.0 * a / (1.0 + df * df);
  };

  virtual double value (double dt) const {
    return fp2a * q * exp(-tpq * fabs(dt));
  };

  void log_a (double log_a) {
    a = exp(log_a);
    fp2a = 4.0 * M_PI * M_PI * a;
  };

  void log_q (double log_q) {
    q = exp(log_q);
    tpq = 2.0 * M_PI * q;
  };

protected:
  double a, q, fp2a, tpq;
};

class PeriodicTerm : public Term {
public:
  PeriodicTerm (double log_a, double log_q, double log_f) : Term(log_a, log_q) {
    this->log_f(log_f);
  };

  size_t size () const { return 3; };

  void set_params (const double* params) {
    log_a(params[0]);
    log_q(params[1]);
    log_f(params[2]);
  };

  void get_params (double* params) const {
    params[0] = log(a);
    params[1] = log(q);
    params[2] = log(f);
  };

  double beta_real () const {
    return tpq;
  };

  double beta_imag () const {
    return tpf;
  };

  void log_f (double log_f) {
    f = exp(log_f);
    tpf = 2.0 * M_PI * f;
  }

  double value (double dt) const {
    return fp2a * q * exp(-tpq * fabs(dt)) * cos(tpf * dt);
  };

  double psd (double f) const {
    double df, psd = 0.0;
    df = (f - this->f) / q;
    psd += a / (1.0 + df * df);
    df = (f + this->f) / q;
    psd += a / (1.0 + df * df);
    return psd;
  };

private:
  double f, tpf;
};


class Kernel {
public:
  Kernel () {};

  size_t size () const {
    size_t size = 0;
    for (size_t i = 0; i < terms_.size(); ++i) size += terms_[i].size();
    for (size_t i = 0; i < pterms_.size(); ++i) size += pterms_[i].size();
    return size;
  };

  size_t p () const { return terms_.size() + pterms_.size(); };
  size_t p_real () const { return terms_.size(); };
  size_t p_complex () const { return pterms_.size(); };

  void add_term (double log_amp, double log_q) {
    Term term(log_amp, log_q);
    terms_.push_back(term);
  };

  void add_term (double log_amp, double log_q, double log_freq) {
    PeriodicTerm term(log_amp, log_q, log_freq);
    pterms_.push_back(term);
  };

  Eigen::VectorXd alpha_real () const {
    Eigen::VectorXd alpha(terms_.size());
    for (size_t i = 0; i < terms_.size(); ++i)
      alpha(i) = terms_[i].alpha();
    return alpha;
  };

  Eigen::VectorXd beta_real () const {
    Eigen::VectorXd beta(terms_.size());
    for (size_t i = 0; i < terms_.size(); ++i)
      beta(i) = terms_[i].beta();
    return beta;
  };

  Eigen::VectorXd alpha_complex () const {
    Eigen::VectorXd alpha(pterms_.size());
    for (size_t i = 0; i < pterms_.size(); ++i)
      alpha(i) = pterms_[i].alpha();
    return alpha;
  };

  Eigen::VectorXd beta_complex_real () const {
    Eigen::VectorXd beta(pterms_.size());
    for (size_t i = 0; i < pterms_.size(); ++i)
      beta(i) = pterms_[i].beta_real();
    return beta;
  };

  Eigen::VectorXd beta_complex_imag () const {
    Eigen::VectorXd beta(pterms_.size());
    for (size_t i = 0; i < pterms_.size(); ++i)
      beta(i) = pterms_[i].beta_imag();
    return beta;
  };

  Eigen::VectorXd params () const {
    size_t i, count;
    Eigen::VectorXd pars(size());
    for (i = 0, count = 0; i < terms_.size(); ++i, count += 2)
      terms_[i].get_params(&(pars(count)));
    for (i = 0; i < pterms_.size(); ++i, count += 3)
      pterms_[i].get_params(&(pars(count)));
    return pars;
  };

  void params (const Eigen::VectorXd& pars) {
    size_t i, count;
    for (i = 0, count = 0; i < terms_.size(); ++i, count += 2)
      terms_[i].set_params(&(pars(count)));
    for (i = 0; i < pterms_.size(); ++i, count += 3)
      pterms_[i].set_params(&(pars(count)));
  };

  double value (double dt) const {
    double result = 0.0;
    for (size_t i = 0; i < terms_.size(); ++i) result += terms_[i].value(dt);
    for (size_t i = 0; i < pterms_.size(); ++i) result += pterms_[i].value(dt);
    return result;
  };

  double psd (double f) const {
    double result = 0.0;
    for (size_t i = 0; i < terms_.size(); ++i) result += terms_[i].psd(f);
    for (size_t i = 0; i < pterms_.size(); ++i) result += pterms_[i].psd(f);
    return result;
  };

private:
  std::vector<Term> terms_;
  std::vector<PeriodicTerm> pterms_;

};

};

#endif
