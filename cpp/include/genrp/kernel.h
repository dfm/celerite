#ifndef _GENRP_KERNEL_H_
#define _GENRP_KERNEL_H_

#include <cmath>
#include <vector>
#include <Eigen/Core>

namespace genrp {


template <typename T>
class Term {
public:
  Term (const T& log_a, const T& log_q) {
    this->log_a(log_a);
    this->log_q(log_q);
  };

  size_t size () const { return 2; };

  void set_params (const T* const params) {
    this->log_a(params[0]);
    this->log_q(params[1]);
  };

  void get_params (T* params) const {
    params[0] = log(a);
    params[1] = log(q);
  };

  T alpha () const {
    return fp2a * q;
  };

  T beta () const {
    return tpq;
  };

  virtual T psd (double f) const {
    T df = f / q;
    return 2.0 * a / (1.0 + df * df);
  };

  virtual T value (double dt) const {
    return fp2a * q * exp(-tpq * fabs(dt));
  };

  void log_a (const T& log_a) {
    a = exp(log_a);
    fp2a = 4.0 * M_PI * M_PI * a;
  };

  void log_q (const T& log_q) {
    q = exp(log_q);
    tpq = 2.0 * M_PI * q;
  };

protected:
  T a, q, fp2a, tpq;
};

template <typename T>
class PeriodicTerm : public Term<T> {
public:
  PeriodicTerm (const T& log_a, const T& log_q, const T& log_f) : Term<T>(log_a, log_q) {
    this->log_f(log_f);
  };

  size_t size () const { return 3; };

  void set_params (const T* const params) {
    this->log_a(params[0]);
    this->log_q(params[1]);
    this->log_f(params[2]);
  };

  void get_params (T* params) const {
    params[0] = log(this->a);
    params[1] = log(this->q);
    params[2] = log(this->f);
  };

  T beta_real () const {
    return this->tpq;
  };

  T beta_imag () const {
    return this->tpf;
  };

  void log_f (const T& log_f) {
    f = exp(log_f);
    tpf = 2.0 * M_PI * f;
  }

  T value (double dt) const {
    return this->fp2a * this->q * exp(-this->tpq * fabs(dt)) * cos(this->tpf * dt);
  };

  T psd (double f) const {
    T df, psd = T(0.0);
    df = (f - this->f) / this->q;
    psd += this->a / (1.0 + df * df);
    df = (f + this->f) / this->q;
    psd += this->a / (1.0 + df * df);
    return psd;
  };

private:
  T f, tpf;
};


template <typename T>
class Kernel {
  typedef Eigen::Matrix<T, Eigen::Dynamic, 1> vector_t;
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

  void add_term (const T& log_amp, const T& log_q) {
    Term<T> term(log_amp, log_q);
    terms_.push_back(term);
  };

  void add_term (const T& log_amp, const T& log_q, const T& log_freq) {
    PeriodicTerm<T> term(log_amp, log_q, log_freq);
    pterms_.push_back(term);
  };

  vector_t alpha_real () const {
    vector_t alpha(terms_.size());
    for (size_t i = 0; i < terms_.size(); ++i)
      alpha(i) = terms_[i].alpha();
    return alpha;
  };

  vector_t beta_real () const {
    vector_t beta(terms_.size());
    for (size_t i = 0; i < terms_.size(); ++i)
      beta(i) = terms_[i].beta();
    return beta;
  };

  vector_t alpha_complex () const {
    vector_t alpha(pterms_.size());
    for (size_t i = 0; i < pterms_.size(); ++i)
      alpha(i) = pterms_[i].alpha();
    return alpha;
  };

  vector_t beta_complex_real () const {
    vector_t beta(pterms_.size());
    for (size_t i = 0; i < pterms_.size(); ++i)
      beta(i) = pterms_[i].beta_real();
    return beta;
  };

  vector_t beta_complex_imag () const {
    vector_t beta(pterms_.size());
    for (size_t i = 0; i < pterms_.size(); ++i)
      beta(i) = pterms_[i].beta_imag();
    return beta;
  };

  vector_t params () const {
    size_t i, count;
    vector_t pars(size());
    for (i = 0, count = 0; i < terms_.size(); ++i, count += 2)
      terms_[i].get_params(&(pars(count)));
    for (i = 0; i < pterms_.size(); ++i, count += 3)
      pterms_[i].get_params(&(pars(count)));
    return pars;
  };

  void params (const vector_t& pars) {
    size_t i, count;
    for (i = 0, count = 0; i < terms_.size(); ++i, count += 2)
      terms_[i].set_params(&(pars(count)));
    for (i = 0; i < pterms_.size(); ++i, count += 3)
      pterms_[i].set_params(&(pars(count)));
  };

  T value (double dt) const {
    T result = 0.0;
    for (size_t i = 0; i < terms_.size(); ++i) result += terms_[i].value(dt);
    for (size_t i = 0; i < pterms_.size(); ++i) result += pterms_[i].value(dt);
    return result;
  };

  T psd (double f) const {
    T result = 0.0;
    for (size_t i = 0; i < terms_.size(); ++i) result += terms_[i].psd(f);
    for (size_t i = 0; i < pterms_.size(); ++i) result += pterms_[i].psd(f);
    return result;
  };

private:
  std::vector<Term<T> > terms_;
  std::vector<PeriodicTerm<T> > pterms_;

};

};

#endif
