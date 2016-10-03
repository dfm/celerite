#ifndef _GENRP_PARAMETER_
#define _GENRP_PARAMETER_

#include <cmath>

namespace genrp {

template <typename T, typename P>
T get_parameter_value (const P& parameter) { return parameter.value(); }

template <>
double get_parameter_value (const double& parameter) { return parameter; }

template <typename T>
class Parameter {
public:
  Parameter () : value_(), frozen_(false) {};
  Parameter (const T& value) : value_(value), frozen_(false) {};
  Parameter (const T& value, bool frozen) : value_(value), frozen_(frozen) {};

  T value () const { return value_; };

private:
  T value_;
  bool frozen_;
};

template <typename T, typename P1, typename P2>
class ParameterSum {
public:
  ParameterSum (const P1& p1, const P2& p2) : p1_(p1), p2_(p2) {};
  T value () const {
    return get_parameter_value<T, P1>(p1_) + get_parameter_value<T, P2>(p2_);
  };

private:
  P1 p1_;
  P2 p2_;
};

template <typename T>
class ParameterExp {
public:
  ParameterExp (const Parameter<T>& parameter) : parameter_(parameter) {};
  T value () const { return exp(get_parameter_value<T>(parameter_)); };

private:
  Parameter<T> parameter_;
};

template <typename T, typename P>
ParameterExp<T, P> exp (const P& p) {
  return ParameterExp<T, P>(p);
}

};

#endif
