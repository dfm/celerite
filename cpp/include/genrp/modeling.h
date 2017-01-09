#ifndef _GENRP_MODELING_H_
#define _GENRP_MODELING_H_

#include <string>

namespace genrp {

class Parameter {
  Parameter () : frozen_(false) {};
  Parameter (std::string name) : name_(name), frozen_(false) {};
  Parameter (std::string name, double value, bool frozen = false) : name_(name), frozen_(frozen), value_(value) {};

  const std::string& get_name () const { return name_; };
  void set_name (std::string name) { name_ = name; };
  double get_value () const { return value_; };
  void set_value (double value) { value_ = value; };

  bool is_frozen () const { return frozen_; };
  void freeze () { frozen_ = true; };
  void thaw () { frozen_ = false; };

private:
  std::string name_;
  bool frozen_;
  double value_;
};

class Kernel {
  virtual p_real () const { return 0; };

};

};

#endif
