#include <iostream>
#include "genrp/parameter.h"

using genrp::Parameter;
using genrp::ParameterExp;

int main ()
{
  Parameter<double> p(1.0);

  std::cout << (exp<double, Parameter<double> >(p).value()) << std::endl;
  std::cout << ParameterExp<double, double>(1.0).value() << std::endl;

  return 0;
}
