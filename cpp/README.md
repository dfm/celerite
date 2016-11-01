Prerequisites
-------------

Eigen, LAPACK.


Building
--------

```
cmake .
make
make test
```

Example usage
-------------

```
#include <iostream>
#include <Eigen/Core>
#include "genrp/genrp.h"

int main ()
{
  genrp::Kernel kernel;
  kernel.add_term(1.0, 0.1);
  kernel.add_term(0.1, 2.0, 1.6);
  genrp::GaussianProcess<genrp::BandSolver> gp(kernel);

  int N = 1024;
  Eigen::VectorXd x = Eigen::VectorXd::Random(N),
                  yerr = Eigen::VectorXd::Random(N),
                  y;
  yerr.array() *= 0.1;
  yerr.array() += 4.0;
  std::sort(x.data(), x.data() + x.size());
  y = sin(x.array());

  gp.compute(x, yerr);
  std::cout << gp.log_likelihood(y) << std::endl;

  return 0;
}
```
