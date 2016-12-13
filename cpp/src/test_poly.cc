#include <vector>
#include <iostream>
#include <Eigen/Core>

#include "genrp/poly.h"

using Eigen::VectorXd;

#define ASSERT_ALL_CLOSE(NAME, VAR1, VAR2)                   \
{                                                            \
  if (VAR1.rows() != VAR2.rows()) {                          \
    std::cerr << "Test failed: " << #NAME << " - dimension mismatch" << std::endl; \
    return 1;                                                \
  }                                                          \
  double base, comp, delta;                                  \
  for (int iii = 0; iii < VAR1.rows(); ++iii) {              \
      base = VAR1[iii];                                      \
      comp = VAR2[iii];                                      \
      delta = std::abs(base - comp);                         \
      if (delta > 1e-10) {                                   \
        std::cerr << "Test failed: " << #NAME << " - " << iii << ": " << base << " != " << comp << std::endl; \
        return 1;                                            \
      }                                                      \
  }                                                          \
  std::cerr << "Test passed: " << #NAME << std::endl; \
}

int main (int argc, char* argv[])
{
  // Polymul
  VectorXd a(3), b(2), c(4), d;
  a << 3.0, 2.0, 1.0;
  b << -2.0, -1.0;
  c << -6.0, -7.0, -4.0, -1.0;
  d = genrp::polymul(a, b);
  ASSERT_ALL_CLOSE("polymul1", c, d);
  d = genrp::polymul(b, a);
  ASSERT_ALL_CLOSE("polymul2", c, d);

  // Polyadd
  c.resize(3);
  c << 3.0, 0.0, 0.0;
  d = genrp::polyadd(a, b);
  ASSERT_ALL_CLOSE("polyadd1", c, d);
  d = genrp::polyadd(b, a);
  ASSERT_ALL_CLOSE("polyadd2", c, d);

  // Polyval
  double v = genrp::polyval(a, 0.5);
  if (std::abs(v - 2.75) > 1e-10) {
    std::cerr << "Test failed: \"polyval\"" << std::endl;
    return 1;
  } else {
    std::cerr << "Test passed: \"polyval\"" << std::endl;
  }

  // Polyrem
  c.resize(1);
  c << 0.75;
  d = genrp::polyrem(a, b);
  ASSERT_ALL_CLOSE("polyrem1", c, d);
  d = genrp::polyrem(b, a);
  ASSERT_ALL_CLOSE("polyrem2", b, d);

  // Polyder
  c.resize(2);
  c << 6.0, 2.0;
  d = genrp::polyder(a);
  ASSERT_ALL_CLOSE("polyder1", c, d);
  c.resize(1);
  c << -2.0;
  d = genrp::polyder(b);
  ASSERT_ALL_CLOSE("polyder2", c, d);

  // Polyder
  a.resize(5);
  a << 1.0, 1.0, 0.0, -1.0, -1.0;
  std::vector<VectorXd> sturm = genrp::polysturm(a);
  if (sturm.size() != 5) {
    std::cerr << "Test failed: \"sturmshape\"" << std::endl;
    return 1;
  } else {
    std::cerr << "Test passed: \"sturmshape\"" << std::endl;
  }
  ASSERT_ALL_CLOSE("sturm1", a, sturm[0]);
  c.resize(4);
  c << 4.0, 3.0, 0.0, -1.0;
  ASSERT_ALL_CLOSE("sturm2", c, sturm[1]);
  c.resize(3);
  c << 3./16., 0.75, 15./16.;
  ASSERT_ALL_CLOSE("sturm3", c, sturm[2]);
  c.resize(2);
  c << -32., -64.;
  ASSERT_ALL_CLOSE("sturm4", c, sturm[3]);
  c.resize(1);
  c << -3./16.;
  ASSERT_ALL_CLOSE("sturm5", c, sturm[4]);

  // Count roots
  int nroots = genrp::polycountroots(a) + genrp::polycountroots(-a);
  if (sturm.size() != 2) {
    std::cerr << "Test failed: \"countroots\"" << std::endl;
    return 1;
  } else {
    std::cerr << "Test passed: \"countroots\"" << std::endl;
  }

  return 0;
}
