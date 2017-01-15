#include <vector>
#include <iostream>
#include <Eigen/Dense>

#include "celerite/sband.h"

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
  int n = 10, p = 3;
  Eigen::VectorXd b(n);
  celerite::SymmetricBandMatrix<double> A(n, p);
  for (int i = 0; i < n; ++i) {
    b.row(i).setConstant(i);
    A(i, i) = 10 + 0.1*i;
    for (int j = i+1; j < std::min(i+p, n); ++j) {
      double val = 0.1 * (i + 1);
      A(j, i) = val;
    }
  }


  Eigen::MatrixXd AA = A.toDenseMatrix();
  Eigen::VectorXd x = AA.ldlt().solve(b);

  A.chofactor();
  A.chosolve(b);

  ASSERT_ALL_CLOSE(blah, x, b)

  return 0;
}
