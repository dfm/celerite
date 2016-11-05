#ifndef _GENRP_LAPACK_
#define _GENRP_LAPACK_

#include <complex>
#include "genrp/banded.h"

extern "C" void dgbtrf_(int* m,
                        int* n,
                        int* kl,
                        int* ku,
                        double* ab,
                        int* ldab,
                        int* ipiv,
                        int* info);

extern "C" void dgbtrs_(char* trans,
                        int* n,
                        int* kl,
                        int* ku,
                        int* nrhs,
                        double* ab,
                        int* ldab,
                        int* ipiv,
                        double* b,
                        int* ldb,
                        int* info);

namespace genrp {

// Real band solver:
int band_factorize (int m, int kl, int ku, Eigen::MatrixXd& a, Eigen::MatrixXd& al, Eigen::VectorXi& ipiv) {
  int n = a.cols(),
      ldab = a.outerStride(),
      info,
      d;
  Eigen::MatrixXd al(n, kl);
  bandec<double>(a.data(), n, kl, ku, al.data(), ipiv.data(), &d);
  return 0;
  // dgbtrf_(&m, &n, &kl, &ku, ab.data(), &ldab, ipiv.data(), &info);
  // return info;
}

int band_solve (int kl, int ku,
                const Eigen::MatrixXd& ab,
                const Eigen::VectorXi& ipiv,
                Eigen::MatrixXd& x) {
  char trans = 'N';
  int n = ab.cols(),
      ldab = ab.outerStride(),
      nrhs = x.cols(),
      info;
  dgbtrs_(&trans, &n, &kl, &ku, &nrhs, const_cast<double* >(ab.data()), &ldab, const_cast<int*>(ipiv.data()), x.data(), &n, &info);
  return info;
}

}

#endif
