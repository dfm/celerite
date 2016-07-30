#ifndef _GENRP_LAPACK_
#define _GENRP_LAPACK_

#include <complex>

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

extern "C" void zgbtrf_(int* m,
                        int* n,
                        int* kl,
                        int* ku,
                        std::complex<double>* ab,
                        int* ldab,
                        int* ipiv,
                        int* info);

extern "C" void zgbtrs_(char* trans,
                        int* n,
                        int* kl,
                        int* ku,
                        int* nrhs,
                        std::complex<double>* ab,
                        int* ldab,
                        int* ipiv,
                        std::complex<double>* b,
                        int* ldb,
                        int* info);

namespace genrp {

int band_factorize (Eigen::internal::BandMatrix<std::complex<double> >& ab, Eigen::VectorXi& ipiv) {
  int m = ab.rows(),
      n = ab.cols(),
      kl = ab.subs(),
      ku = ab.supers() - kl,
      ldab = ab.coeffs().outerStride(),
      info;
  zgbtrf_(&m, &n, &kl, &ku, ab.coeffs().data(), &ldab, ipiv.data(), &info);
  return info;
}

int band_solve (int kl, int ku,
                const Eigen::MatrixXcd& ab,
                const Eigen::VectorXi& ipiv,
                Eigen::MatrixXcd& x) {
  char trans = 'N';
  int m = ab.rows(),
      n = ab.cols(),
      ldab = ab.outerStride(),
      nrhs = x.cols(),
      info;
  zgbtrs_(&trans, &n, &kl, &ku, &nrhs, const_cast<std::complex<double>* >(ab.data()), &ldab, const_cast<int*>(ipiv.data()), x.data(), &n, &info);
  return info;
}

}

#endif
