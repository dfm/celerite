#ifdef WITH_LAPACK

#ifndef _CELERITE_LAPACK_H_
#define _CELERITE_LAPACK_H_

#include <Eigen/Core>

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

extern "C" void dgbmv_(char* trans,
                       int* m,
                       int* n,
                       int* kl,
                       int* ku,
                       double* alpha,
                       double* a,
                       int* lda,
                       double* x,
                       int* incx,
                       double* beta,
                       double* y,
                       int* incy);

namespace celerite {

// Real band solver:
int band_factorize (int m, int kl, int ku, Eigen::MatrixXd& ab, Eigen::VectorXi& ipiv) {
  int n = ab.cols(),
      ldab = ab.outerStride(),
      info;
  dgbtrf_(&m, &n, &kl, &ku, ab.data(), &ldab, ipiv.data(), &info);
  return info;
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

Eigen::MatrixXd band_dot (int kl, int ku, const Eigen::MatrixXd& a, const Eigen::MatrixXd& x) {
  double one = 1.0;
  char trans = 'N';
  int ione = 1,
      n = a.cols(),
      lda = a.outerStride(),
      nrhs = x.cols();
  Eigen::MatrixXd y(n, nrhs);
  y.setConstant(0.0);
  for (int j = 0; j < nrhs; ++j)
    dgbmv_(&trans, &n, &n, &kl, &ku, &one, const_cast<double* >(a.data()), &lda, const_cast<double* >(x.col(j).data()), &ione, &one, y.col(j).data(), &ione);
  return y;
}

}

#endif

#endif
