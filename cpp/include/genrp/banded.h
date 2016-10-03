#ifndef _GENRP_BANDED_
#define _GENRP_BANDED_

#include <cmath>

namespace genrp {

#define SWAP(a,b) {dum=(a);(a)=(b);(b)=dum;}
#define TINY 1.0e-20

size_t get_index (size_t i, size_t j, size_t dx, size_t dy)
{
  return (i-1) + (j-1) * dx;
}

template <typename T>
T bandec(T* a, size_t n, int m1, int m2, T* al, size_t indx[], T* d)
{
  size_t i,j,k,l;
  int mm;
  T dum;

  mm=m1+m2+1;
  l=m1;
  for (i=1;i<=m1;i++) {
    for (j=m1+2-i;j<=mm;j++) a[get_index(i, j-l, n, mm)]=a[get_index(i, j, n, mm)];
    l--;
    for (j=mm-l;j<=mm;j++) a[get_index(i, j, n, mm)]=0.0;
  }
  *d=1.0;
  l=m1;
  for (k=1;k<=n;k++) {
    dum=a[get_index(k, 1, n, mm)];
    i=k;
    if (l < n) l++;
    for (j=k+1;j<=l;j++) {
      if (std::abs(a[get_index(j, 1, n, mm)]) > std::abs(dum)) {
        dum=a[get_index(j, 1, n, mm)];
        i=j;
      }
    }
    indx[k]=i;
    if (dum == 0.0) a[get_index(k, 1, n, mm)]=TINY;
    if (i != k) {
      *d = -(*d);
      for (j=1;j<=mm;j++) SWAP(a[get_index(k, j, n, mm)],a[get_index(i, j, n, mm)])
    }
    for (i=k+1;i<=l;i++) {
      dum=a[get_index(i, 1, n, mm)]/a[get_index(k, 1, n, mm)];
      al[get_index(k, i-k, n, mm)]=dum;
      for (j=2;j<=mm;j++) a[get_index(i, j-1, n, mm)]=a[get_index(i, j, n, mm)]-dum*a[get_index(k, j, n, mm)];
      a[get_index(i, mm, n, mm)]=0.0;
    }
  }

  T logdet = 0.0;
  for (i = 1; i <= n; ++i) logdet += log(a[get_index(i, 1, n, mm)]);
  return logdet;
}

template <typename T>
void banbks(const T* a, size_t n, int m1, int m2, const T* al, const size_t indx[], T b[])
{
  size_t i,k,l;
  int mm;
  T dum;

  mm=m1+m2+1;
  l=m1;
  for (k=1;k<=n;k++) {
    i=indx[k];
    if (i != k) SWAP(b[k-1],b[i-1])
      if (l < n) l++;
    for (i=k+1;i<=l;i++) b[i-1] -= al[get_index(k, i-k, n, mm)]*b[k-1];
  }
  l=1;
  for (i=n;i>=1;i--) {
    dum=b[i-1];
    for (k=2;k<=l;k++) dum -= a[get_index(i, k, n, mm)]*b[k+i-2];
    b[i-1]=dum/a[get_index(i, 1, n, mm)];
    if (l < mm) l++;
  }
}

#undef SWAP
#undef TINY

};

#endif

/* (C) Copr. 1986-92 Numerical Recipes Software ?421.1-9. */
