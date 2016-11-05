#ifndef _GENRP_BANDED_H_
#define _GENRP_BANDED_H_

#include <cmath>

namespace genrp {

#define SWAP(a,b) {dum=(a);(a)=(b);(b)=dum;}
#define TINY 1.0e-20

size_t get_index (size_t i, size_t j, size_t dx)
{
  return (i-1) + (j-1) * dx;
}

template <typename T>
void bandec(T* a, int n, int m1, int m2, T* al, int indx[], int* d)
{
  int i,j,k,l;
  int mm;
  T dum;
  mm=m1+m2+1;
  l=m1;
  for (i=1;i<=m1;i++) {
    for (j=m1+2-i;j<=mm;j++) a[get_index(i, j-l, n)]=a[get_index(i, j, n)];
    l--;
    for (j=mm-l;j<=mm;j++) a[get_index(i, j, n)]=0.0;
  }
  *d=1;
  l=m1;
  for (k=1;k<=n;k++) {
    dum=a[get_index(k, 1, n)];
    i=k;
    if (l < n) l++;
    for (j=k+1;j<=l;j++) {
      if (fabs(a[get_index(j, 1, n)]) > fabs(dum)) {
        dum=a[get_index(j, 1, n)];
        i=j;
      }
    }
    indx[k-1]=i;
    if (dum == 0.0) a[get_index(k, 1, n)]=TINY;
    if (i != k) {
      *d = -(*d);
      for (j=1;j<=mm;j++) SWAP(a[get_index(k, j, n)],a[get_index(i, j, n)])
    }
    for (i=k+1;i<=l;i++) {
      dum=a[get_index(i, 1, n)]/a[get_index(k, 1, n)];
      al[get_index(k, i-k, n)]=dum;
      for (j=2;j<=mm;j++) a[get_index(i, j-1, n)]=a[get_index(i, j, n)]-dum*a[get_index(k, j, n)];
      a[get_index(i, mm, n)]=0.0;
    }
  }
}

template <typename T>
void banbks(T* a, int n, int m1, int m2, T* al, int indx[], T* b) {
  int i,k,l;
  int mm;
  T dum;
  mm=m1+m2+1;
  l=m1;
  for (k=1;k<=n;k++) {
    i=indx[k-1];
    if (i != k) SWAP(b[k-1],b[i-1])
      if (l < n) l++;
    for (i=k+1;i<=l;i++) b[i-1] -= al[get_index(k, i-k, n)]*b[k-1];
  }
  l=1;
  for (i=n;i>=1;i--) {
    dum=b[i-1];
    for (k=2;k<=l;k++) dum -= a[get_index(i, k, n)]*b[k+i-2];
    b[i-1]=dum/a[get_index(i, 1, n)];
    if (l < mm) l++;
  }
}

#undef SWAP
#undef TINY

};

#endif

/* (C) Copr. 1986-92 Numerical Recipes Software ?421.1-9. */
