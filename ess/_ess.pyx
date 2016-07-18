# distutils: language = c++
from __future__ import division

cimport cython

import time
import numpy as np
cimport numpy as np
from libc.math cimport fabs

DTYPE = np.float64
ctypedef np.float64_t DTYPE_t
CDTYPE = np.complex128
ctypedef np.complex128_t CDTYPE_t

cdef extern from "complex":
    double complex exp(double complex)

cdef extern from "GRP.hpp":

    cdef cppclass GRP:
        GRP(int N, int m, double* alpha, double complex* beta, double* t, double* d)
        void assemble_Extended_Matrix()
        void factorize_Extended_Matrix()
        void obtain_Solution(double* rhs, double* solution)
        double obtain_Determinant()


cdef class CythonGRPSolver:

    cdef GRP* solver
    cdef unsigned int m
    cdef unsigned int N
    cdef np.ndarray alpha
    cdef np.ndarray beta
    cdef np.ndarray t
    cdef np.ndarray diagonal

    def __cinit__(self,
                  np.ndarray[DTYPE_t, ndim=1] alpha,
                  np.ndarray[CDTYPE_t, ndim=1] beta,
                  np.ndarray[DTYPE_t, ndim=1] t,
                  d=1e-10):
        if not np.all(np.diff(t) > 0.0):
            raise ValueError("times must be sorted")

        # Check the shape and roots:
        if alpha.shape[0] != beta.shape[0]:
            raise ValueError("dimension mismatch")
        if not np.allclose(0.0, np.sum(alpha).imag):
            raise ValueError("invalid alpha")
        if not np.allclose(0.0, np.sum(beta).imag):
            raise ValueError("invalid beta")

        # Save the parameters
        self.m = alpha.shape[0]
        self.N = t.shape[0]
        self.alpha = alpha
        self.beta = beta
        self.t = t

        # Check that the roots give a real covariance.
        cdef int p
        value = 0.0j
        for p in range(self.m):
            value += self.alpha[p] * np.exp(-self.beta[p])
        if not np.allclose(0.0, np.imag(value)):
            raise ValueError("invalid roots")

        try:
            d = float(d)
        except TypeError:
            d = np.atleast_1d(d)
            if d.shape[0] != self.N:
                raise ValueError("diagonal dimension mismatch")
        else:
            d = d + np.zeros_like(self.t)
        self.diagonal = d + np.sum(alpha).real

        self.solver = new GRP(
            self.N, self.m,
            <double*>(self.alpha.data),
            <double complex*>(self.beta.data),
            <double*>(self.t.data),
            <double*>(self.diagonal.data)
        )
        self.solver.assemble_Extended_Matrix()
        self.solver.factorize_Extended_Matrix()

    def __dealloc__(self):
        del self.solver

    property log_determinant:
        def __get__(self):
            return self.solver.obtain_Determinant()

    def apply_inverse(self, np.ndarray[DTYPE_t, ndim=1] y, in_place=False):
        if y.shape[0] != self.N:
            raise ValueError("dimension mismatch")

        if in_place:
            self.solver.obtain_Solution(<double*>y.data, <double*>y.data)
            return y

        cdef np.ndarray[DTYPE_t, ndim=1] alpha = np.empty_like(y, dtype=DTYPE)
        self.solver.obtain_Solution(<double*>y.data, <double*>alpha.data)
        return alpha

    def get_matrix(self):
        cdef double delta
        cdef double complex value
        cdef int i, j, p
        cdef np.ndarray[DTYPE_t, ndim=2] A = np.empty((self.N, self.N),
                                                      dtype=DTYPE)
        for i in range(self.N):
            A[i, i] = self.diagonal[i]
            for j in range(i + 1, self.N):
                delta = fabs(self.t[i] - self.t[j])
                value = 0.0j
                for p in range(self.m):
                    value += self.alpha[p] * exp(-self.beta[p] * delta)
                A[i, j] = value.real
                A[j, i] = value.real
        return A
