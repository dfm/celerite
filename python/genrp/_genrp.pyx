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

cdef extern from "genrp/genrp.h" namespace "genrp":

    cdef cppclass Kernel:
        Kernel()
        void add_term (double log_amp, double log_q)
        void add_term (double log_amp, double log_q, double log_freq)
        double value (double dt) const
        double psd (double w) const

    cdef cppclass GenRPSolver[T]:
        GenRPSolver (size_t m, const double* alpha, const T* beta)
        void compute (size_t N, const double* t, const double* d)
        void solve (const double* b, double* x) const
        double solve_dot (const double* b) const
        double log_determinant () const

    cdef cppclass GaussianProcess:
        GaussianProcess (Kernel kernel)
        size_t size () const
        size_t num_terms () const
        void compute (size_t N, const double* x, const double* yerr)
        double log_likelihood (const double* y) const
        double kernel_value (double dt) const
        double kernel_psd (double w) const
        void get_params (double* pars) const
        void set_params (const double* pars)
        void get_alpha (double* alpha) const
        void get_beta (double complex* beta) const


cdef class GP:

    cdef GaussianProcess* gp
    cdef Kernel kernel
    cdef np.ndarray x
    cdef np.ndarray yerr
    cdef int _computed
    cdef int _has_x

    def __cinit__(self):
        self.gp = new GaussianProcess(self.kernel)
        self._computed = 0
        self._has_x = 0

    def __dealloc__(self):
        del self.gp

    def __len__(self):
        return self.gp.size()

    property computed:
        def __get__(self):
            return bool(self._computed)

    property alpha:
        def __get__(self):
            cdef np.ndarray[DTYPE_t] a = np.empty(self.gp.num_terms(), dtype=DTYPE)
            self.gp.get_alpha(<double*>a.data)
            return a

    property beta:
        def __get__(self):
            cdef np.ndarray[CDTYPE_t] b = np.empty(self.gp.num_terms(), dtype=CDTYPE)
            self.gp.get_beta(<double complex*>b.data)
            return b

    property params:
        def __get__(self):
            cdef np.ndarray[DTYPE_t] p = np.empty(self.gp.size(), dtype=DTYPE)
            self.gp.get_params(<double*>p.data)
            return p

        def __set__(self, params):
            cdef np.ndarray[DTYPE_t, ndim=1] p = \
                np.atleast_1d(params).astype(DTYPE)
            if p.shape[0] != self.gp.size():
                raise ValueError("dimension mismatch")
            self.gp.set_params(<double*>p.data)

    def add_term(self, double log_amp, double log_q, log_freq=None):
        if log_freq is None:
            self.kernel.add_term(log_amp, log_q)
        else:
            self.kernel.add_term(log_amp, log_q, log_freq)
        del self.gp
        self.gp = new GaussianProcess(self.kernel)
        self._computed = 0

    def get_matrix(self, np.ndarray[DTYPE_t, ndim=1] x1, x2=None):
        if x2 is None:
            x2 = x1
        cdef np.ndarray[DTYPE_t, ndim=2] K = x1[:, None] - x2[None, :]
        cdef int i, j
        for i in range(x1.shape[0]):
            for j in range(x2.shape[0]):
                K[i, j] = self.gp.kernel_value(K[i, j])
        return K

    def get_psd(self, np.ndarray[DTYPE_t, ndim=1] w):
        cdef np.ndarray[DTYPE_t, ndim=1] psd = np.empty_like(w, dtype=DTYPE)
        cdef int i
        for i in range(w.shape[0]):
            psd[i] = self.gp.kernel_psd(w[i])
        return psd

    def compute(self, np.ndarray[DTYPE_t, ndim=1] x, np.ndarray[DTYPE_t, ndim=1] yerr):
        if x.shape[0] != yerr.shape[0]:
            raise ValueError("dimension mismatch")
        cdef int i
        for i in range(x.shape[0] - 1):
            if x[i+1] <= x[i]:
                raise ValueError("the time series must be ordered")
        self.gp.compute(x.shape[0], <double*>x.data, <double*>yerr.data)
        self.x = x
        self.yerr = yerr
        self._computed = 1
        self._has_x = 1

    def log_likelihood(self, np.ndarray[DTYPE_t, ndim=1] y):
        if self._has_x:
            if y.shape[0] != self.x.shape[0]:
                raise ValueError("dimension mismatch")
        if self._computed == 0:
            if self._has_x == 0:
                raise RuntimeError("must call 'compute' first")
            self.gp.compute(self.x.shape[0], <double*>self.x.data,
                            <double*>self.yerr.data)
            self._computed = 1
        return self.gp.log_likelihood(<double*>y.data)

    def sample(self, np.ndarray[DTYPE_t, ndim=1] x, double tiny=1e-10):
        cdef np.ndarray[DTYPE_t, ndim=2] K = self.get_matrix(x)
        K[np.diag_indices_from(K)] += tiny
        return np.random.multivariate_normal(np.zeros_like(x), K)


cdef class Solver:

    cdef GenRPSolver[double complex]* solver
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
        self.diagonal = d

        self.solver = new GenRPSolver[double complex](
            self.m,
            <double*>(self.alpha.data),
            <double complex*>(self.beta.data),
        )
        self.solver.compute(
            self.N,
            <double*>(self.t.data),
            <double*>(self.diagonal.data)
        )

    def __dealloc__(self):
        del self.solver

    property log_determinant:
        def __get__(self):
            return self.solver.log_determinant()

    def apply_inverse(self, np.ndarray[DTYPE_t, ndim=1] y, in_place=False):
        if y.shape[0] != self.N:
            raise ValueError("dimension mismatch")

        if in_place:
            self.solver.solve(<double*>y.data, <double*>y.data)
            return y

        cdef np.ndarray[DTYPE_t, ndim=1] alpha = np.empty_like(y, dtype=DTYPE)
        self.solver.solve(<double*>y.data, <double*>alpha.data)
        return alpha

    def get_matrix(self):
        cdef double delta
        cdef double complex value
        cdef int i, j, p
        cdef np.ndarray[DTYPE_t, ndim=2] A = np.empty((self.N, self.N),
                                                      dtype=DTYPE)
        for i in range(self.N):
            A[i, i] = self.diagonal[i] + self.alpha.sum().real
            for j in range(i + 1, self.N):
                delta = fabs(self.t[i] - self.t[j])
                value = 0.0j
                for p in range(self.m):
                    value += self.alpha[p] * exp(-self.beta[p] * delta)
                A[i, j] = value.real
                A[j, i] = value.real
        return A
