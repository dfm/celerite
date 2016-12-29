# distutils: language = c++
from __future__ import division

__all__ = ["get_library_version", "Solver"]

cimport cython

import time
import numpy as np
cimport numpy as np
from libc.math cimport fabs, exp, cos

DTYPE = np.float64
ctypedef np.float64_t DTYPE_t

cdef extern from "genrp/version.h":
    cdef int GENRP_VERSION_MAJOR
    cdef int GENRP_VERSION_MINOR
    cdef int GENRP_VERSION_REVISION

def get_library_version():
    return "{0}.{1}.{2}".format(
        GENRP_VERSION_MAJOR,
        GENRP_VERSION_MINOR,
        GENRP_VERSION_REVISION
    )

cdef extern from "Eigen/Core" namespace "Eigen":
    cdef cppclass VectorXd:
        VectorXd ()
        double* data()
        double operator () (int i)
        int rows ()

    cdef cppclass Map[T]:
        Map(double*, int)

cdef extern from "genrp/genrp.h" namespace "genrp":
    cdef double get_kernel_value(
        size_t p_real,
        const double* const alpha_real, const double* const beta_real,
        size_t p_complex,
        const double* const alpha_complex_real,
        const double* const alpha_complex_imag,
        const double* const beta_complex_real,
        const double* const beta_complex_imag,
        double tau
    )

cdef extern from "genrp/genrp.h" namespace "genrp::solver":
    cdef cppclass BandSolver[T]:
        BandSolver ()
        int compute (
            size_t p_real, const T* const alpha_real, const T* const beta_real,
            size_t p_complex,
            const T* const alpha_complex_real, const T* const alpha_complex_imag,
            const T* const beta_complex_real, const T* const beta_complex_imag,
            size_t N, const double* t, const T* d)
        void solve (const double* b, T* x) const
        void solve (size_t nrhs, const double* b, T* x) const
        T dot_solve (const double* b) except +
        T log_determinant () except +


cdef class Solver:

    cdef BandSolver[double]* solver
    cdef unsigned int N
    cdef np.ndarray t
    cdef np.ndarray diagonal
    cdef unsigned int p_real
    cdef np.ndarray alpha_real
    cdef np.ndarray beta_real
    cdef unsigned int p_complex
    cdef np.ndarray alpha_complex_real
    cdef np.ndarray alpha_complex_imag
    cdef np.ndarray beta_complex_real
    cdef np.ndarray beta_complex_imag

    def __cinit__(self,
                  np.ndarray[DTYPE_t, ndim=1] alpha_real,
                  np.ndarray[DTYPE_t, ndim=1] beta_real,
                  np.ndarray[DTYPE_t, ndim=1] alpha_complex_real,
                  np.ndarray[DTYPE_t, ndim=1] alpha_complex_imag,
                  np.ndarray[DTYPE_t, ndim=1] beta_complex_real,
                  np.ndarray[DTYPE_t, ndim=1] beta_complex_imag,
                  np.ndarray[DTYPE_t, ndim=1] t,
                  d=1e-10):
        if not np.all(np.diff(t) > 0.0):
            raise ValueError("times must be sorted")

        # Check the shapes:
        if alpha_real.shape[0] != beta_real.shape[0]:
            raise ValueError("dimension mismatch")
        if alpha_complex_real.shape[0] != alpha_complex_imag.shape[0]:
            raise ValueError("dimension mismatch")
        if alpha_complex_real.shape[0] != beta_complex_real.shape[0]:
            raise ValueError("dimension mismatch")
        if alpha_complex_real.shape[0] != beta_complex_imag.shape[0]:
            raise ValueError("dimension mismatch")

        # Save the dimensions
        self.N = t.shape[0]
        self.t = t
        self.p_real = alpha_real.shape[0]
        self.alpha_real = alpha_real
        self.beta_real = beta_real
        self.p_complex = alpha_complex_real.shape[0]
        self.alpha_complex_real = alpha_complex_real
        self.alpha_complex_imag = alpha_complex_imag
        self.beta_complex_real = beta_complex_real
        self.beta_complex_imag = beta_complex_imag

        try:
            d = float(d)
        except TypeError:
            d = np.atleast_1d(d)
            if d.shape[0] != self.N:
                raise ValueError("diagonal dimension mismatch")
        else:
            d = d + np.zeros_like(self.t)
        self.diagonal = d

        self.solver = new BandSolver[double]()
        cdef int flag = self.solver.compute(
            self.p_real,
            <double*>(self.alpha_real.data),
            <double*>(self.beta_real.data),
            self.p_complex,
            <double*>(self.alpha_complex_real.data),
            <double*>(self.alpha_complex_imag.data),
            <double*>(self.beta_complex_real.data),
            <double*>(self.beta_complex_imag.data),
            self.N,
            <double*>(self.t.data),
            <double*>(self.diagonal.data)
        )
        if flag:
            if flag == 1:
                raise ValueError("dimension mismatch")
            elif flag == 2:
                raise np.linalg.LinAlgError("invalid parameters")
            raise RuntimeError("compute failed")

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
        cdef double value
        cdef size_t i, j
        cdef np.ndarray[DTYPE_t, ndim=2] A = np.empty((self.N, self.N),
                                                      dtype=DTYPE)
        for i in range(self.N):
            A[i, i] = self.diagonal[i] + get_kernel_value(
                self.alpha_real.shape[0],
                <double*>self.alpha_real.data,
                <double*>self.beta_real.data,
                self.alpha_complex_real.shape[0],
                <double*>self.alpha_complex_real.data,
                <double*>self.alpha_complex_imag.data,
                <double*>self.beta_complex_real.data,
                <double*>self.beta_complex_imag.data,
                0.0
            )
            for j in range(i + 1, self.N):
                delta = fabs(self.t[i] - self.t[j])
                value = get_kernel_value(
                    self.alpha_real.shape[0],
                    <double*>self.alpha_real.data,
                    <double*>self.beta_real.data,
                    self.alpha_complex_real.shape[0],
                    <double*>self.alpha_complex_real.data,
                    <double*>self.alpha_complex_imag.data,
                    <double*>self.beta_complex_real.data,
                    <double*>self.beta_complex_imag.data,
                    delta
                )
                A[i, j] = value
                A[j, i] = value
        return A
