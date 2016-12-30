# -*- coding: utf-8 -*-

from __future__ import division, print_function
import numpy as np
from ._genrp import Solver

__all__ = ["GP"]


class GP(object):

    def __init__(self, kernel):
        self.kernel = kernel

        self.solver = None
        self._computed = False
        self._t = None
        self._y_var = None

    def compute(self, t, yerr=1.123e-12):
        self._t = np.asarray(t, dtype=float)
        self._y_var = yerr * yerr + np.zeros_like(self._t)
        self.solver = Solver(
            self.kernel.alpha_real,
            self.kernel.beta_real,
            self.kernel.alpha_complex_real,
            self.kernel.alpha_complex_imag,
            self.kernel.beta_complex_real,
            self.kernel.beta_complex_imag,
            self._t,
            self._y_var,
        )
        self._computed = True

    def log_likelihood(self, y):
        return -0.5 * (self.solver.dot_solve(y) +
                       self.solver.log_determinant +
                       len(y) * np.log(2*np.pi))

    @property
    def computed(self):
        return self._computed

    #  property params:
    #      def __get__(self):
    #          cdef np.ndarray[DTYPE_t] p = np.empty(self.gp.size(), dtype=DTYPE)
    #          self.gp.get_params(<double*>p.data)
    #          return p

    #      def __set__(self, params):
    #          if len(params) != self.gp.size():
    #              raise ValueError("dimension mismatch")

    #          cdef np.ndarray[DTYPE_t, ndim=1] p = \
    #              np.atleast_1d(params).astype(DTYPE)

    #          # Set the parameters in the scalar solver
    #          self.gp.set_params(<double*>p.data)

    #          self._data_size = -1

    #  def __getitem__(self, i):
    #      return self.params[i]

    #  def __setitem__(self, i, value):
    #      params = self.params
    #      params[i] = value
    #      self.params = params

    #  def add_term(self, log_a, log_q, log_f=None):
    #      self.kernel.add_term(log_a, log_q, log_f)
    #      self._terms.append((log_a, log_q, log_f))
    #      del self.gp
    #      self.gp = new GaussianProcess[BandSolver[double], double](self.kernel)
    #      self._data_size = -1

    #  def get_matrix(self, np.ndarray[DTYPE_t, ndim=1] x1, x2=None):
    #      if x2 is None:
    #          x2 = x1
    #      cdef np.ndarray[DTYPE_t, ndim=2] K = x1[:, None] - x2[None, :]
    #      cdef int i, j
    #      for i in range(x1.shape[0]):
    #          for j in range(x2.shape[0]):
    #              K[i, j] = self.gp.kernel_value(K[i, j])
    #      return K

    #  def get_psd(self, np.ndarray[DTYPE_t, ndim=1] w):
    #      cdef np.ndarray[DTYPE_t, ndim=1] psd = np.empty_like(w, dtype=DTYPE)
    #      cdef int i
    #      for i in range(w.shape[0]):
    #          psd[i] = self.gp.kernel_psd(w[i])
    #      return psd

    #  def compute(self, np.ndarray[DTYPE_t, ndim=1] x, np.ndarray[DTYPE_t, ndim=1] yerr):
    #      if x.shape[0] != yerr.shape[0]:
    #          raise ValueError("dimension mismatch")
    #      cdef int i
    #      for i in range(x.shape[0] - 1):
    #          if x[i+1] <= x[i]:
    #              raise ValueError("the time series must be ordered")
    #      self._data_size = x.shape[0]
    #      self.gp.compute(self._data_size, <double*>x.data, <double*>yerr.data)

    #  def log_likelihood(self, np.ndarray[DTYPE_t, ndim=1] y):
    #      if not self.computed:
    #          raise RuntimeError("you must call 'compute' first")
    #      if y.shape[0] != self._data_size:
    #          raise ValueError("dimension mismatch")
    #      return self.gp.log_likelihood(<double*>y.data)

    #  def sample(self, np.ndarray[DTYPE_t, ndim=1] x, double tiny=1e-12):
    #      cdef np.ndarray[DTYPE_t, ndim=2] K = self.get_matrix(x)
    #      K[np.diag_indices_from(K)] += tiny
    #      return np.random.multivariate_normal(np.zeros_like(x), K)
