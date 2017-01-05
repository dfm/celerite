# -*- coding: utf-8 -*-

from __future__ import division, print_function
import numpy as np
from collections import OrderedDict
from ._genrp import Solver

__all__ = ["GP"]


class GP(object):

    def __init__(self, kernel, log_white_noise=-np.inf, fit_white_noise=False):
        self.kernel = kernel

        self.solver = None
        self._computed = False
        self._t = None
        self._y_var = None
        self._log_white_noise = log_white_noise
        self.fit_white_noise = fit_white_noise

    def __len__(self):
        return self.vector_size

    @property
    def full_size(self):
        return self.kernel.full_size + 1

    @property
    def vector_size(self):
        if self.fit_white_noise:
            return self.kernel.vector_size + 1
        return self.kernel.vector_size

    def get_parameter_dict(self, include_frozen=False):
        return OrderedDict(zip(
            self.get_parameter_names(include_frozen=include_frozen),
            self.get_parameter_vector(include_frozen=include_frozen),
        ))

    def get_parameter_names(self, include_frozen=False):
        names = self.kernel.get_parameter_names(include_frozen=include_frozen)
        names = list(map("kernel:{0}".format, names))
        if include_frozen or self.fit_white_noise:
            names = ["log_white_noise"] + names
        return tuple(names)

    def get_parameter_vector(self, include_frozen=False):
        v = self.kernel.get_parameter_vector(include_frozen=include_frozen)
        if include_frozen or self.fit_white_noise:
            v = np.append(self.log_white_noise, v)
        return v

    def set_parameter_vector(self, vector, include_frozen=False):
        self._computed = False
        if include_frozen or self.fit_white_noise:
            self.log_white_noise = vector[0]
            self.kernel.set_parameter_vector(vector[1:],
                                             include_frozen=include_frozen)
        else:
            self.kernel.set_parameter_vector(vector,
                                             include_frozen=include_frozen)

    def freeze_parameter(self, name):
        if name == "log_white_noise":
            self.fit_white_noise = False
        else:
            self.kernel.freeze_parameter(name[7:])

    def thaw_parameter(self, name):
        if name == "log_white_noise":
            self.fit_white_noise = True
        else:
            self.kernel.thaw_parameter(name[7:])

    def get_parameter(self, name):
        if name == "log_white_noise":
            return self.log_white_noise
        else:
            return self.kernel.get_parameter(name[7:])

    def set_parameter(self, name, value):
        self._computed = False
        if name == "log_white_noise":
            self.log_white_noise = value
        else:
            self.kernel.set_parameter(name[7:], value)

    @property
    def log_white_noise(self):
        return self._log_white_noise

    @log_white_noise.setter
    def log_white_noise(self, value):
        self._computed = False
        self._log_white_noise = value

    @property
    def computed(self):
        return self._computed and not self.kernel.dirty

    def compute(self, t, yerr=1.123e-12, check_sorted=True):
        t = np.atleast_1d(t)
        if check_sorted and np.any(np.diff(t) < 0.0):
            raise ValueError("the input coordinates must be sorted")
        if check_sorted and len(t.shape) > 1:
            raise ValueError("dimension mismatch")
        self._t = t
        self._yerr = np.empty_like(self._t)
        self._yerr[:] = yerr
        self.solver = Solver()
        self.solver.compute(
            self.kernel.alpha_real,
            self.kernel.beta_real,
            self.kernel.alpha_complex_real,
            self.kernel.alpha_complex_imag,
            self.kernel.beta_complex_real,
            self.kernel.beta_complex_imag,
            t, self._yerr**2 + np.exp(self.log_white_noise)
        )
        self._computed = True
        self.kernel.dirty = False

    def _recompute(self):
        if not self.computed:
            if self._t is None:
                raise RuntimeError("you must call 'compute' first")
            self.compute(self._t, self._yerr, check_sorted=False)

    def _process_input(self, y):
        if len(self._t) != len(y):
            raise ValueError("dimension mismatch")
        return np.ascontiguousarray(y, dtype=float)

    def log_likelihood(self, y, _const=np.log(2*np.pi)):
        y = self._process_input(y)
        self._recompute()
        if len(y.shape) > 1:
            raise ValueError("dimension mismatch")
        return -0.5 * (self.solver.dot_solve(y) +
                       self.solver.log_determinant() +
                       len(y) * _const)

    def apply_inverse(self, y):
        self._recompute()
        return self.solver.solve(self._process_input(y))

    def predict(self, y, t, return_cov=True, return_var=False):
        y = self._process_input(y)
        xs = np.ascontiguousarray(t, dtype=float)
        if len(xs.shape) > 1 or len(y.shape) > 1:
            raise ValueError("dimension mismatch")

        # Make sure that the model is computed
        self._recompute()

        # Compute the predictive mean.
        alpha = self.solver.solve(y).flatten()
        Kxs = self.get_matrix(xs, self._t)
        mu = np.dot(Kxs, alpha)
        if not (return_var or return_cov):
            return mu

        # Predictive variance.
        KxsT = np.ascontiguousarray(Kxs.T, dtype=np.float64)
        if return_var:
            var = -np.sum(KxsT*self.apply_inverse(KxsT), axis=0)
            var += self.kernel.get_value(0.0)
            return mu, var

        # Predictive covariance
        cov = self.kernel.get_value(xs[:, None] - xs[None, :])
        cov -= np.dot(Kxs, self.apply_inverse(KxsT))
        return mu, cov

    def get_matrix(self, x1=None, x2=None, include_diagonal=None):
        if x1 is None and x2 is None:
            if self._t is None or not self.computed:
                raise RuntimeError("you must call 'compute' first")
            K = self._t[:, None] - self._t[None, :]
            if include_diagonal is None or include_diagonal:
                K[np.diag_indices_from(K)] += \
                    self._yerr**2 + np.exp(self.log_white_noise)
            return K

        incl = False
        x1 = np.ascontiguousarray(x1, dtype=float)
        if x2 is None:
            x2 = x1
            incl = include_diagonal is not None and include_diagonal
        K = self.kernel.get_value(x1[:, None] - x2[None, :])
        if incl:
            K[np.diag_indices_from(K)] += np.exp(self.log_white_noise)
        return K

    def sample(self, x, tiny=1e-12, size=None):
        K = self.get_matrix(x, include_diagonal=True)
        K[np.diag_indices_from(K)] += tiny
        return np.random.multivariate_normal(np.zeros_like(x), K, size=size)
