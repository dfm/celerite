# -*- coding: utf-8 -*-

from __future__ import division, print_function
import numpy as np
from ._genrp import Solver

__all__ = ["GP"]


class GP(object):

    def __init__(self, kernel, log_white_noise=-np.inf, fit_white_noise=False):
        self.kernel = kernel

        self.solver = None
        self._computed = False
        self._t = None
        self._y_var = None
        self.log_white_noise = log_white_noise
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

    def compute(self, t, yerr=1.123e-12):
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

    def log_likelihood(self, y):
        if not self.computed:
            if self._t is None:
                raise RuntimeError("you must call 'compute' first")
            self.compute(self._t, self._y_err)
        if len(self._t) != len(y):
            raise ValueError("dimension mismatch")
        y = np.ascontiguousarray(y, dtype=float)
        return -0.5 * (self.solver.dot_solve(y) +
                       self.solver.log_determinant() +
                       len(y) * np.log(2*np.pi))

    @property
    def computed(self):
        return self._computed and not self.kernel.dirty

    def get_matrix(self, x1, x2=None):
        x1 = np.ascontiguousarray(x1, dtype=float)
        if x2 is None:
            x2 = x1
        return self.kernel.get_value(x1[:, None] - x2[None, :])

    def get_kernel_value(self, tau):
        return self.kernel.get_value(tau)

    def get_kernel_psd(self, omega):
        return self.kernel.get_psd(omega)

    def sample(self, x, tiny=1e-12):
        K = self.get_matrix(x)
        K[np.diag_indices_from(K)] += tiny
        return np.random.multivariate_normal(np.zeros_like(x), K)
