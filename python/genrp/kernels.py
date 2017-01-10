# -*- coding: utf-8 -*-

from __future__ import division, print_function
import math
import numpy as np
from itertools import chain

from .modeling import Model
from ._genrp import get_kernel_value, get_psd_value, check_coefficients

__all__ = [
    "Kernel", "Sum", "RealTerm", "ComplexTerm", "SHOTerm", "Matern32Term"
]


class Kernel(Model):

    @property
    def terms(self):
        return [self]

    def get_value(self, x):
        x = np.asarray(x)
        (alpha_real, beta_real, alpha_complex_real, alpha_complex_imag,
         beta_complex_real, beta_complex_imag) = self.coefficients
        k = get_kernel_value(
            alpha_real, beta_real,
            alpha_complex_real, alpha_complex_imag,
            beta_complex_real, beta_complex_imag,
            x.flatten(),
        )
        return np.asarray(k).reshape(x.shape)

    def get_psd(self, w):
        w = np.asarray(w)
        (alpha_real, beta_real, alpha_complex_real, alpha_complex_imag,
         beta_complex_real, beta_complex_imag) = self.coefficients
        p = get_psd_value(
            alpha_real, beta_real,
            alpha_complex_real, alpha_complex_imag,
            beta_complex_real, beta_complex_imag,
            w.flatten(),
        )
        return p.reshape(w.shape)

    def check_parameters(self):
        return check_coefficients(*(self.coefficients))

    def __add__(self, b):
        return Sum(self, b)

    def __radd__(self, b):
        return Sum(b, self)

    def get_real_coefficients(self):
        return np.empty(0), np.empty(0)

    def get_complex_coefficients(self):
        return np.empty(0), np.empty(0), np.empty(0), np.empty(0)

    def get_all_coefficients(self):
        r = self.get_real_coefficients()
        c = self.get_complex_coefficients()
        if len(c) == 3:
            c = (c[0], np.zeros_like(c[0]), c[1], c[2])
        return list(map(np.atleast_1d, chain(r, c)))

    @property
    def coefficients(self):
        pars = self.get_all_coefficients()
        if len(pars) != 6:
            raise ValueError("there must be 6 coefficient blocks")
        if any(len(p.shape) != 1 for p in pars):
            raise ValueError("coefficient blocks must be 1D")
        if len(pars[0]) != len(pars[1]):
            raise ValueError("coefficient blocks must have the same shape")
        if any(len(pars[2]) != len(p) for p in pars[3:]):
            raise ValueError("coefficient blocks must have the same shape")
        return pars


class Sum(Kernel):

    def __init__(self, k1, k2):
        self.k1 = k1
        self.k2 = k2

    def __repr__(self):
        return "{0} + {1}".format(self.k1, self.k2)

    @property
    def terms(self):
        return self.k1.terms + self.k2.terms

    @property
    def dirty(self):
        return self.k1.dirty or self.k2.dirty

    @dirty.setter
    def dirty(self, value):
        self.k1.dirty = value
        self.k2.dirty = value

    @property
    def full_size(self):
        return self.k1.full_size + self.k2.full_size

    @property
    def vector_size(self):
        return self.k1.vector_size + self.k2.vector_size

    @property
    def unfrozen_mask(self):
        return np.concatenate((
            self.k1.unfrozen_mask,
            self.k2.unfrozen_mask,
        ))

    @property
    def parameter_vector(self):
        return np.concatenate((
            self.k1.parameter_vector,
            self.k2.parameter_vector
        ))

    @parameter_vector.setter
    def parameter_vector(self, v):
        i = self.k1.full_size
        self.k1.parameter_vector = v[:i]
        self.k2.parameter_vector = v[i:]

    @property
    def parameter_names(self):
        return tuple(chain(
            map("k1:{0}".format, self.k1.parameter_names),
            map("k2:{0}".format, self.k2.parameter_names),
        ))

    def _apply_to_parameter(self, func, name, *args):
        if name.startswith("k1:"):
            return getattr(self.k1, func)(name[3:], *args)
        if name.startswith("k2:"):
            return getattr(self.k2, func)(name[3:], *args)
        raise ValueError("unrecognized parameter '{0}'".format(name))

    def freeze_parameter(self, name):
        self._apply_to_parameter("freeze_parameter", name)

    def thaw_parameter(self, name):
        self._apply_to_parameter("thaw_parameter", name)

    def get_parameter(self, name):
        return self._apply_to_parameter("get_parameter", name)

    def set_parameter(self, name, value):
        self.dirty = True
        return self._apply_to_parameter("set_parameter", name, value)

    def get_all_coefficients(self):
        return [np.append(a, b) for a, b in zip(
            self.k1.get_all_coefficients(),
            self.k2.get_all_coefficients(),
        )]


class RealTerm(Kernel):

    parameter_names = ("a", "log_c")

    def __repr__(self):
        return "RealTerm({0.a}, {0.log_c})".format(self)

    def get_real_coefficients(self):
        return self.a, math.exp(self.log_c)


class ComplexTerm(Kernel):

    def __init__(self, *args):
        if len(args) == 3:
            a, log_c, log_d = args
            b = 0.0
            self.fit_b = False
            self.parameter_names = ("a", "log_c", "log_d")
        else:
            a, b, log_c, log_d = args
            self.fit_b = True
            self.parameter_names = ("a", "b", "log_c", "log_d")
        self.a = a
        self.b = b
        self.log_c = log_c
        self.log_d = log_d
        super(ComplexTerm, self).__init__()

    def __repr__(self):
        if not self.fit_b:
            return "ComplexTerm({0.a}, {0.log_c}, {0.log_d})".format(self)
        return "ComplexTerm({0.a}, {0.b}, {0.log_c}, {0.log_d})".format(self)

    def get_complex_coefficients(self):
        return self.a, self.b, math.exp(self.log_c), math.exp(self.log_d)


class SHOTerm(Kernel):

    parameter_names = ("log_S0", "log_Q", "log_omega0")

    def __repr__(self):
        return "SHOTerm({0.log_S0}, {0.log_Q}, {0.log_omega0})".format(self)

    def get_real_coefficients(self, log_half=math.log(0.5)):
        if self.log_Q >= log_half:
            return np.empty(0), np.empty(0)

        S0 = math.exp(self.log_S0)
        Q = math.exp(self.log_Q)
        w0 = math.exp(self.log_omega0)
        f = math.sqrt(1.0 - 4.0 * Q**2)
        return (
            0.5*S0*w0*Q*np.array([1.0+1.0/f, 1.0-1.0/f]),
            0.5*w0/Q*np.array([1.0-f, 1.0+f])
        )

    def get_complex_coefficients(self, log_half=math.log(0.5)):
        if self.log_Q < log_half:
            return np.empty(0), np.empty(0), np.empty(0), np.empty(0)

        S0 = math.exp(self.log_S0)
        Q = math.exp(self.log_Q)
        w0 = math.exp(self.log_omega0)
        f = math.sqrt(4.0 * Q**2-1)
        return (
            S0 * w0 * Q,
            S0 * w0 * Q / f,
            0.5 * w0 / Q,
            0.5 * w0 / Q * f,
        )


class Matern32Term(Kernel):

    parameter_names = ("log_sigma", "log_rho")

    def __init__(self, log_sigma, log_rho, eps=0.01):
        super(Matern32Term, self).__init__(log_sigma, log_rho)
        self.eps = eps

    def __repr__(self):
        return "Matern32Term({0.log_sigma}, {0.log_rho}, eps={0.eps})" \
            .format(self)

    def get_complex_coefficients(self):
        w0 = math.sqrt(3.0) * math.exp(-self.log_rho)
        S0 = math.exp(2.0 * self.log_sigma) / w0
        return (w0*S0, w0*w0*S0/self.eps, w0, self.eps)
