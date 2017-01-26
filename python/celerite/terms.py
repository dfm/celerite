# -*- coding: utf-8 -*-

from __future__ import division, print_function
import re
import numpy as np
from itertools import chain

from .modeling import Model
from ._celerite import get_kernel_value, get_psd_value, check_coefficients

__all__ = [
    "Term", "TermSum", "RealTerm", "ComplexTerm", "SHOTerm", "Matern32Term"
]


class Term(Model):

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
        return TermSum(self, b)

    def __radd__(self, b):
        return TermSum(b, self)

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


class TermSum(Term):

    def __init__(self, *terms):
        self._terms = []
        for term in terms:
            self._terms += term.terms

    def __repr__(self):
        return " + ".join(map("{0}".format, self.terms))

    @property
    def terms(self):
        return self._terms

    @property
    def dirty(self):
        return any(t.dirty for t in self._terms)

    @dirty.setter
    def dirty(self, value):
        for t in self._terms:
            t.dirty = value

    @property
    def full_size(self):
        return sum(t.full_size for t in self._terms)

    @property
    def vector_size(self):
        return sum(t.vector_size for t in self._terms)

    @property
    def unfrozen_mask(self):
        return np.concatenate([
            t.unfrozen_mask for t in self._terms
        ])

    @property
    def parameter_vector(self):
        return np.concatenate([
            t.parameter_vector for t in self._terms
        ])

    @parameter_vector.setter
    def parameter_vector(self, v):
        i = 0
        for t in self._terms:
            l = t.full_size
            t.parameter_vector = v[i:i+l]
            i += l

    @property
    def parameter_names(self):
        return tuple(chain(*(
            map("terms[{0}]:{{0}}".format(i).format, t.parameter_names)
            for i, t in enumerate(self._terms)
        )))

    @property
    def parameter_bounds(self):
        return list(chain(*(
            t.parameter_bounds for t in self._terms
        )))

    def _apply_to_parameter(self, func, name, *args):
        groups = re.findall(r"^terms\[([0-9]+)\]:(.*)", name)
        if not len(groups):
            raise ValueError("unrecognized parameter '{0}'".format(name))
        index, subname = groups[0]
        return getattr(self._terms[int(index)], func)(subname, *args)

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
        return [np.concatenate(a) for a in zip(*(
            t.get_all_coefficients() for t in self._terms
        ))]


class RealTerm(Term):

    parameter_names = ("log_a", "log_c")

    def __repr__(self):
        return "RealTerm({0.log_a}, {0.log_c})".format(self)

    def get_real_coefficients(self):
        return np.exp(self.log_a), np.exp(self.log_c)


class ComplexTerm(Term):

    def __init__(self, *args):
        if len(args) == 3:
            log_a, log_c, log_d = args
            log_b = -np.inf
            self.fit_b = False
            self.parameter_names = ("log_a", "log_c", "log_d")
        else:
            log_a, log_b, log_c, log_d = args
            self.fit_b = True
            self.parameter_names = ("log_a", "log_b", "log_c", "log_d")
        self.log_a = log_a
        self.log_b = log_b
        self.log_c = log_c
        self.log_d = log_d
        super(ComplexTerm, self).__init__()

    def __repr__(self):
        if not self.fit_b:
            return "ComplexTerm({0.log_a}, {0.log_c}, {0.log_d})".format(self)
        return ("ComplexTerm({0.log_a}, {0.log_b}, {0.log_c}, {0.log_d})"
                .format(self))

    def get_complex_coefficients(self):
        if not self.fit_b:
            return (
                np.exp(self.log_a), 0.0, np.exp(self.log_c), np.exp(self.log_d)
            )
        return (
            np.exp(self.log_a), np.exp(self.log_b),
            np.exp(self.log_c), np.exp(self.log_d)
        )

    def log_prior(self):
        # Constraint required for term to be positive definite. Can be relaxed
        # with multiple terms but must be treated carefully.
        if self.log_a + self.log_c < self.log_b + self.log_d:
            return -np.inf
        return super(ComplexTerm, self).log_prior()


class SHOTerm(Term):

    parameter_names = ("log_S0", "log_Q", "log_omega0")

    def __repr__(self):
        return "SHOTerm({0.log_S0}, {0.log_Q}, {0.log_omega0})".format(self)

    def get_real_coefficients(self):
        Q = np.exp(self.log_Q)
        if Q >= 0.5:
            return np.empty(0), np.empty(0)

        S0 = np.exp(self.log_S0)
        w0 = np.exp(self.log_omega0)
        f = np.sqrt(1.0 - 4.0 * Q**2)
        return (
            0.5*S0*w0*Q*np.array([1.0+1.0/f, 1.0-1.0/f]),
            0.5*w0/Q*np.array([1.0-f, 1.0+f])
        )

    def get_complex_coefficients(self):
        Q = np.exp(self.log_Q)
        if Q < 0.5:
            return np.empty(0), np.empty(0), np.empty(0), np.empty(0)

        S0 = np.exp(self.log_S0)
        w0 = np.exp(self.log_omega0)
        f = np.sqrt(4.0 * Q**2-1)
        return (
            S0 * w0 * Q,
            S0 * w0 * Q / f,
            0.5 * w0 / Q,
            0.5 * w0 / Q * f,
        )


class Matern32Term(Term):

    parameter_names = ("log_sigma", "log_rho")

    def __init__(self, log_sigma, log_rho, eps=0.01):
        super(Matern32Term, self).__init__(log_sigma, log_rho)
        self.eps = eps

    def __repr__(self):
        return "Matern32Term({0.log_sigma}, {0.log_rho}, eps={0.eps})" \
            .format(self)

    def get_complex_coefficients(self):
        w0 = np.sqrt(3.0) * np.exp(-self.log_rho)
        S0 = np.exp(2.0 * self.log_sigma) / w0
        return (w0*S0, w0*w0*S0/self.eps, w0, self.eps)
