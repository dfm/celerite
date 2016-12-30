# -*- coding: utf-8 -*-

from __future__ import division, print_function
import numpy as np

from ._genrp import kernel_value, kernel_psd

__all__ = [
    "Kernel", "Sum", "RealTerm", "ComplexTerm", "SHOTerm", "Matern32Term"
]


class Kernel(object):

    def get_value(self, x):
        x = np.asarray(x)
        k = kernel_value(
            self.alpha_real.astype(float),
            self.beta_real.astype(float),
            self.alpha_complex_real.astype(float),
            self.alpha_complex_imag.astype(float),
            self.beta_complex_real.astype(float),
            self.beta_complex_imag.astype(float),
            x.flatten().astype(float),
        )
        return k.reshape(x.shape)

    def get_psd(self, w):
        w = np.asarray(w)
        p = kernel_psd(
            self.alpha_real.astype(float),
            self.beta_real.astype(float),
            self.alpha_complex_real.astype(float),
            self.alpha_complex_imag.astype(float),
            self.beta_complex_real.astype(float),
            self.beta_complex_imag.astype(float),
            w.flatten().astype(float),
        )
        return p.reshape(w.shape)

    def __add__(self, b):
        return Sum(self, b)

    def __radd__(self, b):
        return Sum(b, self)

    @property
    def p_real(self):
        return 0

    @property
    def p_complex(self):
        return 0

    @property
    def alpha_real(self):
        return np.empty(0)

    @property
    def beta_real(self):
        return np.empty(0)

    @property
    def alpha_complex_real(self):
        return np.empty(0)

    @property
    def alpha_complex_imag(self):
        return np.empty(0)

    @property
    def beta_complex_real(self):
        return np.empty(0)

    @property
    def beta_complex_imag(self):
        return np.empty(0)


class Sum(Kernel):
    is_kernel = False

    def __init__(self, k1, k2):
        self.k1 = k1
        self.k2 = k2

    def __repr__(self):
        return "{0} + {1}".format(self.k1, self.k2)

    @property
    def p_real(self):
        return self.k1.p_real + self.k2.p_real

    @property
    def p_complex(self):
        return self.k1.p_complex + self.k2.p_complex

    @property
    def alpha_real(self):
        return np.append(self.k1.alpha_real, self.k2.alpha_real)

    @property
    def beta_real(self):
        return np.append(self.k1.beta_real, self.k2.beta_real)

    @property
    def alpha_complex_real(self):
        return np.append(self.k1.alpha_complex_real,
                         self.k2.alpha_complex_real)

    @property
    def alpha_complex_imag(self):
        return np.append(self.k1.alpha_complex_imag,
                         self.k2.alpha_complex_imag)

    @property
    def beta_complex_real(self):
        return np.append(self.k1.beta_complex_real,
                         self.k2.beta_complex_real)

    @property
    def beta_complex_imag(self):
        return np.append(self.k1.beta_complex_imag,
                         self.k2.beta_complex_imag)


class RealTerm(Kernel):

    def __init__(self, a, c):
        self.a = a
        self.c = c

    def __repr__(self):
        return "RealTerm({0.a}, {0.c})".format(self)

    @property
    def p_real(self):
        return 1

    @property
    def alpha_real(self):
        return np.array([self.a])

    @property
    def beta_real(self):
        return np.array([self.c])


class ComplexTerm(Kernel):

    def __init__(self, *args):
        if len(args) == 3:
            a, c, d = args
            b = 0.0
        else:
            a, b, c, d = args
        self.a = a
        self.b = b
        self.c = c
        self.d = d

    def __repr__(self):
        return "ComplexTerm({0.a}, {0.b}, {0.c}, {0.d})".format(self)

    @property
    def p_complex(self):
        return 1

    @property
    def alpha_complex_real(self):
        return np.array([self.a])

    @property
    def alpha_complex_imag(self):
        return np.array([self.b])

    @property
    def beta_complex_real(self):
        return np.array([self.c])

    @property
    def beta_complex_imag(self):
        return np.array([self.d])


class SHOTerm(Kernel):

    def __init__(self, S0, Q, w0):
        self.S0 = S0
        self.Q = Q
        self.w0 = w0

    def __repr__(self):
        return "SHOTerm({0.S0}, {0.Q}, {0.w0})".format(self)

    @property
    def p_real(self):
        return 2 if self.Q < 0.5 else 0

    @property
    def p_complex(self):
        return 0 if self.Q < 0.5 else 1

    @property
    def alpha_real(self):
        if self.Q >= 0.5:
            return np.empty(0)
        f = 1.0 / np.sqrt(1.0 - 4.0 * self.Q**2)
        return 0.5*self.S0*self.w0*self.Q*np.array([1.0+f, 1.0-f])

    @property
    def beta_real(self):
        if self.Q >= 0.5:
            return np.empty(0)
        f = np.sqrt(1.0 - 4.0 * self.Q**2)
        return 0.5*self.w0/self.Q*np.array([1.0-f, 1.0+f])

    @property
    def alpha_complex_real(self):
        if self.Q < 0.5:
            return np.empty(0)
        return np.array([self.S0 * self.w0 * self.Q])

    @property
    def alpha_complex_imag(self):
        if self.Q < 0.5:
            return np.empty(0)
        return np.array([self.S0*self.w0*self.Q/np.sqrt(4*self.Q**2-1.0)])

    @property
    def beta_complex_real(self):
        if self.Q < 0.5:
            return np.empty(0)
        return np.array([0.5*self.w0/self.Q])

    @property
    def beta_complex_imag(self):
        if self.Q < 0.5:
            return np.empty(0)
        return np.array([0.5*self.w0/self.Q*np.sqrt(4*self.Q**2-1.0)])


class Matern32Term(Kernel):

    def __init__(self, log_sigma, log_rho, eps=1e-5):
        self.log_sigma = log_sigma
        self.log_rho = log_rho
        self.eps = eps

    def __repr__(self):
        return "Matern32Term({0.log_sigma}, {0.log_rho}, eps={0.eps})" \
            .format(self)

    @property
    def p_complex(self):
        return 1

    @property
    def alpha_complex_real(self):
        w0 = np.sqrt(3.0) * np.exp(-self.log_rho)
        S0 = np.exp(2.0 * self.log_sigma) / w0
        return np.array([w0 * S0])

    @property
    def alpha_complex_imag(self):
        w0 = np.sqrt(3.0) * np.exp(-self.log_rho)
        S0 = np.exp(2.0 * self.log_sigma) / w0
        return np.array([w0 * S0 * w0 / self.eps])

    @property
    def beta_complex_real(self):
        w0 = np.sqrt(3.0) * np.exp(-self.log_rho)
        return np.array([w0])

    @property
    def beta_complex_imag(self):
        return np.array([self.eps])
