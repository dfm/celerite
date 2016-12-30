# -*- coding: utf-8 -*-

from __future__ import division, print_function
import math
import numpy as np

from ._genrp import kernel_value, kernel_psd

__all__ = [
    "Kernel", "Sum", "RealTerm", "ComplexTerm", "SHOTerm", "Matern32Term"
]


class Kernel(object):

    def __len__(self):
        return 0

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

    def __init__(self, k1, k2):
        self.k1 = k1
        self.k2 = k2

    def __repr__(self):
        return "{0} + {1}".format(self.k1, self.k2)

    def __len__(self):
        return len(self.k1) + len(self.k2)

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

    def __init__(self, a, log_c):
        self.a = a
        self.log_c = log_c

    def __repr__(self):
        return "RealTerm({0.a}, {0.log_c})".format(self)

    def __len__(self):
        return 2

    @property
    def p_real(self):
        return 1

    @property
    def alpha_real(self):
        return np.array([self.a])

    @property
    def beta_real(self):
        return np.array([math.exp(self.log_c)])


class ComplexTerm(Kernel):

    def __init__(self, *args):
        if len(args) == 3:
            a, log_c, log_d = args
            b = 0.0
        else:
            a, b, log_c, log_d = args
        self.a = a
        self.b = b
        self.log_c = log_c
        self.log_d = log_d

    def __repr__(self):
        return "ComplexTerm({0.a}, {0.b}, {0.log_c}, {0.log_d})".format(self)

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
        return np.array([math.exp(self.log_c)])

    @property
    def beta_complex_imag(self):
        return np.array([math.exp(self.log_d)])


class SHOTerm(Kernel):

    def __init__(self, log_S0, log_Q, log_w0):
        self.log_S0 = log_S0
        self.log_Q = log_Q
        self.log_w0 = log_w0

    def __repr__(self):
        return "SHOTerm({0.log_S0}, {0.log_Q}, {0.log_w0})".format(self)

    @property
    def p_real(self):
        Q = math.exp(self.log_Q)
        return 2 if Q < 0.5 else 0

    @property
    def p_complex(self):
        Q = math.exp(self.log_Q)
        return 0 if Q < 0.5 else 1

    @property
    def alpha_real(self):
        Q = math.exp(self.log_Q)
        if Q >= 0.5:
            return np.empty(0)
        f = 1.0 / math.sqrt(1.0 - 4.0 * Q**2)
        return 0.5*math.exp(self.log_S0+self.log_w0)*Q*np.array([1.0+f, 1.0-f])

    @property
    def beta_real(self):
        Q = math.exp(self.log_Q)
        if Q >= 0.5:
            return np.empty(0)
        f = math.sqrt(1.0 - 4.0 * Q**2)
        return 0.5*math.exp(self.log_w0)/Q*np.array([1.0-f, 1.0+f])

    @property
    def alpha_complex_real(self):
        Q = math.exp(self.log_Q)
        if Q < 0.5:
            return np.empty(0)
        return np.array([math.exp(self.log_S0 + self.log_w0) * Q])

    @property
    def alpha_complex_imag(self):
        Q = math.exp(self.log_Q)
        if Q < 0.5:
            return np.empty(0)
        return np.array([math.exp(self.log_S0+self.log_w0)*Q /
                         math.sqrt(4*Q**2-1.0)])

    @property
    def beta_complex_real(self):
        Q = math.exp(self.log_Q)
        if Q < 0.5:
            return np.empty(0)
        return np.array([0.5*math.exp(self.log_w0)/Q])

    @property
    def beta_complex_imag(self):
        Q = math.exp(self.log_Q)
        if Q < 0.5:
            return np.empty(0)
        return np.array([0.5*math.exp(self.log_w0)/Q*math.sqrt(4*Q**2-1.0)])


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
        w0 = math.sqrt(3.0) * math.exp(-self.log_rho)
        S0 = math.exp(2.0 * self.log_sigma) / w0
        return np.array([w0 * S0])

    @property
    def alpha_complex_imag(self):
        w0 = math.sqrt(3.0) * math.exp(-self.log_rho)
        S0 = math.exp(2.0 * self.log_sigma) / w0
        return np.array([w0 * S0 * w0 / self.eps])

    @property
    def beta_complex_real(self):
        w0 = math.sqrt(3.0) * math.exp(-self.log_rho)
        return np.array([w0])

    @property
    def beta_complex_imag(self):
        return np.array([self.eps])
