# -*- coding: utf-8 -*-

from __future__ import division, print_function
import math
import numpy as np
from itertools import chain
from collections import OrderedDict

from ._genrp import get_kernel_value, get_psd_value, check_coefficients

__all__ = [
    "Kernel", "Sum", "RealTerm", "ComplexTerm", "SHOTerm", "Matern32Term"
]


class Kernel(object):

    parameter_names = tuple()

    def __init__(self):
        self.unfrozen_mask = np.ones(self.full_size, dtype=bool)
        self.dirty = True

    def __len__(self):
        return self.vector_size

    def __getitem__(self, name):
        return self.get_parameter(name)

    def __setitem__(self, name, value):
        return self.set_parameter(name, value)

    @property
    def full_size(self):
        return len(self.parameter_names)

    @property
    def vector_size(self):
        return self.unfrozen_mask.sum()

    @property
    def terms(self):
        return [self]

    def get_value(self, x):
        x = np.asarray(x)
        k = get_kernel_value(
            self.alpha_real,
            self.beta_real,
            self.alpha_complex_real,
            self.alpha_complex_imag,
            self.beta_complex_real,
            self.beta_complex_imag,
            x.flatten(),
        )
        return np.asarray(k).reshape(x.shape)

    def get_psd(self, w):
        w = np.asarray(w)
        p = get_psd_value(
            self.alpha_real,
            self.beta_real,
            self.alpha_complex_real,
            self.alpha_complex_imag,
            self.beta_complex_real,
            self.beta_complex_imag,
            w.flatten(),
        )
        return p.reshape(w.shape)

    def check_parameters(self):
        return check_coefficients(
            self.alpha_real,
            self.beta_real,
            self.alpha_complex_real,
            self.alpha_complex_imag,
            self.beta_complex_real,
            self.beta_complex_imag,
        )

    def __add__(self, b):
        return Sum(self, b)

    def __radd__(self, b):
        return Sum(b, self)

    def get_parameter_dict(self, include_frozen=False):
        return OrderedDict(zip(
            self.get_parameter_names(include_frozen=include_frozen),
            self.get_parameter_vector(include_frozen=include_frozen),
        ))

    def get_parameter_names(self, include_frozen=False):
        if include_frozen:
            return self.parameter_names
        return tuple(p
                     for p, f in zip(self.parameter_names, self.unfrozen_mask)
                     if f)

    def get_parameter_vector(self, include_frozen=False):
        if include_frozen:
            return self.parameter_vector
        return self.parameter_vector[self.unfrozen_mask]

    def set_parameter_vector(self, vector, include_frozen=False):
        v = self.parameter_vector
        if include_frozen:
            v[:] = vector
        else:
            v[self.unfrozen_mask] = vector
        self.parameter_vector = v
        self.dirty = True

    def freeze_parameter(self, name):
        i = self.get_parameter_names(include_frozen=True).index(name)
        self.unfrozen_mask[i] = False

    def thaw_parameter(self, name):
        i = self.get_parameter_names(include_frozen=True).index(name)
        self.unfrozen_mask[i] = True

    def get_parameter(self, name):
        i = self.get_parameter_names(include_frozen=True).index(name)
        return self.get_parameter_vector(include_frozen=True)[i]

    def set_parameter(self, name, value):
        i = self.get_parameter_names(include_frozen=True).index(name)
        v = self.get_parameter_vector(include_frozen=True)
        v[i] = value
        self.set_parameter_vector(v, include_frozen=True)

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

    def get_parameter_names(self, include_frozen=False):
        return tuple(chain(
            map("k1:{0}".format,
                self.k1.get_parameter_names(include_frozen=include_frozen)),
            map("k2:{0}".format,
                self.k2.get_parameter_names(include_frozen=include_frozen)),
        ))

    def get_parameter_vector(self, include_frozen=False):
        return np.append(
            self.k1.get_parameter_vector(include_frozen=include_frozen),
            self.k2.get_parameter_vector(include_frozen=include_frozen),
        )

    def set_parameter_vector(self, vector, include_frozen=False):
        i = self.k1.full_size if include_frozen else self.k1.vector_size
        self.k1.set_parameter_vector(vector[:i], include_frozen=include_frozen)
        self.k2.set_parameter_vector(vector[i:], include_frozen=include_frozen)

    def freeze_parameter(self, name):
        if name.startswith("k1:"):
            self.k1.freeze_parameter(name[3:])
        elif name.startswith("k2:"):
            self.k2.freeze_parameter(name[3:])
        else:
            raise ValueError("unrecognized parameter '{0}'".format(name))

    def thaw_parameter(self, name):
        if name.startswith("k1:"):
            self.k1.thaw_parameter(name[3:])
        elif name.startswith("k2:"):
            self.k2.thaw_parameter(name[3:])
        else:
            raise ValueError("unrecognized parameter '{0}'".format(name))

    def get_parameter(self, name):
        if name.startswith("k1:"):
            return self.k1.get_parameter(name[3:])
        elif name.startswith("k2:"):
            return self.k2.get_parameter(name[3:])
        else:
            raise ValueError("unrecognized parameter '{0}'".format(name))

    def set_parameter(self, name, value):
        if name.startswith("k1:"):
            self.k1.set_parameter(name[3:], value)
        elif name.startswith("k2:"):
            self.k2.set_parameter(name[3:], value)
        else:
            raise ValueError("unrecognized parameter '{0}'".format(name))

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

    parameter_names = ("a", "log_c")

    def __init__(self, a, log_c):
        super(RealTerm, self).__init__()
        self.a = a
        self.log_c = log_c

    @property
    def parameter_vector(self):
        return np.array([self.a, self.log_c])

    @parameter_vector.setter
    def parameter_vector(self, v):
        self.a = v[0]
        self.log_c = v[1]

    def __repr__(self):
        return "RealTerm({0.a}, {0.log_c})".format(self)

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

    @property
    def parameter_vector(self):
        if self.fit_b:
            return np.array([self.a, self.b, self.log_c, self.log_d])
        return np.array([self.a, self.log_c, self.log_d])

    @parameter_vector.setter
    def parameter_vector(self, v):
        self.a = v[0]
        if self.fit_b:
            self.b = v[1]
            self.log_c = v[2]
            self.log_d = v[3]
        else:
            self.log_c = v[1]
            self.log_d = v[2]

    def __repr__(self):
        if not self.fit_b:
            return "ComplexTerm({0.a}, {0.log_c}, {0.log_d})".format(self)
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

    parameter_names = ("log_S0", "log_Q", "log_w0")

    def __init__(self, log_S0, log_Q, log_w0):
        super(SHOTerm, self).__init__()
        self.log_S0 = log_S0
        self.log_Q = log_Q
        self.log_w0 = log_w0

    @property
    def parameter_vector(self):
        return np.array([self.log_S0, self.log_Q, self.log_w0])

    @parameter_vector.setter
    def parameter_vector(self, v):
        self.log_S0 = v[0]
        self.log_Q = v[1]
        self.log_w0 = v[2]

    def __repr__(self):
        return "SHOTerm({0.log_S0}, {0.log_Q}, {0.log_w0})".format(self)

    @property
    def p_real(self, log_half=math.log(0.5)):
        return 2 if self.log_Q < log_half else 0

    @property
    def p_complex(self, log_half=math.log(0.5)):
        return 0 if self.log_Q < log_half else 1

    @property
    def alpha_real(self, log_half=math.log(0.5)):
        if self.log_Q >= log_half:
            return np.empty(0)
        Q = math.exp(self.log_Q)
        f = 1.0 / math.sqrt(1.0 - 4.0 * Q**2)
        return 0.5*math.exp(self.log_S0+self.log_w0)*Q*np.array([1.0+f, 1.0-f])

    @property
    def beta_real(self, log_half=math.log(0.5)):
        if self.log_Q >= log_half:
            return np.empty(0)
        Q = math.exp(self.log_Q)
        f = math.sqrt(1.0 - 4.0 * Q**2)
        return 0.5*math.exp(self.log_w0)/Q*np.array([1.0-f, 1.0+f])

    @property
    def alpha_complex_real(self, log_half=math.log(0.5)):
        if self.log_Q < log_half:
            return np.empty(0)
        Q = math.exp(self.log_Q)
        return np.array([math.exp(self.log_S0 + self.log_w0) * Q])

    @property
    def alpha_complex_imag(self, log_half=math.log(0.5)):
        if self.log_Q < log_half:
            return np.empty(0)
        Q = math.exp(self.log_Q)
        return np.array([math.exp(self.log_S0+self.log_w0)*Q /
                         math.sqrt(4*Q**2-1.0)])

    @property
    def beta_complex_real(self, log_half=math.log(0.5)):
        if self.log_Q < log_half:
            return np.empty(0)
        Q = math.exp(self.log_Q)
        return np.array([0.5*math.exp(self.log_w0)/Q])

    @property
    def beta_complex_imag(self, log_half=math.log(0.5)):
        if self.log_Q < log_half:
            return np.empty(0)
        Q = math.exp(self.log_Q)
        return np.array([0.5*math.exp(self.log_w0)/Q*math.sqrt(4*Q**2-1.0)])


class Matern32Term(Kernel):

    parameter_names = ("log_sigma", "log_rho")

    def __init__(self, log_sigma, log_rho, eps=1e-5):
        super(Matern32Term, self).__init__()
        self.log_sigma = log_sigma
        self.log_rho = log_rho
        self.eps = eps

    @property
    def parameter_vector(self):
        return np.array([self.log_sigma, self.log_rho])

    @parameter_vector.setter
    def parameter_vector(self, v):
        self.log_sigma = v[0]
        self.log_rho = v[1]

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
