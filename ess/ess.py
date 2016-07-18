# -*- coding: utf-8 -*-

from __future__ import division, print_function

import numpy as np
from ._grp import CythonGRPSolver

__all__ = ["GRPSolver"]


class GRPSolver(object):

    def __init__(self, ln_amplitudes, ln_qfactors, frequencies):
        alpha = []
        beta = []
        for a, q, f in zip(ln_amplitudes, ln_qfactors, frequencies):
            if f is None:
                alpha.append(np.exp(a))
                beta.append(np.exp(-q) + 0j)
                continue
            alpha += [np.exp(a), np.exp(a)]
            beta += [np.exp(-q) + 2j*np.pi*f, np.exp(-q) - 2j*np.pi*f]
        self._alpha = np.array(alpha, dtype=np.float64)
        self._beta = np.array(beta, dtype=np.complex128)
        self._solver = None

    def compute(self, t, yerr):
        del self._solver
        self._solver = CythonGRPSolver(self._alpha, self._beta,
                                       np.atleast_1d(t, dtype=np.float64),
                                       yerr**2)
        self._logdet = None

    @property
    def log_determinant(self):
        if self._logdet is None:
            self._logdet = self._solver.log_determinant
        return self._logdet

    def apply_inverse(self, b, inplace=False):
        return self._solver.apply_inverse(np.atleast_1d(b), inplace=inplace)
