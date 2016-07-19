# -*- coding: utf-8 -*-

from __future__ import division, print_function

import numpy as np
from ._genrp import CythonGRPSolver

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

    def compute(self, t, yerr=1e-10):
        del self._solver
        t = np.atleast_1d(t).astype(np.float64)
        try:
            yerr = float(yerr)
        except TypeError:
            yerr = np.atleast_1d(yerr).astype(np.float64)
        else:
            yerr = yerr + np.zeros_like(t)
        if len(t.shape) != 1 or t.shape != yerr.shape:
            raise ValueError("dimension mismatch")
        self._inds = np.argsort(t)
        self._solver = CythonGRPSolver(self._alpha, self._beta, t[self._inds],
                                       yerr[self._inds]**2)
        self._logdet = None

    @property
    def solver(self):
        if self._solver is None:
            raise RuntimeError("the solver must be computed first using the "
                               "'compute' method")
        return self._solver

    @property
    def log_determinant(self):
        if self._logdet is None:
            self._logdet = self.solver.log_determinant
        return self._logdet

    def apply_inverse(self, b):
        b = np.atleast_1d(b).astype(np.float64)
        r = np.empty_like(b)
        r[self._inds] = self.solver.apply_inverse(b[self._inds], in_place=True)
        return r
