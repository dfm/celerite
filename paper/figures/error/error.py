#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import division, print_function

import h5py
import numpy as np

import celerite
from celerite import terms, plot_setup

np.random.seed(42)
plot_setup.setup(auto=True)

K = 10
J = np.arange(2, 64, 8)
N = 2**np.arange(6, 16)

alpha_error = np.empty((K, len(J), len(N)))
logdet_error = np.empty((K, len(J), len(N)))
logdet_error[:, :, :] = np.nan

for k in range(K):
    t = np.sort(np.random.uniform(0, N.max() * 0.8, N.max()))
    yerr = np.random.uniform(1.0, 1.5, len(t))

    for ix, j in enumerate(J):
        kernel = terms.RealTerm(np.random.uniform(-1, 1),
                                np.random.uniform(-5, -1))
        kernel += terms.RealTerm(np.random.uniform(-1, 1),
                                 np.random.uniform(-5, -1))
        while (len(kernel.coefficients[0]) + 2*len(kernel.coefficients[2]) <
               2*j):
            kernel += terms.SHOTerm(
                log_S0=np.random.uniform(-1, 1),
                log_omega0=np.random.uniform(-5, 0),
                log_Q=np.random.uniform(0, 1),
            )
        kernel += terms.JitterTerm(np.random.uniform(-1, 1))
        assert (
            len(kernel.coefficients[0]) + 2*len(kernel.coefficients[2]) == 2*j
        )

        gp = celerite.GP(kernel)

        for iy, n in enumerate(N):
            gp.compute(t[:n], yerr[:n])
            alpha_true = np.random.randn(n)
            args = [kernel.jitter] + list(kernel.coefficients)
            args += [t[:n], alpha_true]
            y = gp.solver.dot(*args)[:, 0] + yerr[:n]**2 * alpha_true

            alpha = gp.apply_inverse(y[:n])[:, 0]
            logdet = gp.solver.log_determinant()
            alpha_error[k, ix, iy] = np.max(np.abs(alpha-alpha_true))

            if n <= 2048:
                logdet0 = np.linalg.slogdet(gp.get_matrix())[1]
                logdet_error[k, ix, iy] = np.abs((logdet - logdet0) / logdet)

            print(j, n, alpha_error[k, ix, iy], logdet_error[k, ix, iy])

with h5py.File("error.h5", "w") as f:
    f.create_dataset("J", data=J)
    f.create_dataset("N", data=N)
    f.create_dataset("alpha", data=alpha_error)
    f.create_dataset("logdet", data=logdet_error)
