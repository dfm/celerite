#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import division, print_function

import numpy as np
import matplotlib.pyplot as plt

from timer import benchmark

from celerite.solver import Solver, CARMASolver
from celerite import plot_setup
plot_setup.setup()

# Simulate a "dataset"
np.random.seed(42)
t = np.sort(np.random.rand(2**11))
yerr = np.random.uniform(0.1, 0.2, len(t))
y = np.sin(t)

ps = 2 ** np.arange(8)
js = np.empty_like(ps)
times = np.empty((len(ps), 2))
times[:] = np.nan

for i, p in enumerate(ps):
    arparams = np.random.randn(p)
    maparams = np.random.randn(p - 1)

    times[i, 0] = benchmark("""
carma_solver = CARMASolver(0.0, arparams, maparams)
carma_solver.log_likelihood(t, y, yerr)
""", "from __main__ import arparams, maparams, t, y, yerr, CARMASolver")

    carma_solver = CARMASolver(0.0, arparams, maparams)
    params = list(carma_solver.get_celerite_coeffs())
    params += [t, yerr**2]
    celerite_solver = Solver()

    # Hack to deal with extra terms
    js[i] = len(params[0]) + 2*len(params[2])
    diff = js[i] - p
    if diff > 0:
        m = diff // 2
        if m > 0:
            for j in range(2, 6):
                params[j] = params[j][:-m]
            js[i] = len(params[0]) + 2*len(params[2])
            diff = js[i] - p
        if diff > 0:
            for j in range(2):
                params[j] = params[j][:-diff]
            js[i] = len(params[0]) + 2*len(params[2])

    times[i, 1] = benchmark("""
celerite_solver.compute(*params)
celerite_solver.dot_solve(y)
""", "from __main__ import params, celerite_solver, y")

    print(p, js[i], len(params[0]), len(params[2]), times[i])

a = np.log(ps)
b = np.log(times[:, 0])
print(np.polyfit(a, b, 1))

m = np.isfinite(times[:, 1])
a = np.log(js[m])
b = np.log(times[m, 1])
print(np.polyfit(a, b, 1))

fig, ax = plt.subplots(1, 1, figsize=plot_setup.get_figsize(1, 1))
ax.plot(ps, 1e-4 * ps**2, "k")
ax.plot(ps, times[:, 0], ".--")
ax.plot(js[m], times[m, 1], ".-")
ax.set_xlim(ps.min(), max(ps.max(), js.max()))
ax.set_ylim(4e-4, 4)
ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xlabel("number of terms")
ax.set_ylabel("computational cost [seconds]")
fig.savefig("carma_comp.pdf", bbox_inches="tight")
