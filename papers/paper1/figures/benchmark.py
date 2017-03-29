#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import division, print_function

import numpy as np
import matplotlib.pyplot as plt

from timer import benchmark

from celerite import GP, terms
from celerite import plot_setup
plot_setup.setup()

# Set up the dimensions of the problem
N = 2**np.arange(6, 20)
times = np.empty((len(N), 3))
times[:] = np.nan

# Simulate a "dataset"
np.random.seed(42)
t = np.sort(np.random.rand(np.max(N)))
yerr = np.random.uniform(0.1, 0.2, len(t))
y = np.sin(t)

# Set up the GP model
kernel = terms.RealTerm(1.0, 0.1) + terms.ComplexTerm(0.1, 2.0, 1.6)
gp = GP(kernel)

for i, n in enumerate(N):
    times[i, 0] = benchmark("gp.compute(t[:{0}], yerr[:{0}])".format(n),
                            "from __main__ import gp, t, yerr")

    gp.compute(t[:n], yerr[:n])
    times[i, 1] = benchmark("gp.log_likelihood(y[:{0}])".format(n),
                            "from __main__ import gp, y")

    if n <= 4096:
        times[i, 2] = benchmark("""
C = gp.get_matrix(t[:{0}])
C[np.diag_indices_from(C)] += yerr[:{0}]**2
cho_factor(C)
""".format(n), """
from __main__ import gp, t, yerr
import numpy as np
from scipy.linalg import cho_factor
""")

    print(n, times[i])

fig, ax = plt.subplots(1, 1, figsize=plot_setup.get_figsize(1, 1))
ax.plot(N, N / N[-1] * 2.0, "k", label="$\mathcal{O}(N)$")
ax.plot(N, times[:, 0], ".-", label="compute")
ax.plot(N, times[:, 1], ".--", label="log likelihood")
m = np.isfinite(times[:, 2])
ax.plot(N[m], times[:, 2][m], ".:", label="Cholesky")
ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xlim(N.min(), N.max())
ax.set_ylim(2e-5, 3.0)
ax.set_xlabel("number of data points")
ax.set_ylabel("computational cost [seconds]")
fig.savefig("benchmark.pdf", bbox_inches="tight")
