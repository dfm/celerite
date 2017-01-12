from cycler import cycler
from matplotlib import rcParams
rcParams["font.size"] = 16
rcParams["font.family"] = "sans-serif"
rcParams["font.sans-serif"] = ["Computer Modern Sans"]
rcParams["text.usetex"] = True
rcParams["text.latex.preamble"] = r"\usepackage{cmbright}"
rcParams["axes.prop_cycle"] = cycler("color", (
    "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b",
    "#e377c2", "#7f7f7f", "#bcbd22", "#17becf",
))  # d3.js color cycle

import time
import numpy as np
import matplotlib.pyplot as pl
from scipy.linalg import cho_factor

from celerite import terms, GP

kernel = terms.RealTerm(1.0, 0.1) + terms.ComplexTerm(0.1, 2.0, 1.6)
gp = GP(kernel)

N = 2**np.arange(6, 20)
K = np.maximum((N.max() / N), 5*np.ones_like(N)).astype(int)
K_chol = np.maximum((4096 / N), 5*np.ones_like(N)).astype(int)
times = np.empty((len(N), 3))
times[:] = np.nan

t = np.sort(np.random.rand(np.max(N)))
yerr = np.random.uniform(0.1, 0.2, len(t))
y = np.sin(t)

for i, n in enumerate(N):
    strt = time.time()
    for k in range(K[i]):
        gp.compute(t[:n], yerr[:n])
    times[i, 0] = (time.time() - strt) / K[i]

    strt = time.time()
    for k in range(K[i]):
        gp.log_likelihood(y[:n])
    times[i, 1] = (time.time() - strt) / K[i]

    if n <= 4096:
        strt = time.time()
        for k in range(K_chol[i]):
            C = gp.get_matrix(t[:n])
            C[np.diag_indices_from(C)] += yerr[:n]**2
            cho_factor(C)
        times[i, 2] = (time.time() - strt) / K_chol[i]

    print(n, times[i, :])

pl.plot(N, N / N[-1] * 2.0, "k", label="$\mathcal{O}(N)$")
pl.plot(N, times[:, 0], ".-", label="compute")
pl.plot(N, times[:, 1], ".-", label="log likelihood")
m = np.isfinite(times[:, 2])
pl.plot(N[m], times[:, 2][m], ".-", label="Cholesky")
pl.xscale("log")
pl.yscale("log")
pl.legend(loc=4, fontsize=15)
pl.xlim(N.min(), N.max())
pl.ylim(1e-5, 3.0)
pl.xlabel("number of data points")
pl.ylabel("computational cost [seconds]")
pl.savefig("demo.png", dpi=300, bbox_inches="tight")
