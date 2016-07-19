from matplotlib import rcParams
rcParams["font.size"] = 16
rcParams["font.family"] = "sans-serif"
rcParams["font.sans-serif"] = ["Computer Modern Sans"]
rcParams["text.usetex"] = True
rcParams["text.latex.preamble"] = r"\usepackage{cmbright}"

import time
import numpy as np
import matplotlib.pyplot as pl

from ess import GRPSolver

solver = GRPSolver(
    np.log([10.0, 5.0]),  # log-amplitudes
    np.log([0.1, 10.0]),  # log-Q-factors
    [None, 50.0],         # frequencies
)

N = 2**np.arange(5, 20)
times = np.empty((len(N), 3))

t = np.random.rand(np.max(N))
yerr = np.random.uniform(0.1, 0.2, len(t))
b = np.random.randn(len(t))

for i, n in enumerate(N):
    strt = time.time()
    solver.compute(t[:n], yerr[:n])
    times[i, 0] = time.time() - strt

    strt = time.time()
    solver.log_determinant
    times[i, 1] = time.time() - strt

    strt = time.time()
    solver.apply_inverse(b[:n])
    times[i, 2] = time.time() - strt

pl.plot(N, times[:, 0], ".-", label="factorization")
pl.plot(N, times[:, 1], ".-", label="determinant")
pl.plot(N, times[:, 2], ".-", label="solve")
pl.xscale("log")
pl.yscale("log")
pl.legend(loc=2, fontsize=14)
pl.xlim(N.min(), N.max())
pl.ylim(7e-6, 15)
pl.xlabel("number of data points")
pl.ylabel("computational cost")
pl.savefig("demo.png", dpi=150, bbox_inches="tight")
