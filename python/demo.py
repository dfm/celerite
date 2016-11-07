from matplotlib import rcParams
rcParams["font.size"] = 16
rcParams["font.family"] = "sans-serif"
rcParams["font.sans-serif"] = ["Computer Modern Sans"]
rcParams["text.usetex"] = True
rcParams["text.latex.preamble"] = r"\usepackage{cmbright}"

import time
import numpy as np
import matplotlib.pyplot as pl

from genrp import GP

gp = GP()
gp.add_term(1.0, 0.1)
gp.add_term(0.1, 2.0, 1.6)

N = 2**np.arange(5, 20)
K = np.maximum((N.max() / N), 5*np.ones_like(N)).astype(int)
times = np.empty((len(N), 2))
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

    print(n, times[i, :])

pl.plot(N, N / N[-1] * 2.0, "k", label="$\mathcal{O}(N)$")
pl.plot(N, times[:, 0], ".-", label="compute")
pl.plot(N, times[:, 1], ".-", label="log likelihood")
pl.xscale("log")
pl.yscale("log")
pl.legend(loc=2, fontsize=15)
pl.xlim(N.min(), N.max())
pl.ylim(1e-5, 3.0)
pl.xlabel("number of data points")
pl.ylabel("computational cost [seconds]")
pl.savefig("demo.png", dpi=300, bbox_inches="tight")
