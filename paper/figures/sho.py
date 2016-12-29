#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import division, print_function

import numpy as np
import matplotlib.pyplot as plt
from plot_setup import setup, SQUARE_FIGSIZE

setup()

def sho_psd(Q, x):
    x2 = x*x
    return 1.0 / ((x2 - 1)**2 + x2 / Q**2)

def lorentz_psd(Q, x):
    return Q**2 * (1.0 / ((x - 1)**2 * (2*Q)**2 + 1) +
                   1.0 / ((x + 1)**2 * (2*Q)**2 + 1))

fig, ax = plt.subplots(1, 1, figsize=SQUARE_FIGSIZE)
x = 10**np.linspace(-1.1, 1.1, 5000)

for Q_name, Q in [("1/2", 0.5), ("1/\\sqrt{2}", 1./np.sqrt(2)),
                  ("2", 2.0), ("10", 10.0)]:
    ax.plot(x, sho_psd(Q, x), label="$Q = {0}$".format(Q_name), lw=1.5)

ax.plot(x, lorentz_psd(10.0, x), "--k")

ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xlim(x.min(), x.max())
ax.set_ylim(2e-4, 200.0)
ax.legend(loc=3, fontsize=11)
ax.set_xlabel("$\omega/\omega_0$")
ax.set_ylabel("$S(\omega) / S(0)$")
fig.savefig("sho.pdf", bbox_inches="tight")
