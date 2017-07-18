#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import division, print_function

import numpy as np
import matplotlib.pyplot as plt
from celerite.plot_setup import setup, get_figsize

np.random.seed(42)
setup(auto=True)

def sho_psd(Q, x):
    x2 = x*x
    return 1.0 / ((x2 - 1)**2 + x2 / Q**2)

def sho_acf(Q, tau):
    t = np.abs(tau)
    if np.allclose(Q, 0.5):
        return np.exp(-t) * (1.0 + t)
    b = 1.0 / np.sqrt(4*Q**2 - 1)
    c = 0.5 / Q
    d = 0.5 * np.sqrt(4*Q**2 - 1) / Q
    return np.exp(-c * t) * (np.cos(d*t)+b*np.sin(d*t))

def lorentz_psd(Q, x):
    return Q**2 * (1.0 / ((x - 1)**2 * (2*Q)**2 + 1) +
                   1.0 / ((x + 1)**2 * (2*Q)**2 + 1))

def lorentz_acf(Q, tau):
    t = np.abs(tau)
    return np.exp(-0.5*t/Q) * np.cos(t)

fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=get_figsize(1, 3))
x = 10**np.linspace(-1.1, 1.1, 5000)
tau = np.linspace(0, 20, 1000)

for i, (Q_name, Q) in enumerate(
        [("1/2", 0.5), ("1/\\sqrt{2}", 1./np.sqrt(2)), ("2", 2.0),
         ("10", 10.0)]):
    l, = ax1.plot(x, sho_psd(Q, x), label="$Q = {0}$".format(Q_name), lw=1.5)
    c = l.get_color()
    ax2.plot(tau, sho_acf(Q, tau), label="$Q = {0}$".format(Q_name), lw=1.5,
             color=c)

    K = sho_acf(Q, tau[:, None] - tau[None, :])
    y = np.random.multivariate_normal(np.zeros(len(tau)), K, size=3)
    ax3.axhline(-5*i, color="k", lw=0.75)
    ax3.plot(tau, -5*i + (y - np.mean(y, axis=1)[:, None]).T, color=c,
             lw=1)

ax1.plot(x, lorentz_psd(10.0, x), "--k")
ax2.plot(tau, lorentz_acf(10.0, tau), "--k")

ax1.set_xscale("log")
ax1.set_yscale("log")
ax1.set_xlim(x.min(), x.max())
ax1.set_ylim(2e-4, 200.0)
ax1.legend(loc=3, fontsize=11)
ax1.set_xlabel("$\omega/\omega_0$")
ax1.set_ylabel("$S(\omega) / S(0)$")

ax2.set_xlim(tau.min(), tau.max())
ax2.set_ylim(-1.1, 1.1)
ax2.set_xlabel("$\omega_0\,\\tau$")
ax2.set_ylabel("$k(\\tau) / k(0)$")

ax3.set_xlim(0, 20)
ax3.set_yticklabels([])
ax3.set_xlabel("$\omega_0\,t$")

fig.savefig("sho.pdf", bbox_inches="tight")
