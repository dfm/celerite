#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import division, print_function
import numpy as np
import matplotlib.pyplot as plt

def kernel(terms, tau):
    k = np.zeros_like(tau)
    t = np.abs(tau)
    for a, b, c, d in terms:
        k += np.exp(-c*t)*(a*np.cos(d*t) + b*np.sin(d*t))
    return k

def psd(terms, omega):
    w2 = omega ** 2
    p = np.zeros_like(w2)
    for a, b, c, d in terms:
        p += np.sqrt(2.0 / np.pi) * (
            (a*c+b*d)*(c**2+d**2)+(a*c-b*d)*w2
        ) / (
            w2**2 + 2.0*(c**2-d**2)*w2+(c**2+d**2)**2
        )
    return p

tau = np.linspace(0, 5, 5000)
t = np.linspace(0, 3, 500)
omega = np.linspace(0, 10, 5000)
models = [
    [(1.0, 0.0, 1.0, 0.0), ],
    [(1.0, 0.0, 1.0, 2.5), ],
    [(1.0, 0.2, 1.0, 5.0), ],
    [(0.5, 0.2, 1.0, 5.0), (0.6, 0.0, 1.0, 0.0), ],
]

fig, axes = plt.subplots(1, 3, figsize=(12, 4))

for terms in models:
    axes[0].plot(tau, kernel(terms, tau))
    axes[1].plot(omega, psd(terms, omega))

    np.random.seed(42)
    K = kernel(terms, t[:, None] - t[None, :])
    y = np.random.multivariate_normal(np.zeros_like(t), K)
    axes[2].plot(t, y)

axes[1].set_yscale("log")

fig.savefig("kernel_psd.png")
