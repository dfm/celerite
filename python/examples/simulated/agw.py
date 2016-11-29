#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import division, print_function

import emcee
import corner
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize

import george
from george import kernels

import genrp

def savefig(name, fig=None):
    if fig is None:
        fig = plt.gcf()
    fig.savefig("agw_" + name + ".pdf", bbox_inches="tight")

np.random.seed(42)

# Simulate the data
t = np.sort(np.random.uniform(0, 5, 100))
yerr = np.random.uniform(0.1, 0.2, len(t))

true_freq = 2.0
#  kernel = 0.5*kernels.ExpKernel(2.0)
kernel = 0.5*kernels.ExpSquaredKernel(2.0)

kernel += 1.0*kernels.ExpSquaredKernel(0.1) * \
    kernels.CosineKernel(period=1.0 / true_freq)
#  kernel += 1.0*kernels.ExpKernel(1.0) * \
#      kernels.CosineKernel(period=1.0 / true_freq)
true_gp = george.GP(kernel)
K = true_gp.get_matrix(t)
K[np.diag_indices_from(K)] += yerr**2
y = np.random.multivariate_normal(np.zeros_like(t), K)

# Set up the fit
fit_gp = genrp.GP()
fit_gp.add_term(0.0, 0.0)
fit_gp.add_term(0.0, 0.0, 0.0)

def log_likelihood(params):
    fit_gp.params = params[1:]
    fit_gp.compute(t, np.sqrt(yerr**2 + np.exp(params[0])))
    return fit_gp.log_likelihood(y)

parameter_bounds = [(-8.0, 8.0) for _ in range(len(fit_gp.params) + 1)]

# Optimize with random restarts
nll = lambda p: -log_likelihood(p)
best = (np.inf, fit_gp.params)
for i in range(10):
    p0 = np.array([np.random.uniform(*a) for a in parameter_bounds])
    r = minimize(nll, p0, method="L-BFGS-B", bounds=parameter_bounds)
    if r.fun < best[0]:
        best = (r.fun, r.x)
fit_gp.params = best[1][1:]

# Use MCMC to sample
def log_prob(p):
    if not all(((b[0] <= v <= b[1]) for v, b in zip(p, parameter_bounds))):
        return -np.inf
    return log_likelihood(p)

nwalkers, ndim = 24, len(fit_gp.params) + 1
sampler = emcee.EnsembleSampler(nwalkers, ndim, log_prob)

# Burn-in
pos, _, _ = sampler.run_mcmc(best[1]+1e-5*np.random.randn(nwalkers, ndim), 500)
sampler.reset()
sampler.run_mcmc(pos, 1000)

# Plot results
fig = corner.corner(sampler.flatchain)
savefig("corner", fig)
plt.close(fig)

samples = sampler.flatchain
tau = np.linspace(0, 3, 5000)
acor = np.empty((len(samples), len(tau)))
f = np.linspace(0.1, 5, 5000)
psd = np.empty((len(samples), len(f)))
for i, s in enumerate(samples):
    fit_gp.params = s[1:]
    acor[i, :] = fit_gp.get_matrix(tau, np.array([tau[0]]))[:, 0]
    psd[i, :] = fit_gp.get_psd(f)

q = np.percentile(acor, [16, 50, 84], axis=0)

fig, ax = plt.subplots(1, 1, figsize=(5, 5))
ax.fill_between(tau, q[0], q[2], color="k", alpha=0.2, lw=0)
ax.plot(tau, q[1], "k", lw=2)
ax.plot(tau, true_gp.get_matrix(tau, np.array([tau[0]])), "g", lw=2)
ax.set_ylabel("autocovariance")
savefig("acor", fig)

q = np.percentile(psd, [16, 50, 84], axis=0)
fig, ax = plt.subplots(1, 1, figsize=(5, 5))
ax.fill_between(f, q[0], q[2], color="k", alpha=0.2, lw=0)
ax.plot(f, q[1], "k", lw=2)
# ax.plot(f, true_gp.get_psd(f), "g", lw=2)
ax.set_xscale("log")
ax.set_xlabel("frequency")
ax.set_xlim(f.min(), f.max())
#  ax.set_xticks([0.1, 0.2, 0.4, 0.8, 1.6, 3.2])
#  ax.get_xaxis().set_major_formatter(plt.ScalarFormatter())

ax.set_yscale("log")
ax.set_ylabel("power spectral density")
#  ax.set_ylim(3e-3, 2e-1)
#  ax.set_yticks([4e-3, 8e-3, 1.6e-2, 3.2e-2, 6.4e-2, 1.28e-1])
#  ax.get_yaxis().set_major_formatter(plt.ScalarFormatter())
savefig("psd", fig)

fig, ax = plt.subplots(1, 1, figsize=(5, 5))
tt = np.linspace(0, 5, 1000)
for i in np.random.randint(len(samples), size=50):
    fit_gp.params = samples[i, 1:]
    K = fit_gp.get_matrix(t)
    K[np.diag_indices_from(K)] += yerr**2 + np.exp(samples[i, 0])
    pred = np.dot(fit_gp.get_matrix(tt, t), np.linalg.solve(K, y))
    ax.plot(tt, pred, "k", alpha=0.3)

ax.errorbar(t, y, yerr=yerr, fmt=".k", capsize=0)
#  ax.set_ylim(-1.1, 1.1)
ax.set_xlabel("t")
ax.set_ylabel("y")
savefig("data", fig)

fig, ax = plt.subplots(1, 1, figsize=(5, 5))
ax.hist(np.exp(samples[:, -1]), 50, histtype="step", color="k")
ax.axvline(true_freq, color="g", lw=2)
savefig("fit_freq", fig)
