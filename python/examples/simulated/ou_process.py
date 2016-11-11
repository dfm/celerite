#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import division, print_function

import emcee
import corner
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize

import genrp

def savefig(name, fig=None):
    if fig is None:
        fig = plt.gcf()
    fig.savefig("ou_process_" + name + ".pdf", bbox_inches="tight")

np.random.seed(42)

# Simulate the data
t = np.sort(np.random.uniform(0, 5, 100))
yerr = np.random.uniform(0.05, 0.1, len(t))

true_gp = genrp.GP()
true_gp.add_term(-3.0, 0.0)
K = true_gp.get_matrix(t)
K[np.diag_indices_from(K)] += yerr**2

y = np.random.multivariate_normal(np.zeros_like(t), K)

# Set up the fit
fit_gp = genrp.GP()
fit_gp.add_term(0.0, 0.0)

def log_likelihood(params):
    fit_gp.params = params
    fit_gp.compute(t, yerr)
    return fit_gp.log_likelihood(y)

parameter_bounds = [(-5.0, 5.0), (-5.0, 5.0)]

# Optimize with random restarts
nll = lambda p: -log_likelihood(p)
best = (np.inf, fit_gp.params)
for i in range(10):
    p0 = np.array([np.random.uniform(*a) for a in parameter_bounds])
    r = minimize(nll, p0, method="L-BFGS-B", bounds=parameter_bounds)
    if r.fun < best[0]:
        best = (r.fun, r.x)
fit_gp.params = best[1]

# Use MCMC to sample
def log_prob(p):
    if not all(((b[0] <= v <= b[1]) for v, b in zip(p, parameter_bounds))):
        return -np.inf
    return log_likelihood(p)

nwalkers, ndim = 24, 2
sampler = emcee.EnsembleSampler(nwalkers, ndim, log_prob)

# Burn-in
pos, _, _ = sampler.run_mcmc(best[1]+1e-5*np.random.randn(nwalkers, ndim), 100)
sampler.reset()
sampler.run_mcmc(pos, 1000)

# Plot results
fig = corner.corner(sampler.flatchain, truths=true_gp.params)
savefig("corner", fig)
plt.close(fig)

samples = sampler.flatchain
f = np.linspace(0.1, 5, 5000)
psd = np.empty((len(samples), len(f)))
for i, s in enumerate(samples):
    fit_gp.params = s
    psd[i, :] = fit_gp.get_psd(f)

q = np.percentile(psd, [16, 50, 84], axis=0)

fig, ax = plt.subplots(1, 1, figsize=(5, 5))
ax.fill_between(f, q[0], q[2], color="k", alpha=0.2, lw=0)
ax.plot(f, q[1], "k", lw=2)
ax.plot(f, true_gp.get_psd(f), "g", lw=2)

ax.set_xscale("log")
ax.set_xlabel("frequency")
ax.set_xlim(f.min(), f.max())
ax.set_xticks([0.1, 0.2, 0.4, 0.8, 1.6, 3.2])
ax.get_xaxis().set_major_formatter(plt.ScalarFormatter())

ax.set_yscale("log")
ax.set_ylabel("power spectral density")
ax.set_ylim(3e-3, 2e-1)
ax.set_yticks([4e-3, 8e-3, 1.6e-2, 3.2e-2, 6.4e-2, 1.28e-1])
ax.get_yaxis().set_major_formatter(plt.ScalarFormatter())

savefig("psd", fig)

fig, ax = plt.subplots(1, 1, figsize=(5, 5))
ax.plot(t, y, "-k")
ax.set_ylim(-3.1, 3.1)
ax.set_xlabel("t")
ax.set_ylabel("y")

savefig("data", fig)
