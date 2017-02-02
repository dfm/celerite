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

import celerite
from celerite import terms

def savefig(name, fig=None):
    if fig is None:
        fig = plt.gcf()
    fig.savefig("agw_" + name + ".pdf", bbox_inches="tight")

np.random.seed(42)

# Simulate the data
t = np.sort(np.random.uniform(0, 5, 100))
yerr = np.random.uniform(0.1, 0.2, len(t))

true_freq = 2.0
true_q = 10.0
kernel = 0.5*kernels.ExpSquaredKernel(2.0)
kernel += 1.0*kernels.ExpSquaredKernel(true_q) * \
    kernels.CosineKernel(period=1.0 / true_freq)
true_gp = george.GP(kernel)
K = true_gp.get_matrix(t)
K[np.diag_indices_from(K)] += yerr**2
y = np.random.multivariate_normal(np.zeros_like(t), K)

# Set up the fit
# bounds = [(-8.0, 8.0), (-8.0, 8.0), (-8.0, 8.0)]
# kernel = SHOTerm(0.0, -0.5 * np.log(2), 0.0, bounds=bounds)
# kernel.freeze_parameter("log_Q")
# kernel += SHOTerm(0.0, 0.0, 0.0, bounds=bounds)
# kernel += SHOTerm(0.0, 0.0, 0.0, bounds=bounds)
kernel = terms.RealTerm(0.0, 0.0, bounds=[(-8, 8), (-8, 8)])
kernel += terms.ComplexTerm(0.0, 0.0, 0.0, 0.0,
                            bounds=[(-8, 8), (-8, 8), (-8, 8), (-8, 8)])
fit_gp = celerite.GP(kernel)
fit_gp.compute(t, yerr)

def nll(params):
    fit_gp.set_parameter_vector(params)
    if not np.isfinite(fit_gp.log_prior()):
        return 1e10
    return -fit_gp.log_likelihood(y)

npars = len(fit_gp.get_parameter_vector())
parameter_bounds = fit_gp.get_parameter_bounds()

# Optimize with random restarts
best = (np.inf, fit_gp.get_parameter_vector())
for i in range(10):
    v = 1e10
    while v > 1e9:
        p0 = np.array([np.random.uniform(*a) for a in parameter_bounds])
        v = nll(p0)
    r = minimize(nll, p0, method="L-BFGS-B", bounds=parameter_bounds)
    if r.fun < best[0]:
        best = (r.fun, r.x)
fit_gp.set_parameter_vector(best[1])

# Use MCMC to sample
def log_prob(p):
    fit_gp.set_parameter_vector(p)
    lp = fit_gp.log_prior()
    if not np.isfinite(lp):
        return -np.inf
    return fit_gp.log_likelihood(y) + lp

nwalkers, ndim = 24, npars
sampler = emcee.EnsembleSampler(nwalkers, ndim, log_prob)

# Burn-in
p0 = best[1] + 1e-5*np.random.randn(nwalkers, ndim)
pos, lp, _ = sampler.run_mcmc(p0, 500)
p0 = pos[np.argmax(lp)] + 1e-5*np.random.randn(nwalkers, ndim)
pos, lp, _ = sampler.run_mcmc(p0, 500)
sampler.reset()
sampler.run_mcmc(pos, 1000)

# Plot results
fig = corner.corner(sampler.flatchain)
savefig("corner", fig)
plt.close(fig)

samples = sampler.flatchain
tau = np.linspace(0, 3, 5000)
acor = np.empty((len(samples), len(tau)))
f = np.exp(np.linspace(0.5*np.log(2*np.pi/5), np.log(5), 5000))
omega = 2*np.pi*f
psd = np.empty((len(samples), len(f)))
for i, s in enumerate(samples):
    fit_gp.set_parameter_vector(s)
    acor[i, :] = fit_gp.kernel.get_value(tau)
    psd[i, :] = fit_gp.kernel.get_psd(2*np.pi*f)

q = np.percentile(acor, [16, 50, 84], axis=0)

fig, ax = plt.subplots(1, 1, figsize=(5, 5))
ax.fill_between(tau, q[0], q[2], color="k", alpha=0.2, lw=0)
ax.plot(tau, q[1], "k", lw=2)
# ax.plot(tau, true_gp.get_matrix(tau, np.array([tau[0]])), "g", lw=2)
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
ylim = np.array(ax.get_ylim())

true_psd = 0.5*np.sqrt(2) * np.exp(-0.5*2*omega**2)
true_psd += 0.5*0.5*np.sqrt(true_q)*np.exp(-0.5*true_q*(omega-2*np.pi*true_freq)**2)
true_psd += 0.5*0.5*np.sqrt(true_q)*np.exp(-0.5*true_q*(omega+2*np.pi*true_freq)**2)
ax.plot(f, true_psd, "--g")

ax.set_ylabel("power spectral density")
ax.set_ylim(ylim)
#  ax.set_ylim(3e-3, 2e-1)
#  ax.set_yticks([4e-3, 8e-3, 1.6e-2, 3.2e-2, 6.4e-2, 1.28e-1])
#  ax.get_yaxis().set_major_formatter(plt.ScalarFormatter())
savefig("psd", fig)

fig, ax = plt.subplots(1, 1, figsize=(5, 5))
tt = np.linspace(0, 5, 1000)
for i in np.random.randint(len(samples), size=50):
    fit_gp.params = samples[i, 1:]
    pred = fit_gp.predict(y, tt, return_cov=False)
    ax.plot(tt, pred, "k", alpha=0.3)

ax.errorbar(t, y, yerr=yerr, fmt=".k", capsize=0)
ax.set_xlabel("t")
ax.set_ylabel("y")
savefig("data", fig)

fig, ax = plt.subplots(1, 1, figsize=(5, 5))
ax.hist(np.exp(samples[:, -1])/(2*np.pi), 50, histtype="step", color="k")
ax.axvline(true_freq, color="g", lw=2)
savefig("fit_freq", fig)
