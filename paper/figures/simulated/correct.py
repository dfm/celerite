#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import division, print_function

import emcee
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize

import celerite
from celerite import terms
from celerite import plot_setup

np.random.seed(42)
plot_setup.setup(auto=True)

# Simulate some data
kernel = terms.SHOTerm(log_S0=0.0, log_omega0=2.0, log_Q=2.0,
                       bounds=[(-10, 10), (-10, 10), (-10, 10)])
gp = celerite.GP(kernel)
true_params = np.array(gp.get_parameter_vector())
omega = 2*np.pi*np.exp(np.linspace(-np.log(10.0), -np.log(0.1), 5000))
true_psd = gp.kernel.get_psd(omega)
N = 200
t = np.sort(np.random.uniform(0, 10, N))
yerr = 2.5
y = gp.sample(t, diag=yerr**2)

# Find the maximum likelihood model
gp.compute(t, yerr)

def nll(params, gp, y):
    gp.set_parameter_vector(params)
    if not np.isfinite(gp.log_prior()):
        return 1e10
    ll = gp.log_likelihood(y)
    return -ll if np.isfinite(ll) else 1e10

# Run the MCMC
p0 = true_params + 1e-4*np.random.randn(len(true_params))
soln = minimize(nll, p0, method="L-BFGS-B", args=(gp, y))
gp.set_parameter_vector(soln.x)
ml_psd = gp.kernel.get_psd(omega)

def log_probability(params):
    gp.set_parameter_vector(params)
    lp = gp.log_prior()
    if not np.isfinite(lp):
        return -np.inf
    ll = gp.log_likelihood(y)
    return ll + lp if np.isfinite(ll) else -np.inf

ndim = len(soln.x)
nwalkers = 32
coords = soln.x + 1e-4 * np.random.randn(nwalkers, ndim)

sampler = emcee.EnsembleSampler(nwalkers, ndim, log_probability)
coords, _, _ = sampler.run_mcmc(coords, 500)
sampler.reset()
coords, _, _ = sampler.run_mcmc(coords, 2000)

# Compute the posterior PSD inference
samples = sampler.flatchain[::15, :]
post_psd = np.empty((len(samples), len(omega)))
for i, s in enumerate(samples):
    gp.set_parameter_vector(s)
    post_psd[i] = gp.kernel.get_psd(omega)
q = np.percentile(post_psd, [16, 50, 84], axis=0)

# Plot the results
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=plot_setup.get_figsize(1, 2))

ax1.errorbar(t, y, yerr=yerr, fmt=".k", capsize=0)
ax1.set_ylim(-26, 26)
ax1.set_xlim(0, 10)
ax1.set_xlabel("time [s]")
ax1.set_ylabel("relative flux [ppm]")
ax1.annotate("simulated data", xy=(0, 0), xycoords="axes fraction",
             xytext=(5, 5), textcoords="offset points",
             ha="left", va="bottom")

factor = 1.0 / (2*np.pi)
f = omega * factor
ax2.plot(f, q[1] * factor)
ax2.fill_between(f, q[0] * factor, q[2] * factor, alpha=0.3)
ax2.plot(f, true_psd * factor, "--k")
ax2.set_xlim(f[0], f[-1])
ax2.set_yscale("log")
ax2.set_xscale("log")
ax2.set_xlabel("frequency [Hz]")
ax2.set_ylabel("power [ppm$^2$ Hz$^{-1}$]")
ax2.annotate("inferred psd", xy=(0, 0), xycoords="axes fraction",
             xytext=(5, 5), textcoords="offset points",
             ha="left", va="bottom")

fig.savefig("correct.pdf", bbox_inches="tight")
