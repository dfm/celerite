#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import division, print_function

import emcee
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from scipy.linalg import cho_factor, cho_solve

import celerite
from celerite import terms
from celerite.modeling import Model

from celerite import plot_setup

plot_setup.setup(auto=True)

class TrueModel(Model):
    parameter_names = ("log_amp", "log_ell", "log_period")

    def get_K(self, x):
        tau = x[:, None] - x[None, :]
        return (
            np.exp(self.log_amp - 0.5 * tau**2 * np.exp(-2.0*self.log_ell)) *
            np.cos(2*np.pi*tau*np.exp(-self.log_period))
        )

    def __call__(self, params, x, y, yerr):
        self.set_parameter_vector(params)
        lp = self.log_prior()
        if not np.isfinite(lp):
            return -np.inf

        K = self.get_K(x)
        K[np.diag_indices_from(K)] += yerr**2
        try:
            factor = cho_factor(K, overwrite_a=True)
        except (np.linalg.LinAlgError, ValueError):
            return -np.inf
        ld = 2.0 * np.sum(np.log(np.diag(factor[0])))
        return -0.5*(np.dot(y, cho_solve(factor, y))+ld) + lp

true_model = TrueModel(log_amp=0.0, log_ell=np.log(5.0), log_period=0.0,
                       bounds=[(-10, 10), (-10, 10), (-10, 10)])

# Simulate a dataset from the true model
np.random.seed(42)
N = 100
t = np.sort(np.random.uniform(0, 20, N))
yerr = 0.5
K = true_model.get_K(t)
K[np.diag_indices_from(K)] += yerr**2
y = np.random.multivariate_normal(np.zeros(N), K)

# Set up the celerite model that we will use to fit - product of two SHOs
log_Q = 1.0
kernel = terms.SHOTerm(log_S0=np.log(np.var(y))-2*log_Q, log_Q=log_Q,
                       log_omega0=np.log(2*np.pi))
kernel *= terms.SHOTerm(log_S0=0.0, log_omega0=0.0, log_Q=-0.5*np.log(2))
kernel.freeze_parameter("k2:log_S0")
kernel.freeze_parameter("k2:log_Q")

gp = celerite.GP(kernel)
gp.compute(t, yerr)

# Fit for the maximum likelihood
def nll(params, gp, y):
    gp.set_parameter_vector(params)
    if not np.isfinite(gp.log_prior()):
        return 1e10
    ll = gp.log_likelihood(y)
    return -ll if np.isfinite(ll) else 1e10

p0 = gp.get_parameter_vector()
soln = minimize(nll, p0, method="L-BFGS-B", args=(gp, y))
gp.set_parameter_vector(soln.x)

kernel.freeze_parameter("k1:log_S0")
p0 = gp.get_parameter_vector()
soln = minimize(nll, p0, method="L-BFGS-B", args=(gp, y))
gp.set_parameter_vector(soln.x)

# Do the MCMC with the correct model
ndim = 3
nwalkers = 32
coords = true_model.get_parameter_vector() + 1e-4 * np.random.randn(nwalkers,
                                                                    ndim)
true_sampler = emcee.EnsembleSampler(nwalkers, ndim, true_model,
                                     args=(t, y, yerr))
coords, _, _ = true_sampler.run_mcmc(coords, 500)
true_sampler.reset()
coords, _, _ = true_sampler.run_mcmc(coords, 2000)

# Do the MCMC with the celerite model
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

# Plot the results
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=plot_setup.get_figsize(1, 2))

ax1.errorbar(t, y, yerr=yerr, fmt=".k", capsize=0)
ax1.set_ylim(-3.25, 3.25)
ax1.set_xlim(0, 20)
ax1.set_xlabel("time [day]")
ax1.set_ylabel("relative flux [ppm]")
ax1.annotate("simulated data", xy=(0, 0), xycoords="axes fraction",
             xytext=(5, 5), textcoords="offset points",
             ha="left", va="bottom")

n, b, p = ax2.hist(np.exp(-sampler.flatchain[:, -2])*(2*np.pi), 20,
                   color="k", histtype="step", lw=2, normed=True)
ax2.hist(np.exp(true_sampler.flatchain[:, -1]), b,
         color=plot_setup.COLORS["MODEL_1"],
         lw=2, histtype="step", normed=True, ls="dashed")
ax2.yaxis.set_major_locator(plt.NullLocator())
ax2.set_xlim(b.min(), b.max())
ax2.axvline(1.0, color=plot_setup.COLORS["MODEL_2"], lw=2)
ax2.set_xlabel("period [day]")
fig.savefig("wrong-qpo.pdf", bbox_inches="tight")
