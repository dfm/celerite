#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import division, print_function

import os
import sys
import corner
import emcee3
import pickle
import fitsio
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize

import celerite
from celerite.plot_setup import setup, get_figsize
from transit_model import RotationTerm, TransitModel

setup(auto=True)
np.random.seed(42)

data = fitsio.read("data/kplr001430163-2013011073258_llc.fits")
texp = 1625.3467838829 / 60. / 60. / 24.

N = 1000
m = data["SAP_QUALITY"] == 0
m &= np.isfinite(data["TIME"])
m &= np.isfinite(data["PDCSAP_FLUX"])
t = np.ascontiguousarray(data["TIME"][m], dtype=np.float64)[:N]
y = np.ascontiguousarray(data["PDCSAP_FLUX"][m], dtype=np.float64)[:N]
yerr = np.ascontiguousarray(data["PDCSAP_FLUX_ERR"][m], dtype=np.float64)[:N]
t -= 0.5 * (t.min() + t.max())

# Build the true model
true_model = TransitModel(
    texp,
    0.0,
    np.log(8.0),    # period
    np.log(0.015),  # Rp / Rs
    np.log(0.5),    # duration
    0.0,            # t_0
    0.5,            # impact
    0.5,            # q_1
    0.5,            # q_2
)
true_params = np.array(true_model.get_parameter_vector())

# Inject the transit into the data
true_transit = 1e-3*true_model.get_value(t) + 1.0
y *= true_transit

# Normalize the data
med = np.median(y)
y = (y / med - 1.0) * 1e3
yerr *= 1e3 / med

# Set up the GP model
mean = TransitModel(
    texp,
    0.0,
    np.log(8.0),
    np.log(0.015),
    np.log(0.5),
    0.0,
    0.5,
    0.5,
    0.5,
    bounds=[
        (-0.5, 0.5),
        np.log([7.9, 8.1]),
        (np.log(0.005), np.log(0.1)),
        (np.log(0.4), np.log(0.6)),
        (-0.1, 0.1),
        (0, 1.0), (1e-5, 1-1e-5), (1e-5, 1-1e-5)
    ]
)

kernel = RotationTerm(
    np.log(np.var(y)), np.log(0.5*t.max()), np.log(4.5), 0.0,
    bounds=[
        np.log(np.var(y) * np.array([0.01, 100])),
        np.log([np.max(np.diff(t)), (t.max() - t.min())]),
        np.log([3*np.median(np.diff(t)), 0.5*(t.max() - t.min())]),
        [-8.0, np.log(5.0)],
    ]
)

gp = celerite.GP(kernel, mean=mean, fit_mean=True,
                 log_white_noise=2*np.log(0.5*yerr.min()),
                 fit_white_noise=True)
gp.compute(t, yerr)
print("Initial log-likelihood: {0}".format(gp.log_likelihood(y)))

# Define the model
def neg_log_like(params, y, gp):
    gp.set_parameter_vector(params)
    return -gp.log_likelihood(y)

# Optimize with random restarts
p0 = gp.get_parameter_vector()
bounds = gp.get_parameter_bounds()
r = minimize(neg_log_like, p0, method="L-BFGS-B", bounds=bounds, args=(y, gp))
gp.set_parameter_vector(r.x)
ml_params = np.array(r.x)
print("Maximum log-likelihood: {0}".format(gp.log_likelihood(y)))

# Compute the maximum likelihood predictions
x = np.linspace(t.min(), t.max(), 5000)
trend = gp.predict(y, t, return_cov=False)
trend -= gp.mean.get_value(t) - gp.mean.mean_flux
mu, var = gp.predict(y, x, return_var=True)
std = np.sqrt(var)
mean_mu = gp.mean.get_value(x)
mu -= mean_mu
wn = np.exp(gp.log_white_noise.value)
ml_yerr = np.sqrt(yerr**2 + wn)

# Plot the maximum likelihood predictions
fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=get_figsize(1, 2))
ax1.errorbar(t - t.min(), y, yerr=ml_yerr, fmt=".k", capsize=0, zorder=-1)
ax1.plot(x - t.min(), mu, zorder=100)
ax1.set_ylim(-0.72, 0.72)
ax1.yaxis.set_major_locator(plt.MaxNLocator(5))
ax1.set_ylabel("raw [ppt]")
ax1.yaxis.set_label_coords(-0.1, 0.5)

ax2.errorbar(t - t.min(), y-trend, yerr=ml_yerr, fmt=".k", capsize=0,
             zorder=-1)
ax2.plot(x - t.min(), mean_mu - gp.mean.mean_flux, zorder=100)
ax2.set_xlim(0, t.max()-t.min())
ax2.set_ylim(-0.41, 0.1)
ax2.yaxis.set_major_locator(plt.MaxNLocator(5))
ax2.set_ylabel("de-trended [ppt]")
ax2.set_xlabel("time [days]")
ax2.yaxis.set_label_coords(-0.1, 0.5)
fig.savefig("transit-ml.pdf")

# Save the current state of the GP and data
with open("transit.pkl", "wb") as f:
    pickle.dump((gp, y, true_model.get_parameter_dict()), f, -1)

if os.path.exists("transit.h5"):
    result = input("MCMC save file exists. Overwrite? (type 'yes'): ")
    if result.lower() != "yes":
        sys.exit(0)

# Do the MCMC
def log_prob(params):
    gp.set_parameter_vector(params)
    lp = gp.log_prior()
    if not np.isfinite(lp):
        return -np.inf
    return gp.log_likelihood(y) + lp

# Initialize
print("Running MCMC sampling...")
ndim = len(ml_params)
nwalkers = 32
pos = ml_params + 1e-5 * np.random.randn(nwalkers, ndim)
lp = np.array(list(map(log_prob, pos)))
m = ~np.isfinite(lp)
while np.any(m):
    pos[m] = ml_params + 1e-5 * np.random.randn(m.sum(), ndim)
    lp[m] = np.array(list(map(log_prob, pos[m])))
    m = ~np.isfinite(lp)

# Sample
sampler = emcee3.Sampler(backend=emcee3.backends.HDFBackend("transit.h5"))
with emcee3.pools.InterruptiblePool() as pool:
    ensemble = emcee3.Ensemble(emcee3.SimpleModel(log_prob), pos, pool=pool)
    sampler.run(ensemble, 15000, progress=True)
