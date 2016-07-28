# -*- coding: utf-8 -*-

from __future__ import division, print_function

import emcee
import corner
import numpy as np
import matplotlib.pyplot as pl
from scipy.optimize import minimize

from genrp import GP

np.random.seed(42)


true_gp = GP()
true_gp.add_term(np.log(1.0), np.log(10.0))
true_gp.add_term(np.log(0.5), np.log(40.0), -np.log(3.0))

t = np.linspace(0, 200, 150)
# omega = np.fft.rfftfreq(len(t), t[1] - t[0])
omega = np.linspace(1e-2, 1.0, 500)
t += 0.5 * (t[1] - t[0]) * (np.random.rand(len(t)) - 0.5)
t = np.sort(t)
# t = np.linspace(0, 200, 1000)
y = true_gp.sample(t, tiny=1e-12)
yerr = np.random.uniform(0.25, 0.5, len(t))
y += yerr * np.random.randn(len(t))
# t, y, yerr = t[:100], y[:100], yerr[:100]

# inds = np.sort(np.random.choice(np.arange(len(t)), size=100, replace=False))
# t, y, yerr = t[inds], y[inds], yerr[inds]

fit_gp = GP()
amp = 1.0
rng = t.max() - t.min()
fit_gp.add_term(np.log(amp), np.log(rng))
fit_gp.add_term(np.log(amp), np.log(0.1*rng), -np.log(1.0))

fig, axes = pl.subplots(2, 2, figsize=(10, 10))

# Data
ax = axes[0, 0]
ax.plot(t, y, "k")
ax.set_xlabel("t")
ax.set_ylabel("y")
mx, mn = y.max(), y.min()
rng = 1.1 * max(mx, -mn)
ax.set_ylim(-rng, rng)
ax.set_title("simulated data")

# FFT
ax = axes[0, 1]
# fft = np.fft.rfft(y)
# ax.plot(omega, np.abs(fft) / len(t), "k", label="fft")
ax.plot(omega, true_gp.get_psd(omega), "g", label="truth")
ax.set_xlim(omega[1], omega[-1])
ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xlabel("f")
ax.set_ylabel("psd(f)")
ax.legend(fontsize=15, loc=3)
ax.set_title("Fourier transform")

# Grid of likelihoods
def nll(theta):
    fit_gp.params = theta
    fit_gp.compute(t, yerr)
    ll = fit_gp.log_likelihood(y)
    if not np.isfinite(ll):
        return 1e10
    return -ll

result = minimize(nll, fit_gp.params, method="L-BFGS-B",
                  bounds=[(-5, 5) for _ in range(len(fit_gp.params))])
print(result)
fit_gp.params = result.x

log_f = np.linspace(np.log(omega[1]), np.log(omega.max()), 100)
log_like = np.empty_like(log_f)
params = fit_gp.params
for i, p in enumerate(log_f):
    params[-1] = p
    fit_gp.params = params
    fit_gp.compute(t, yerr)
    log_like[i] = fit_gp.log_likelihood(y)

params[-1] = log_f[np.argmax(log_like)]
fit_gp.params = params

def nll(theta):
    params[:-1] = theta
    fit_gp.params = params
    fit_gp.compute(t, yerr)
    ll = fit_gp.log_likelihood(y)
    if not np.isfinite(ll):
        return 1e10
    return -ll

result = minimize(nll, fit_gp.params[:-1], method="L-BFGS-B",
                  bounds=[(-5, 5) for _ in range(len(fit_gp.params)-1)])
print(result)
params[:-1] = result.x
fit_gp.params = params

ax = axes[1, 0]
ax.plot(np.exp(log_f), log_like, "k")
ax.set_xlim(np.exp(log_f[0]), np.exp(log_f[-1]))
ax.set_ylim(np.median(log_like), np.max(log_like))
ax.set_xscale("log")
ax.set_xlabel("f")
ax.set_ylabel("log likelihood(f)")
ax.set_title("grid search")

# Sampling
def log_prob(theta):
    if np.any(theta > 10.0) or np.any(theta < -10.0):
        return -np.inf
    fit_gp.params = theta
    fit_gp.compute(t, yerr)
    ll = fit_gp.log_likelihood(y)
    if not np.isfinite(ll):
        return -np.inf
    return ll

ndim = len(params)
nwalkers = 32
sampler = emcee.EnsembleSampler(nwalkers, ndim, log_prob, threads=4)
params = params[None, :] + 1e-8 * np.random.randn(nwalkers, ndim)
params, lp, _ = sampler.run_mcmc(params, 250)
params = params[np.argmax(lp)][None, :] + 1e-8*np.random.randn(nwalkers, ndim)
params, _, _ = sampler.run_mcmc(params, 200)
sampler.reset()
sampler.run_mcmc(params, 300)

samples = sampler.flatchain
psds = np.empty((len(samples), len(log_f)))
for i, p in enumerate(samples):
    fit_gp.params = p
    psds[i] = fit_gp.get_psd(np.exp(log_f))

q = np.percentile(psds, [16, 50, 84], axis=0)

ax = axes[1, 1]
ax.fill_between(np.exp(log_f), q[0], q[2], color="k", alpha=0.3)
ax.plot(np.exp(log_f), q[1], "k", label="posterior")
ax.plot(np.exp(log_f), true_gp.get_psd(np.exp(log_f)), "g", label="truth")
ax.set_xlim(np.exp(log_f[0]), np.exp(log_f[-1]))
ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xlabel("f")
ax.set_ylabel("psd(f)")
ax.legend(fontsize=15, loc=3)
ax.set_title("posterior")

pl.tight_layout()
fig.savefig("sample.png")

pl.close(fig)

fig = corner.corner(samples)
fig.savefig("corner.png")
