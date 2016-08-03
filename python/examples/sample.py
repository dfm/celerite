# -*- coding: utf-8 -*-

from __future__ import division, print_function

import sys
import emcee
import corner
import numpy as np
import matplotlib.pyplot as pl
from scipy.optimize import minimize

from astropy.stats import LombScargle

from genrp import GP

np.random.seed(123)


true_gp = GP()
true_gp.add_term(np.log(10.0), np.log(0.5))
true_gp.add_term(np.log(10.0), np.log(0.1), np.log(1.0)-1.0)
true_gp.add_term(np.log(10.0), np.log(0.1), np.log(1.0))
# true_gp.add_term(np.log(2.5), np.log(50.0), -np.log(5.0)+1.0)

if "uniform" in sys.argv:
    t = np.linspace(0, 200, 1000)
    freqs = np.fft.rfftfreq(len(t), d=t[1]-t[0])[1:-1]
else:
    t = np.random.uniform(0, 200, 1000)
    t = np.sort(t)
    freqs = np.exp(np.linspace(-np.log(t.max() - t.min()),
                               -np.log(2*np.mean(np.diff(t))), 1000))
y = true_gp.sample(t, tiny=1e-12)
yerr = np.random.uniform(0.005, 0.01, len(t))
y += yerr * np.random.randn(len(t))

fit_gp = GP()
amp = 1.0
rng = t.max() - t.min()
fit_gp.add_term(np.log(amp), np.log(rng))
fit_gp.add_term(np.log(amp), np.log(0.1*rng), -np.log(1.0))

fig, axes = pl.subplots(2, 2, figsize=(10, 10))

# Data
ax = axes[0, 0]
ax.plot(t, y, ".k")
ax.set_xlabel("t")
ax.set_ylabel("y")
mx, mn = y.max(), y.min()
rng = 1.1 * max(mx, -mn)
ax.set_ylim(-rng, rng)
# ax.set_xlim(t.min(), t.max())
ax.set_title("simulated data")

# k = true_gp.get_matrix(t)[0]
# print(k)
# ax.plot(t - t[0], k)
# ax.set_xlim(0, 50)

# Lomb-scargle.
ax = axes[0, 1]

if "uniform" in sys.argv:
    fft = np.abs(2*np.pi*np.fft.rfft(y) / len(t))**2
    ax.plot(freqs, fft[1:-1], "b", label="fft")

power = (2*np.pi)**2*LombScargle(t, y).power(freqs, normalization="psd")/len(t)
ax.plot(freqs, power, "k", label="lomb-scargle")

ax.plot(freqs, true_gp.get_psd(freqs), "--g", label="truth")
ax.set_xlim(freqs[1], freqs[-1])
ax.set_xlabel("f")
ax.set_ylabel("psd(f)")
ax.set_xscale("log")
# ax.legend(fontsize=15, loc=1)
ax.set_title("periodogram")

if "ml" in sys.argv or "mcmc" in sys.argv:
    # Maximum-likelihood
    def nll(theta):
        fit_gp.params = theta
        fit_gp.compute(t, yerr)
        ll = fit_gp.log_likelihood(y)
        if not np.isfinite(ll):
            return 1e10
        return -ll

    def min_nll(p):
        result = minimize(nll, p, method="L-BFGS-B",
                          bounds=[(-5, 5) for _ in range(len(p))])
        print(result.status, result.fun, result.x)
        return result

    results = list(map(
        min_nll, np.random.uniform(-5, 5, (10, len(fit_gp.params)))
    ))
    i = np.argmin([r.fun for r in results])
    result = results[i]

    print(result)
    fit_gp.params = result.x

    ax = axes[1, 0]
    ax.plot(freqs, fit_gp.get_psd(freqs), "k", label="maximum likelihood")
    ax.plot(freqs, true_gp.get_psd(freqs), "--g", label="truth")
    ax.set_xlim(freqs[0], freqs[-1])
    ax.set_xlabel("f")
    ax.set_ylabel("psd(f)")
    ax.set_xscale("log")
    # ax.legend(fontsize=15, loc=1)
    ax.set_title("maximum likelihood")

# Sampling
if "mcmc" in sys.argv:
    def log_prob(theta):
        if np.any(theta > 10.0) or np.any(theta < -10.0):
            return -np.inf
        fit_gp.params = theta
        fit_gp.compute(t, yerr)
        ll = fit_gp.log_likelihood(y)
        if not np.isfinite(ll):
            return -np.inf
        return ll

    params = fit_gp.params
    ndim = len(params)
    nwalkers = 32
    sampler = emcee.EnsembleSampler(nwalkers, ndim, log_prob, threads=4)
    params = params[None, :] + 1e-8 * np.random.randn(nwalkers, ndim)
    params, lp, _ = sampler.run_mcmc(params, 250)
    params = params[np.argmax(lp)][None, :] + 1e-8*np.random.randn(nwalkers,
                                                                   ndim)
    params, _, _ = sampler.run_mcmc(params, 200)
    sampler.reset()
    sampler.run_mcmc(params, 300)

    samples = sampler.flatchain
    psds = np.empty((len(samples), len(freqs)))
    for i, p in enumerate(samples):
        fit_gp.params = p
        psds[i] = fit_gp.get_psd(freqs)

    q = np.percentile(psds, [16, 50, 84], axis=0)

    ax = axes[1, 1]
    ax.fill_between(freqs, q[0], q[2], color="k", alpha=0.3)
    ax.plot(freqs, q[1], "k", label="posterior")
    ax.plot(freqs, true_gp.get_psd(freqs), "--g", label="truth")
    ax.set_xlim(freqs[0], freqs[-1])
    ax.set_xlabel("f")
    ax.set_ylabel("psd(f)")
    # ax.legend(fontsize=15, loc=1)
    ax.set_xscale("log")
    ax.set_title("posterior")

pl.tight_layout()
fig.savefig("sample.png")

for ax in axes[np.array([[False, True], [True, True]])]:
    ax.set_yscale("log")
fig.savefig("sample-log.png")

pl.close(fig)

if "mcmc" in sys.argv:
    fig = corner.corner(samples)
    fig.savefig("corner.png")
