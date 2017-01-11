#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import division, print_function

import os
import sys
import kplr
import pickle
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from scipy.ndimage.filters import gaussian_filter

import emcee3
from emcee3 import autocorr

from astropy.stats import LombScargle

import genrp
from genrp import modeling

from astero_term import AsteroTerm
from plot_setup import setup, get_figsize, COLORS

setup()

def format_filename(name):
    base = "astero-{0}-".format(kicid)
    return base + name + ".pdf"

# Download the data for a giant star from MAST
kicid = 11615890
client = kplr.API()
star = client.star(kicid)

x = []
y = []
yerr = []

for lc in star.get_light_curves():
    data = lc.read()
    x0 = data["TIME"]
    y0 = data["PDCSAP_FLUX"]
    m = (data["SAP_QUALITY"] == 0) & np.isfinite(x0) & np.isfinite(y0)
    x.append(x0[m])
    mu = np.median(y0[m])
    y.append((y0[m] / mu - 1.0) * 1e6)
    yerr.append(1e6 * data["PDCSAP_FLUX_ERR"][m] / mu)

x = np.concatenate(x)
y = np.concatenate(y)
yerr = np.concatenate(yerr)

inds = np.argsort(x)
x = np.ascontiguousarray(x[inds], dtype=float)
y = np.ascontiguousarray(y[inds], dtype=float)
yerr = np.ascontiguousarray(yerr[inds], dtype=float)

# Plot the light curve.
fig, ax = plt.subplots(1, 1, figsize=get_figsize())
ax.plot(x, y, "k", rasterized=True)
ax.set_xlim(x.min(), x.max())
ax.set_ylim(np.std(y) * np.array([-5.0, 5.0]))
ax.set_xlabel("time [KBJD]")
ax.set_ylabel("relative flux [ppm]")
ax.xaxis.set_major_locator(plt.MaxNLocator(4))
fig.savefig(format_filename("time_series"), bbox_inches="tight", dpi=300)
plt.close(fig)

# Define a frequency grid for the periodogram
freq_uHz = np.linspace(1, 300, 50000)
freq = freq_uHz * 1e-6 * 24 * 60 * 60

# Compute the periodogram on the full dataset
model = LombScargle(x, y)
power_all = model.power(freq, method="fast", normalization="psd")

# Select a subset of the data
np.random.seed(1234)
n = int(30 * 48)
n0 = np.random.randint(len(x)-n-1)
fit_x, fit_y, fit_yerr = x[n0:n0+n], y[n0:n0+n], yerr[n0:n0+n]
print("Range in subset of data: {0:.1f} days".format(fit_x.max()-fit_x.min()))
print("Fraction of full dataset: {0:.1f}%".format(100 * n / len(x)))

# Compute the periodogram on the subset
model = LombScargle(fit_x, fit_y)
power_some = model.power(freq, method="fast", normalization="psd")

# Remove background from periodograms
def estimate_background(x, y, log_width=0.005):
    count = np.zeros(len(x), dtype=int)
    bkg = np.zeros_like(x)
    x0 = np.log10(x[0])
    while x0 < np.log10(x[-1]):
        m = np.abs(np.log10(x) - x0) < log_width
        bkg[m] += np.median(y[m])
        count[m] += 1
        x0 += 0.5 * log_width
    return bkg / count
bkg_all = estimate_background(freq_uHz, power_all)
bkg_some = estimate_background(freq_uHz, power_some)

# Plot the periodograms
fig, axes = plt.subplots(1, 2, figsize=get_figsize(1, 2), sharey=True)
axes[0].plot(freq_uHz, np.sqrt(power_all), "k", rasterized=True)
axes[1].plot(freq_uHz, np.sqrt(power_some), "k", rasterized=True)
axes[0].set_ylabel("power")
axes[0].set_xlabel("frequency [$\mu$Hz]")
axes[1].set_xlabel("frequency [$\mu$Hz]")
axes[0].set_title("all data")
axes[1].set_title("subset of data")
[ax.set_yscale("log") for ax in axes]
fig.savefig(format_filename("periodogram"), bbox_inches="tight", dpi=300)
plt.close(fig)

# Compute $\nu_\mathrm{max}$ and $\Delta \nu$ from the full dataset
for name, ps in zip(("subset of data", "all data"),
                    (power_some-bkg_some, power_all-bkg_all)):
    # Compute the smoothed power spectrum
    df = freq_uHz[1] - freq_uHz[0]
    smoothed_ps = gaussian_filter(ps, 10 / df)

    # And the autocorrelation function of a lightly smoothed power spectrum
    acor_func = autocorr.function(gaussian_filter(ps, 0.5 / df))
    lags = df*np.arange(len(acor_func))
    acor_func = acor_func[lags < 30]
    lags = lags[lags < 30]

    # Find the peaks
    def find_peaks(z):
        peak_inds = (z[1:-1] > z[:-2]) * (z[1:-1] > z[2:])
        peak_inds = np.arange(1, len(z)-1)[peak_inds]
        peak_inds = peak_inds[np.argsort(z[peak_inds])][::-1]
        return peak_inds

    peak_freqs = freq_uHz[find_peaks(smoothed_ps)]
    nu_max = peak_freqs[peak_freqs > 5][0]

    # Expected delta_nu: Stello et al (2009)
    dnu_expected = 0.263 * nu_max ** 0.772
    peak_lags = lags[find_peaks(acor_func)]
    delta_nu = peak_lags[np.argmin(np.abs(peak_lags - dnu_expected))]
    print("{0}: nu_max = {1}, delta_nu = {2}".format(name, nu_max, delta_nu))

    # Plot the smoothed power spectrum and autocorrelation function
    fig, axes = plt.subplots(1, 2, figsize=get_figsize(1, 2))
    axes[0].plot(freq_uHz, smoothed_ps, "k")
    axes[0].axvline(nu_max, color=COLORS["MODEL_1"])
    axes[0].set_ylabel("smoothed power spectrum")
    axes[0].set_xlabel("frequency [$\mu$Hz]")

    axes[1].plot(lags, acor_func, "k")
    axes[1].axvline(delta_nu, color=COLORS["MODEL_1"])
    axes[1].set_ylabel("autocorrelation function")
    axes[1].set_xlabel("frequency spacing [$\mu$Hz]")
    axes[1].set_xlim(0, 30)

    for ax in axes:
        ax.annotate(name, xy=(1, 1), xycoords="axes fraction",
                    xytext=(-5, -5), textcoords="offset points",
                    ha="right", va="top")

    fig.savefig(format_filename("numax_deltanu_"+name.split()[0]),
                bbox_inches="tight")
    plt.close(fig)

# Factor to convert between day^-1 and uHz
uHz_conv = 1e-6 * 24 * 60 * 60

# Parameter bounds
bounds = [(-15, 15) for _ in range(8)]
bounds[2] = np.log(nu_max*uHz_conv) + np.array([-0.1, 0.1])
bounds[3] = np.log(delta_nu*uHz_conv) + np.array([-0.1, 0.1])

# Set up the GP model
kernel = AsteroTerm(
    np.log(np.var(y)),
    2.0,
    np.log(nu_max*uHz_conv),        # log(nu_max)
    np.log(delta_nu*uHz_conv),      # log(delta_nu)
    0.0,                            # offset between nu_max and central mode
    np.log(np.var(y)),              # log(amp_max)
    5.0,                            # log(q_factor)
    np.log(delta_nu*uHz_conv),      # width of envelope
    bounds=bounds,
    nterms=1,
)
log_white_noise = modeling.ConstantModel(
    2.0*np.log(np.median(np.abs(np.diff(fit_y)))),
    bounds=[(-15, 15)]
)
gp = genrp.GP(kernel, log_white_noise=log_white_noise)
gp.compute(fit_x, fit_yerr)
print("Initial log-likelihood: {0}".format(gp.log_likelihood(fit_y)))
print(gp.get_parameter_dict(include_frozen=True))

# The objective function for optimization
def nll(params):
    gp.set_parameter_vector(params)
    ll = gp.log_likelihood(fit_y)
    if not np.isfinite(ll):
        return 1e10
    return -ll+0.5*gp.kernel.epsilon**2

# Grid initialize
print("Running a grid of optimizations...")
gp.log_white_noise.thaw_all_parameters()
gp.kernel.thaw_all_parameters()
initial = np.array(gp.get_parameter_vector())

def get_ml_params(log_nu_max):
    gp.set_parameter_vector(initial)
    gp.kernel.set_parameter("log_nu_max", log_nu_max)
    gp.kernel.set_parameter(
        "log_delta_nu",
        np.log(0.263 * (np.exp(log_nu_max)/uHz_conv) ** 0.772 * uHz_conv)
    )
    p0 = gp.get_parameter_vector()
    bounds = gp.get_parameter_bounds()
    r = minimize(nll, p0, method="L-BFGS-B", bounds=bounds)
    gp.set_parameter_vector(r.x)
    return r.fun, r.x

with emcee3.pools.InterruptiblePool() as pool:
    results = list(sorted(pool.map(
        get_ml_params, gp.kernel.log_nu_max + np.linspace(-0.05, 0.05, 5)
    ), key=lambda o: o[0]))
gp.set_parameter_vector(results[0][1])

# Use more modes in the MCMC:
gp.kernel.nterms = 3

fig, ax = plt.subplots(1, 1, figsize=get_figsize())
ax.plot(freq_uHz, np.sqrt(power_all), "k", alpha=0.8, rasterized=True)
ax.plot(freq_uHz, gp.kernel.get_psd(2*np.pi*freq), alpha=0.5,
        rasterized=True)
ax.set_xlabel("frequency [$\mu$Hz]")
ax.set_ylabel("power")
ax.set_yscale("log")
fig.savefig(format_filename("initial_psd"), bbox_inches="tight", dpi=300)

# Set up the probabilistic model for sampling
def log_prob(p):
    gp.set_parameter_vector(p)
    lp = gp.log_prior()
    if not np.isfinite(lp):
        return -np.inf
    ll = gp.log_likelihood(fit_y)
    if not np.isfinite(ll):
        return -np.inf
    return ll + lp

# Initialize and set bounds
ndim, nwalkers = gp.vector_size, 32
initial_samples = \
    gp.get_parameter_vector() + 1e-5 * np.random.randn(nwalkers, ndim)
gp.kernel.parameter_bounds[2] = np.log(nu_max*uHz_conv) + np.array([-1, 1])
gp.kernel.parameter_bounds[3] = np.log(delta_nu*uHz_conv) + np.array([-1, 1])

# Save the current state of the GP and data
with open("astero-{0}.pkl".format(kicid), "wb") as f:
    pickle.dump((
        gp, fit_y, freq, power_all, power_some,
    ), f, -1)

if os.path.exists("astero-{0}.h5".format(kicid)):
    result = input("MCMC save file exists. Overwrite? (type 'yes'): ")
    if result.lower() != "yes":
        sys.exit(0)

# Define a custom proposal
def astero_move(rng, x0):
    x = np.array(x0)
    f = 2.0 * (rng.rand(len(x)) < 0.5) - 1.0
    x[:, 3] = np.log(np.exp(x[:, 3]) + f * np.exp(x[:, 4]))
    return x, np.zeros(len(x))

# The sampler will use a mixture of proposals
sampler = emcee3.Sampler([
    emcee3.moves.StretchMove(),
    emcee3.moves.DEMove(1e-3),
    emcee3.moves.KDEMove(),
    emcee3.moves.MHMove(astero_move),
], backend=emcee3.backends.HDFBackend("astero-{0}.h5".format(kicid)))

# Sample!
with emcee3.pools.InterruptiblePool() as pool:
    ensemble = emcee3.Ensemble(emcee3.SimpleModel(log_prob), initial_samples,
                               pool=pool)
    ensemble = sampler.run(ensemble, 10000, progress=True)
