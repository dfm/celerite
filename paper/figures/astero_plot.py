#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import division, print_function

import os
import sys
import kplr
import pickle
import corner
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from scipy.ndimage.filters import gaussian_filter

import emcee3
from emcee3 import autocorr

from astropy.stats import LombScargle

import genrp
from genrp import terms, modeling

from plot_setup import setup, get_figsize, COLORS

setup()

# Helpers
def format_filename(name):
    base = "astero-{0}-".format(kicid)
    return base + name + ".pdf"

kicid = 11615890
uHz_conv = 1e-6 * 24 * 60 * 60

# Save the current state of the GP and data
with open("astero-{0}.pkl".format(kicid), "rb") as f:
    gp, fit_y, freq, power_all, power_some = pickle.load(f)


assert 0

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
print(sampler.acceptance_fraction)
assert 0


# In[ ]:

plt.plot(sampler.get_coords()[:, :, 3], color="k", alpha=0.3);


# In[ ]:

autocorr.integrated_time(np.mean(sampler.get_coords(discard=1000), axis=1), c=1)


# In[113]:

time_grid = np.linspace(0, 1.4, 5000)
psds = []
acors = []
envs = []
samples = sampler.get_coords(discard=1000, flat=True)
for s in samples[np.random.randint(len(samples), size=1000)]:
#     s = np.array(s)
#     s[7] = 2*s[4]
    gp.set_parameter_vector(s)
    psds.append(gp.kernel.get_psd(2*np.pi*freq))
    acors.append(gp.kernel.get_value(time_grid))
    envs.append(0.5*np.log(2./np.pi) + s[5] - 0.5*(freq - np.exp(s[3]))**2 * np.exp(-s[7]))


# In[114]:

q = np.percentile(acors, [16, 50, 84], axis=0)
plt.fill_between(time_grid * 24, q[0], q[2], color="k", alpha=0.3)
plt.plot(time_grid * 24, q[1], "k", alpha=0.8)
plt.xlabel(r"$\tau$ [hours]")
plt.ylabel(r"$C(\tau)$")
plt.savefig(format_filename("acor"), bbox_inches="tight")


# In[115]:

q = np.percentile(psds, [16, 50, 84], axis=0)
plt.fill_between(freq_uHz, q[0], q[2], color="k", alpha=0.3)
plt.plot(freq_uHz, q[1], "k", alpha=0.8)
plt.yscale("log")
ylim = plt.gca().get_ylim()

# q = np.percentile(np.exp(envs), [16, 50, 84], axis=0)
# plt.fill_between(freq_uHz, q[0], q[2], color="g", alpha=0.3)
# plt.plot(freq_uHz, q[1], "g", alpha=0.8)
plt.ylim(ylim)


# In[116]:

s = np.exp(samples[:, 3:5])/uHz_conv
nu_max_pub = 171.94, 3.62
delta_nu_pub = 13.28, 0.29
fig = corner.corner(s, smooth=0.7, smooth1d=1.0);
fig.axes[2].errorbar(nu_max_pub[0], delta_nu_pub[0], xerr=nu_max_pub[1], yerr=delta_nu_pub[1],
                     fmt=".", color="r", capsize=0, lw=2, mec="none")
fig.savefig(format_filename("numax_deltanu_corner"), bbox_inches="tight")


# In[117]:

corner.corner(samples);


# In[118]:

fig, axes = plt.subplots(3, 1, sharex=True, figsize=(5, 8))

axes[0].plot(freq_uHz, np.sqrt(power_all), "k", alpha=0.3)
axes[0].plot(freq_uHz, np.sqrt(gaussian_filter(power_all, 5)), "k")

axes[1].plot(freq_uHz, np.sqrt(power_some), "k", alpha=0.3)
axes[1].plot(freq_uHz, np.sqrt(gaussian_filter(power_some, 5)), "k")

q = np.percentile(psds, [16, 50, 84], axis=0)
axes[2].fill_between(freq_uHz, q[0], q[2], color="k", alpha=0.3)
axes[2].plot(freq_uHz, q[1], "k", alpha=0.8)

for ax in axes:
    ax.set_yscale("log")

frac = 100 * len(fit_x) / len(x)
axes[0].set_ylabel("periodogram; all data")
axes[1].set_ylabel("periodogram; {0:.0f}\% of data".format(frac))
axes[2].set_ylabel("posterior psd; {0:.0f}\% of data".format(frac))
axes[2].set_xlabel("frequency [$\mu$Hz]")

fig.savefig(format_filename("comparisons"), bbox_inches="tight")


# In[ ]:




# In[ ]:



