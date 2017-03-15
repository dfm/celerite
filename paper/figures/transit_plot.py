#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import division, print_function

import corner
import emcee3
import pickle
import numpy as np

from celerite.plot_setup import setup

setup(auto=True)

gp, y, true_params = pickle.load(open("transit.pkl", "rb"))
f = emcee3.backends.HDFBackend("transit.h5")

# Plot the parameter constraints
names = gp.get_parameter_names()
cols = ["log_period", "log_ror", "log_duration", "t0"]
inds = [names.index("mean:{0}".format(c)) for c in cols]
samples = np.array(f.get_coords(discard=5000, flat=True, thin=13))
samples = samples[:, inds]
samples[:, :-1] = np.exp(samples[:, :-1])
truths = np.array([true_params[k] for k in cols])
truths[:-1] = np.exp(truths[:-1])
fig = corner.corner(samples, truths=truths, smooth=0.5,
                    labels=[r"period", r"$R_\mathrm{P}/R_\star$", r"duration",
                            r"$t_0$"])
for ax in np.array(fig.axes).flatten():
    ax.xaxis.set_label_coords(0.5, -0.4)
    ax.yaxis.set_label_coords(-0.4, 0.5)
fig.savefig("transit-corner.pdf")
