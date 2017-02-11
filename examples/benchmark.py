#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import division, print_function

import os
import sys
import argparse
import numpy as np
import numpy.__config__ as npconf

import celerite
from celerite import GP, terms
from celerite.timer import benchmark

parser = argparse.ArgumentParser()
parser.add_argument("--lapack", action="store_true")
parser.add_argument("--minnpow", type=int, default=6)
parser.add_argument("--maxnpow", type=int, default=19)
parser.add_argument("--minjpow", type=int, default=0)
parser.add_argument("--maxjpow", type=int, default=7)
parser.add_argument("--outdir",
                    default=os.path.dirname(os.path.abspath(__file__)))
args = parser.parse_args()

# BLAS info
blas_opt_info = npconf.get_info("blas_opt_info")

# The dimension of the problem
N = 2**np.arange(args.minnpow, args.maxnpow + 1)
J = 2**np.arange(args.minjpow, args.maxjpow + 1)

header = ""
for k in ["lapack", "minnpow", "maxnpow", "minjpow", "maxjpow"]:
    header += "# {0}: {1}\n".format(k, getattr(args, k))
header += "# platform: {0}\n".format(sys.platform)
header += "# with_lapack: {0}\n".format(celerite.__with_lapack__)
header += "# blas_opt_info: {0}\n".format(blas_opt_info)
header += "# N: {0}\n".format(list(N))
header += "# J: {0}\n".format(list(J))
header += "xi,yi,j,n,comp_time,ll_time\n"

fn = "benchmark_{0}".format(sys.platform)
if args.lapack:
    fn += "_lapack"
fn += ".csv"
fn = os.path.join(args.outdir, fn)
print("filename: {0}".format(fn))
with open(fn, "w") as f:
    f.write(header)
print(header, end="")

# Simulate a "dataset"
np.random.seed(42)
t = np.sort(np.random.rand(np.max(N)))
yerr = np.random.uniform(0.1, 0.2, len(t))
y = np.sin(t)

for xi, j in enumerate(J):
    kernel = terms.RealTerm(1.0, 0.1)
    for k in range((j - 1) % 2):
        kernel += terms.RealTerm(1.0, 0.1)
    for k in range((j - 1) // 2):
        kernel += terms.ComplexTerm(0.1, 2.0, 1.6)
    c = kernel.coefficients
    assert j == len(c[0]) + 2*len(c[2]), "Wrong number of terms"
    gp = GP(kernel, use_lapack=args.lapack)
    for yi, n in enumerate(N):
        comp_time = benchmark("gp.compute(t[:{0}], yerr[:{0}])".format(n),
                            "from __main__ import gp, t, yerr")
        gp.compute(t[:n], yerr[:n])
        ll_time = benchmark("gp.log_likelihood(y[:{0}])".format(n),
                            "from __main__ import gp, y")
        msg = "{0},{1},{2},{3},{4:e},{5:e}\n".format(xi, yi, j, n, comp_time,
                                                     ll_time)
        with open(fn, "a") as f:
            f.write(msg)
        print(msg, end="")

        if comp_time + ll_time >= 1:
            break
