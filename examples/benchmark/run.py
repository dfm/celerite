#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import division, print_function

import os
import sys
import argparse
import numpy as np
from scipy.linalg import cho_factor, cho_solve

from celerite import terms
from celerite.timer import benchmark
from celerite.solver import CholeskySolver

parser = argparse.ArgumentParser()
parser.add_argument("--grad", action="store_true")
parser.add_argument("--george", action="store_true")
parser.add_argument("--carma", action="store_true")
parser.add_argument("--minnpow", type=int, default=6)
parser.add_argument("--maxnpow", type=int, default=19)
parser.add_argument("--minjpow", type=int, default=0)
parser.add_argument("--maxjpow", type=int, default=8)
parser.add_argument("--outdir",
                    default=os.path.dirname(os.path.abspath(__file__)))
args = parser.parse_args()

if args.george:
    try:
        import george
        from george.kernels import CeleriteKernel
    except ImportError:
        print("To run the george benchmark, you must install the dev version "
              "of george with the 'CeleriteKernel' included")
        raise

# The dimension of the problem
N = 2**np.arange(args.minnpow, args.maxnpow + 1)
J = 2**np.arange(args.minjpow, args.maxjpow + 1)

header = ""
for k in ["grad", "george",
          "minnpow", "maxnpow", "minjpow", "maxjpow"]:
    header += "# {0}: {1}\n".format(k, getattr(args, k))
header += "# platform: {0}\n".format(sys.platform)
header += "# N: {0}\n".format(list(N))
header += "# J: {0}\n".format(list(J))
header += "xi,yi,j,n,comp_time,ll_time,numpy_comp_time,numpy_ll_time\n"

fn = "benchmark_{0}".format(sys.platform)
if args.grad:
    fn += "_grad"
elif args.george:
    fn += "_george"
elif args.carma:
    fn += "_carma"

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

def numpy_compute(kernel, x, yerr):
    K = kernel.get_value(x[:, None] - x[None, :])
    K[np.diag_indices_from(K)] += yerr ** 2
    return cho_factor(K)

def numpy_log_like(factor, y):
    np.dot(y, cho_solve(factor, y)) + np.sum(np.log(np.diag(factor[0])))

for xi, j in enumerate(J):
    kernel = terms.RealTerm(1.0, 0.1)
    for k in range((2*j - 1) % 2):
        kernel += terms.RealTerm(1.0, 0.1)
    for k in range((2*j - 1) // 2):
        kernel += terms.ComplexTerm(0.1, 2.0, 1.6)
    coeffs = kernel.coefficients
    assert 2*j == len(coeffs[0]) + 2*len(coeffs[2]), "Wrong number of terms"

    if args.george:
        george_kernel = None
        for a, c in zip(*(coeffs[:2])):
            k = CeleriteKernel(a=a, b=0.0, c=c, d=0.0)
            george_kernel = k if george_kernel is None else george_kernel + k
        for a, b, c, d in zip(*(coeffs[2:])):
            k = CeleriteKernel(a=a, b=0.0, c=c, d=0.0)
            george_kernel = k if george_kernel is None else george_kernel + k
        solver = george.GP(george_kernel, solver=george.HODLRSolver)
    elif args.carma:
        arparams = np.random.randn(2*j)
        maparams = np.random.randn(2*j - 1)
    else:
        solver = CholeskySolver()

    for yi, n in enumerate(N):
        np_comp_time = np.nan
        np_ll_time = np.nan
        if args.george:
            params = [t[:n], yerr[:n]]
            comp_time = benchmark("solver.compute(*params)",
                                  "from __main__ import solver, params")
            solver.compute(*params)
            y0 = y[:n]
            ll_time = benchmark("solver.lnlikelihood(y0)",
                                "from __main__ import solver, y0")
        elif args.carma:
            params = [arparams, maparams]
            funcargs = [t[:n], y[:n], yerr[:n]**2]
            comp_time = benchmark(
                "solver = CARMASolver(0.0, *params)\n"
                "solver.log_likelihood(*funcargs)",
                "from __main__ import params, funcargs\n"
                "from celerite.solver import CARMASolver"
            )
            ll_time = 0.0
        elif args.grad:
            params = [0.0] + list(coeffs)
            params += [t[:n], y[:n], yerr[:n]**2]
            comp_time = benchmark("solver.grad_log_likelihood(*params)",
                                  "from __main__ import solver, params")
            ll_time = 0.0
        else:
            params = [0.0] + list(coeffs)
            params += [t[:n], yerr[:n]**2]
            comp_time = benchmark("solver.compute(*params)",
                                  "from __main__ import solver, params")
            solver.compute(*params)
            y0 = y[:n]
            ll_time = benchmark("solver.dot_solve(y0)",
                                "from __main__ import solver, y0")

            if xi == 0 and n <= 8192:
                # Do numpy calculation
                params = [kernel, t[:n], yerr[:n]]
                np_comp_time = benchmark("numpy_compute(*params)",
                                         "from __main__ import numpy_compute, "
                                         "params")
                factor = numpy_compute(*params)
                params = [factor, y[:n]]
                np_ll_time = benchmark("numpy_log_like(*params)",
                                       "from __main__ import "
                                       "numpy_log_like, params")

        msg = ("{0},{1},{2},{3},{4:e},{5:e},{6:e},{7:e}\n"
               .format(xi, yi, j, n, comp_time, ll_time, np_comp_time,
                       np_ll_time))
        with open(fn, "a") as f:
            f.write(msg)
        print(msg, end="")

        if comp_time + ll_time >= 5:
            break
