#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import division, print_function

import numpy as np
from itertools import product

import matplotlib
import matplotlib.pyplot as plt

from celerite.plot_setup import setup, get_figsize

setup()

J = 2
width = 2*J + 1 + 4*J + 1 + 2*J
height = 1 + 4*J

block = np.zeros((height, width), dtype=int)

# y_n
block[0, 2*J] = 1  # diag
block[0, :2*J:2] = 2  # a/2
block[0, 1:2*J:2] = 3  # b/2
block[0, 2*J+1:4*J+1:2] = 4  # phi
block[0, 2*J+2:4*J+1:2] = 5  # psi

# g_{n,j}
for j in range(J):
    block[1+2*j, 2*j] = 4  # phi
    block[1+2*j, 2*j+1] = 5  # psi
    block[1+2*j, 2*J] = 4  # phi
    block[1+2*j, 4*J+2*j+1] = 7  # -1

# h_{n,j}
for j in range(J):
    block[2+2*j, 2*j] = 5  # psi
    block[2+2*j, 2*j+1] = 6  # -phi
    block[2+2*j, 2*J] = 5  # psi
    block[2+2*j, 4*J+2*j+2] = 8  # 1

# u_{n,j}
for j in range(J):
    block[1+2*J+2*j, 6*J+1] = 2  # a/2
    block[1+2*J+2*j, 6*J+2+2*j] = 4  # phi
    block[1+2*J+2*j, 6*J+3+2*j] = 5  # psi
    block[1+2*J+2*j, 2*J+2*j+1] = 7  # -1

# v_{n,j}
for j in range(J):
    block[2+2*J+2*j, 6*J+1] = 3  # b/2
    block[2+2*J+2*j, 6*J+2+2*j] = 5  # psi
    block[2+2*J+2*j, 6*J+3+2*j] = 6  # -phi
    block[2+2*J+2*j, 2*J+2*j+2] = 8  # 1

# Combine the blocks
N = 5
full_dim = (1 + 4*J) * (N-1) + 1
full = np.zeros((full_dim, full_dim), dtype=int)

full[:height, :6*J+2] = block[:, 2*J:]

for n in range(1, N):
    x_strt = n*height
    x_end = min((n+1)*height, full_dim)
    y_strt = 2*J+1+(n-1)*(4*J+1)
    y_end = min(y_strt+width, full_dim)
    full[x_strt:x_end, y_strt:y_end] = block[:x_end-x_strt, :y_end-y_strt]

nband = len(full) * 2*(2*J+2)
nnz = np.sum(full > 0)
print("nnz: {0}, {1}".format(nnz, (20*J+1)*N - 28*J))
# print("nnz: {0}, {1}".format(nnz, (20*J+1)*(N-2)+12*J+2))

# Legend
d = 1
for i in range(8):
    full[-2*d-2*d*i:-d-2*d*i, d:2*d] = 8-i

# Seaborn set1
c = [(0.89411765336990356, 0.10196078568696976, 0.10980392247438431),
     (0.21602460800432691, 0.49487120380588606, 0.71987698697576341),
     (0.30426760128900115, 0.68329106055054012, 0.29293349969620797),
     (0.60083047361934883, 0.30814303335021526, 0.63169552298153153),
     (1.0, 0.50591311045721465, 0.0031372549487095253),
     (0.99315647868549117, 0.9870049982678657, 0.19915417450315812),
     (0.65845446095747107, 0.34122261685483596, 0.1707958535236471),
     (0.95850826852461868, 0.50846600392285513, 0.74492888871361229),
     (0.60000002384185791, 0.60000002384185791, 0.60000002384185791),
     (0.89411765336990356, 0.10196078568696976, 0.10980392247438431)]
cmap = matplotlib.colors.ListedColormap(["white"] + list(c), name="cmap")
fig, ax = plt.subplots(1, 1, figsize=get_figsize(2.3, 2.3))
ax.pcolor(full, cmap=cmap, vmin=0, vmax=len(c))

# Plot the edges
for i, j in product(range(full_dim), range(full_dim)):
    if full[i, j] == 0:
        continue
    ax.plot((j, j, j+1, j+1, j), (i, i+1, i+1, i, i), "k", lw=0.5)

ax.set_ylim(full_dim, 0)
ax.set_xlim(0, full_dim)
ax.set_frame_on(False)
ax.set_xticks([])
ax.set_yticks([])

names = [
    r"${\sigma_n}^2 + \sum a_j$",
    r"$a_j$",
    r"$b_j$",
    r"$\phi_{n,j}$",
    r"$\psi_{n,j}$",
    r"$-\phi_{n,j}$",
    r"$-1$",
    r"$1$",
]
for i, name in enumerate(names):
    ax.annotate(name, xy=(2.75, full_dim+0.5 - 2*(8-i)), xycoords="data",
                ha="left", va="center")

ax.plot([0.25, 0.25, 8.5, 8.5, 0.25],
        full_dim - np.array([0.25, 16.75, 16.75, 0.25, 0.25]),
        "k")

fig.savefig("matrix.pdf", bbox_inches="tight")

# From: https://github.com/MattShannon/bandmat
def band_e(l, u, mat_full):
    assert l >= 0
    assert u >= 0
    assert mat_full.shape[1] == mat_full.shape[0]
    size = mat_full.shape[0]
    mat_rect = np.zeros((l + u + 1, size), dtype=int)
    for i in range(-u, l + 1):
        row = u + i
        for j in range(size):
            if j + i < 0:
                continue
            mat_rect[row, j] = mat_full[j + i, j] if j + i < size else 0.0
    return mat_rect


band = band_e(2*J+2, 2*J+2, full)
print(band.shape)
fig, ax = plt.subplots(1, 1, figsize=get_figsize(2.3, 2.3))
ax.pcolor(band, cmap=cmap, vmin=0, vmax=len(c))
ax.set_ylim(band.shape[0], 0)
ax.set_xlim(0, band.shape[1])
ax.set_frame_on(False)
ax.set_xticks([])
ax.set_yticks([])
ax.set_aspect("equal")
fig.savefig("matrix-band.pdf", bbox_inches="tight")
