# -*- coding: utf-8 -*-

from __future__ import division, print_function

from cycler import cycler
from matplotlib import rcParams
try:
    from savefig import monkey_patch
except ImportError:
    def monkey_patch(include_diff=False):
        pass

__all__ = ["setup", "get_figsize", "COLORS", "COLOR_CYCLE", "SQUARE_FIGSIZE"]

COLOR_CYCLE = (
    "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b",
    "#e377c2", "#7f7f7f", "#bcbd22", "#17becf",
)

COLORS = dict(
    DATA="k",
    MODEL_1="#1f77b4",
    MODEL_2="#ff7f0e",
)

def setup(auto=False):
    monkey_patch()
    rcParams["font.size"] = 16
    rcParams["font.family"] = "sans-serif"
    rcParams["font.sans-serif"] = ["Computer Modern Sans"]
    rcParams["text.usetex"] = True
    rcParams["text.latex.preamble"] = r"\usepackage{cmbright}"
    if auto:
        rcParams["figure.autolayout"] = True
    rcParams["axes.prop_cycle"] = cycler("color", COLOR_CYCLE)

def get_figsize(rows=1, cols=1):
    return (4 * cols, 4 * rows)


SQUARE_FIGSIZE = get_figsize(1, 1)
