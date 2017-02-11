# -*- coding: utf-8 -*-

from __future__ import division, print_function
from timeit import Timer

__all__ = ["benchmark"]

def benchmark(stmt, setup="pass"):
    timer = Timer(stmt, setup)
    total = 0.0
    k = 1
    while total < 0.2:
        total = min(timer.repeat(3, k))
        k *= 10
    return 10 * total / k
