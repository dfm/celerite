#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import division, print_function

import os
import sys
import argparse
import numpy as np
import pandas as pd

from celerite.plot_setup import setup, COLOR_CYCLE
setup(auto=True)

parser = argparse.ArgumentParser()
parser.add_argument("with_lapack")
parser.add_argument("without_lapack")
args = parser.parse_args()

print(pd.read_csv(args.with_lapack, comment="#"))
