# -*- coding: utf-8 -*-

__version__ = "0.0.1.dev0"

try:
    __ESS_SETUP__
except NameError:
    __ESS_SETUP__ = False

if not __ESS_SETUP__:
    __all__ = ["GRPSolver"]

    from ._ess import GRPSolver
