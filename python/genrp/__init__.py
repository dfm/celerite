# -*- coding: utf-8 -*-

__version__ = "0.0.1.dev0"

try:
    __GENRP_SETUP__
except NameError:
    __GENRP_SETUP__ = False

if not __GENRP_SETUP__:
    __all__ = ["GRPSolver"]

    from .genrp import GRPSolver
