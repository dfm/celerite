# -*- coding: utf-8 -*-

__version__ = "0.1.1"

try:
    __CELERITE_SETUP__
except NameError:
    __CELERITE_SETUP__ = False

if not __CELERITE_SETUP__:
    __all__ = ["Solver", "GP", "terms",
               "__library_version__", "__with_lapack__"]

    from . import terms, solver
    from .celerite import GP
    from .solver import Solver

    __library_version__ = solver.get_library_version()
    __with_lapack__ = solver.with_lapack()
