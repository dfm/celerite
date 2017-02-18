# -*- coding: utf-8 -*-

__version__ = "0.1.2.dev0"

try:
    __CELERITE_SETUP__
except NameError:
    __CELERITE_SETUP__ = False

if not __CELERITE_SETUP__:
    __all__ = [
        "terms", "solver", "modeling", "GP",
        "Solver", "SparseSolver",
        "__library_version__", "__with_lapack__", "__lapack_variant__",
    ]

    from . import terms, solver, modeling
    from .celerite import GP
    from .solver import Solver, SparseSolver

    __library_version__ = solver.get_library_version()
    __with_lapack__ = solver.with_lapack()
    __lapack_variant__ = solver.lapack_variant()
