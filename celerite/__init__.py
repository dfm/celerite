# -*- coding: utf-8 -*-

__version__ = "0.2.0.dev0"

try:
    __CELERITE_SETUP__
except NameError:
    __CELERITE_SETUP__ = False

if not __CELERITE_SETUP__:
    __all__ = [
        "terms", "solver", "modeling", "GP", "CholeskySolver",
        "__library_version__",
    ]

    from . import terms, solver, modeling
    from .celerite import GP
    from .solver import CholeskySolver
    __library_version__ = solver.get_library_version()
