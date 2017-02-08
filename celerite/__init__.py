# -*- coding: utf-8 -*-

__version__ = "0.1.0.dev0"

try:
    __CELERITE_SETUP__
except NameError:
    __CELERITE_SETUP__ = False

if not __CELERITE_SETUP__:
    __all__ = ["Solver", "GP", "terms"]

    from . import terms
    from .celerite import GP
    from .solver import Solver, get_library_version

    __library_version__ = get_library_version()
