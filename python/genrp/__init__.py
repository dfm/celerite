# -*- coding: utf-8 -*-

__version__ = "0.0.1.dev1"

try:
    __GENRP_SETUP__
except NameError:
    __GENRP_SETUP__ = False

if not __GENRP_SETUP__:
    __all__ = ["Solver", "GP"]

    from .genrp import GP
    from ._genrp import Solver, get_library_version

    __library_version__ = get_library_version()
