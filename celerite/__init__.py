# -*- coding: utf-8 -*-

__version__ = "0.3.1"
__bibtex__ = """
@article{celerite,
    author = {{Foreman-Mackey}, D. and {Agol}, E. and {Angus}, R. and
              {Ambikasaran}, S.},
     title = {Fast and scalable Gaussian process modeling
              with applications to astronomical time series},
      year = {2017},
   journal = {AJ},
    volume = {154},
     pages = {220},
       doi = {10.3847/1538-3881/aa9332},
       url = {https://arxiv.org/abs/1703.09710}
}
"""

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
