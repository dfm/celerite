# -*- coding: utf-8 -*-

__version__ = "0.2.1.dev0"
__bibtex__ = """
@article{genrp,
     author = {Sivaram Ambikasaran},
      title = {Generalized Rybicki Press algorithm},
       year = {2015},
    journal = {Numer. Linear Algebra Appl.},
     volume = {22},
     number = {6},
      pages = {1102--1114},
        doi = {10.1002/nla.2003},
        url = {https://arxiv.org/abs/1409.7852}
}

@article{celerite,
    author = {{Foreman-Mackey}, D. and {Agol}, E. and {Angus}, R. and
              {Ambikasaran}, S.},
     title = {Fast and scalable Gaussian process modeling
              with applications to astronomical time series},
      year = {2017},
   journal = {ArXiv},
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
