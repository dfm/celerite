**celerite** — Scalable 1D Gaussian Processes in C++, Python, and Julia

Read the documentation at: `celerite.rtfd.io <http://celerite.readthedocs.io>`_.

.. image:: https://img.shields.io/badge/GitHub-dfm%2Fcelerite-blue.svg?style=flat
    :target: https://github.com/dfm/celerite
.. image:: http://img.shields.io/badge/license-MIT-blue.svg?style=flat&bust
    :target: https://github.com/dfm/celerite/blob/master/LICENSE
.. image:: http://img.shields.io/travis/dfm/celerite/master.svg?style=flat
    :target: https://travis-ci.org/dfm/celerite
.. image:: https://ci.appveyor.com/api/projects/status/74al24yklrlrvwni/branch/master?svg=true&style=flat
    :target: https://ci.appveyor.com/project/dfm/celerite
.. image:: https://readthedocs.org/projects/celerite/badge/?version=latest&style=flat
    :target: http://celerite.readthedocs.io/en/latest/?badge=latest
.. image:: https://zenodo.org/badge/DOI/10.5281/zenodo.438359.svg?style=flat
   :target: https://doi.org/10.5281/zenodo.438359
.. image:: https://img.shields.io/badge/PDF-latest-orange.svg?style=flat
    :target: https://github.com/dfm/celerite/blob/master-pdf/papers/paper1/ms.pdf
.. image:: https://img.shields.io/badge/ArXiv-1703.09710-orange.svg?style=flat
    :target: https://arxiv.org/abs/1703.09710

The Julia implementation is being developed in a different repository:
`ericagol/celerite.jl <https://github.com/ericagol/celerite.jl>`_. Issues
related to that implementation should be opened there.

If you make use of this code, please cite the following papers:

.. code-block:: tex

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
