**This project has been superseded by** `celerite2 <https://github.com/exoplanet-dev/celerite2>`_.
**This project will continue to be maintained at a basic level, but no new features will be added and I have limited capacity so you're encouraged to check out the new version.**

**celerite**: Scalable 1D Gaussian Processes in C++, Python, and Julia

Read the documentation at: `celerite.rtfd.io <http://celerite.readthedocs.io>`_.

.. image:: https://img.shields.io/badge/GitHub-dfm%2Fcelerite-blue.svg?style=flat
    :target: https://github.com/dfm/celerite
.. image:: http://img.shields.io/badge/license-MIT-blue.svg?style=flat&bust
    :target: https://github.com/dfm/celerite/blob/main/LICENSE
.. image:: https://github.com/dfm/celerite/actions/workflows/python.yml/badge.svg
    :target: https://github.com/dfm/celerite/actions/workflows/python.yml
.. image:: https://readthedocs.org/projects/celerite/badge/?version=latest&style=flat&bust=truer
    :target: http://celerite.readthedocs.io/en/latest/?badge=latest
.. image:: https://zenodo.org/badge/DOI/10.5281/zenodo.806847.svg?style=flat
   :target: https://doi.org/10.5281/zenodo.806847
.. image:: https://img.shields.io/badge/PDF-latest-orange.svg?style=flat
    :target: https://github.com/dfm/celerite/blob/main-pdf/paper/ms.pdf
.. image:: https://img.shields.io/badge/ArXiv-1703.09710-orange.svg?style=flat
    :target: https://arxiv.org/abs/1703.09710

The Julia implementation is being developed in a different repository:
`ericagol/celerite.jl <https://github.com/ericagol/celerite.jl>`_. Issues
related to that implementation should be opened there.

If you make use of this code, please cite the following paper:

.. code-block:: tex

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
