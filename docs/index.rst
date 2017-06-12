celerite
========

.. raw:: html

    <div style="color: #555; border-left: 8px solid #ddd; padding: 5px 15px 8px; margin: 20px 0">
    <strong>celerite</strong> &nbsp; &nbsp;
    \se.le.ʁi.te\ &nbsp; &nbsp;
    <i>noun, archaic literary</i><br>
    A scalable method for Gaussian Process regression. From French
    <i>célérité</i>.
    </div>


celerite is a library for fast and scalable Gaussian Process (GP) Regression
in one dimension with implementations in C++, Python, and Julia. The Python
implementation is the most stable and it exposes the most features but it
relies on the C++ implementation for computational efficiency. This
documentation won't teach you the fundamentals of GP modeling but the best
resource for learning about this is available for free online: `Rasmussen &
Williams (2006) <http://www.gaussianprocess.org/gpml/>`_.

The celerite API is designed to be familiar to users of `george
<https://github.com/dfm/george>`_ and, like george, celerite is designed to
efficiently evaluate the marginalized likelihood of a dataset under a GP
model. This is then meant to be used alongside your favorite `non-linear
optimization
<http://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.minimize.html>`_
or `posterior inference <http://dfm.io/emcee>`_ library for the best results.

celerite is being actively developed in `a public repository on GitHub
<https://github.com/dfm/celerite>`_ so if you have any trouble, `open an issue
<https://github.com/dfm/celerite/issues>`_ there.

.. note:: To work with the Julia package manager, the Julia implementation of
    the algorithm is being developed in `a separate
    repository <https://github.com/ericagol/celerite.jl>`_ but the
    documentation is still included here.

.. image:: https://img.shields.io/badge/GitHub-dfm%2Fcelerite-blue.svg?style=flat
    :target: https://github.com/dfm/celerite
.. image:: http://img.shields.io/badge/license-MIT-blue.svg?style=flat
    :target: https://github.com/dfm/celerite/blob/master/LICENSE
.. image:: http://img.shields.io/travis/dfm/celerite/master.svg?style=flat
    :target: https://travis-ci.org/dfm/celerite
.. image:: https://ci.appveyor.com/api/projects/status/74al24yklrlrvwni?svg=true&style=flat
    :target: https://ci.appveyor.com/project/dfm/celerite

.. raw:: html

    <br>

.. image:: https://zenodo.org/badge/DOI/10.5281/zenodo.806847.svg?style=flat
   :target: https://doi.org/10.5281/zenodo.806847
.. image:: https://img.shields.io/badge/ArXiv-1703.09710-orange.svg?style=flat
    :target: https://arxiv.org/abs/1703.09710

.. toctree::
   :maxdepth: 2
   :caption: Python Usage

   python/install
   python/gp
   python/kernel
   python/modeling
   python/solver
   python/benchmark

.. toctree::
   :maxdepth: 1
   :caption: Python Tutorials

   tutorials/first
   tutorials/modeling
   tutorials/normalization

.. toctree::
   :maxdepth: 2
   :caption: C++ Usage

   cpp/install
   cpp/start
   cpp/api

.. toctree::
   :maxdepth: 2
   :caption: Julia Usage

   julia/install


.. toctree::
   :maxdepth: 1
   :caption: Julia Tutorials

   tutorials/julia-first


Contributors
------------

.. include:: ../AUTHORS.rst


License & Attribution
---------------------

Copyright 2016, 2017, Daniel Foreman-Mackey, Eric Agol and contributors.

The source code is made available under the terms of the MIT license.

If you make use of this code, please cite the following paper:

.. code-block:: tex

    @article{celerite,
        author = {{Foreman-Mackey}, D. and {Agol}, E. and {Angus}, R. and
                  {Ambikasaran}, S.},
         title = {Fast and scalable Gaussian process modeling
                  with applications to astronomical time series},
          year = {2017},
       journal = {ArXiv},
           url = {https://arxiv.org/abs/1703.09710}
    }


Changelog
---------

.. include:: ../HISTORY.rst
