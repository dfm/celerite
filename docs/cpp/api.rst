.. _cpp-api:

API
===

Solver Interface
----------------

The main interface to the celerite solver is via the
:cpp:class:`celerite::solver::CholeskySolver` class but many of the key
interfaces are abstracted into the :cpp:class:`celerite::solver::Solver`.

.. doxygenclass:: celerite::solver::Solver
    :members:

Fast Solver
-----------

The :cpp:class:`celerite::solver::CholeskySolver` is an implementation of
:cpp:class:`celerite::solver::Solver` that exploits the semi-separable nature
of the matrix to solve and compute determinants in :math:`\mathcal{O}(N)`
operations.

.. doxygenclass:: celerite::solver::CholeskySolver
    :members:
