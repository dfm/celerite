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

The :cpp:class:`celerite::solver::CholeskySolver` is an :math:`\mathcal{O}(N)`
solver that exploits the semi-separable structure of the matrix to solve and
compute determinants.

.. doxygenclass:: celerite::solver::CholeskySolver
    :members:
