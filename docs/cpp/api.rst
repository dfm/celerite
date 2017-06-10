.. _cpp-api:

API
===

Fast Solver
-----------

The :cpp:class:`celerite::solver::CholeskySolver` is an :math:`\mathcal{O}(N)`
solver that exploits the semi-separable structure of the matrix to solve and
compute determinants.

.. doxygenclass:: celerite::solver::CholeskySolver
    :members:
