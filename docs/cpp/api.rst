.. _cpp-api:

API
===

Fast Solver
-----------

The main interface to the celerite solver is via the
:cpp:class:`celerite::solver::BandSolver` class but many of the key interfaces
are abstracted into the :cpp:class:`celerite::solver::Solver`.

.. doxygenclass:: celerite::solver::Solver
    :members:

.. doxygenclass:: celerite::solver::BandSolver
    :members:
