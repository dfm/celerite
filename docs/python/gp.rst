.. module:: celerite

.. _gp:

Gaussian Process Computations
=============================

The main interface to the celerite solver is through the :class:`GP` class.
This main purpose of this class is to exposes methods for efficiently
computing the marginalized likelihood of a Gaussian Process (GP) model.
The covariance matrix for the GP will be specified by a kernel function as
described in the :ref:`kernel` section.

The :class:`GP` class implements the :ref:`modeling` with submodels called
``kernel``, ``mean``, and ``log_white_noise`` so the modeling language can be
used to fit for parameters for each of these models.

Below, the methods of the :class:`GP` object are described in detail but the
most important methods to look at are probably :func:`GP.compute`,
:func:`GP.log_likelihood`, and :func:`GP.predict`.

.. autoclass:: celerite.GP
    :members:
