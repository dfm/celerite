.. module:: celerite.terms

.. _kernel:

Kernel Building
===============

The class of kernels that can be used in celerite are sums of terms of the
following form:

.. math::

    k(\tau) &= \sum_{j=1}^J \frac{1}{2}\left[
        (a_j + i\,b_j)\,e^{-(c_j+i\,d_j)\,\tau} +
        (a_j - i\,b_j)\,e^{-(c_j-i\,d_j)\,\tau}
    \right] \\
    &= \sum_{j=1}^J \frac{1}{2}\left[
        \alpha_j\,e^{-\beta_j\,\tau} +
        {\alpha_j}^*\,e^{-{\beta_j}^*\,\tau}
    \right]

At a basic level, the interface involves providing the solver with lists of
:math:`\alpha_j` and :math:`\beta_j`. It's useful to know if :math:`b_j = 0`
and :math:`d_j = 0` for any terms because we can use that fact to gain
computational efficiency. Therefore, there are 6 arrays that should be
provided to the code:

- ``a_real``: an array of :math:`a_j` for the terms where :math:`b_j = 0`
  and :math:`d_j = 0`,
- ``c_real``: an array of :math:`c_j` for the terms where :math:`b_j = 0`
  and :math:`d_j = 0`,
- ``a_comp``: an array of :math:`a_j` for the other terms,
- ``b_comp``: an array of :math:`b_j` for the other terms,
- ``c_comp``: an array of :math:`c_j` for the other terms, and
- ``d_comp``: an array of :math:`d_j` for the other terms.

In practice, this is further abstracted and users should use the provided term
objects described below or subclass :class:`Term` to implement a custom
kernel. Subclasses should overload the
:func:`Term.get_real_coefficients` and
:func:`Term.get_complex_coefficients` methods to provide custom
:math:`\alpha` and :math:`\beta` arrays.

The following gives a simple example for a custom term of the form:

.. math::

    k(\tau) = \frac{a}{2+b}\,e^{-c\,\tau}\left[
        \cos\left(\frac{2\,\pi\,\tau}{P}\right) + (1+b)
    \right]

with the parameters ``log_a``, ``log_b``, ``log_c``, and ``log_P``.

.. code-block:: python

    # If we use the autograd wrappers around numpy then the gradients
    # of the term will be computed automatically.
    import autograd.numpy as np

    class CustomTerm(terms.Term):
        parameter_names = ("log_a", "log_b", "log_c", "log_P")

        def get_real_coefficients(self, params):
            log_a, log_b, log_c, log_P = params
            b = np.exp(log_b)
            return (
                np.exp(log_a) * (1.0 + b) / (2.0 + b), np.exp(log_c),
            )

        def get_complex_coefficients(self, params):
            log_a, log_b, log_c, log_P = params
            b = np.exp(log_b)
            return (
                np.exp(log_a) / (2.0 + b), 0.0,
                np.exp(log_c), 2*np.pi*np.exp(-log_P),
            )

In this example, all of the returned coefficients are scalars but they can
also be returned as arrays.
To use this kernel, we would use something like the following:

.. code-block:: python

    bounds = dict(log_a=(None, None), log_b=(None, 5.0), log_c=(-1.0, 1.0),
                  log_P=(-0.5, 0.5))
    kernel = CustomTerm(log_a=0.1, log_b=0.5, log_c=-0.01, log_P=0.0,
                        bounds=bounds)

where ``bounds`` is optional.

Finally, terms are combined by adding or multiplying them:

.. code-block:: python

    term1 = terms.RealTerm(0.5, -5.0)
    term2 = terms.ComplexTerm(1.0, 0.1, 0.6, -0.3)
    term3 = terms.SHOTerm(1.0, 0.1, 0.6)

    # Adding kernels works
    kernel = term1 + term2

    # So does multiplying them
    kernel = term1 * term2

    # Or, any combination of these operations
    kernel = term1 * (term2 + term3)

These combined terms are also ``Term`` objects and they provide the same
interface, concatenating the parameters and coefficients in the correct order.

Since ``Term`` objects implement the :ref:`modeling-protocol`, for the kernel
described above, we can do things like the following:

.. code-block:: python

    print(kernel.get_parameter_dict())

With the output:

.. code-block:: python

    OrderedDict([('terms[0]:log_a', 0.5),
                 ('terms[0]:log_c', -5.0),
                 ('terms[1]:log_a', 1.0),
                 ('terms[1]:log_b', 0.1),
                 ('terms[1]:log_c', 0.6),
                 ('terms[1]:log_d', -0.2)])

The following are the ``Term`` objects provided by celerite.

.. autoclass:: celerite.terms.Term
    :members:
.. autoclass:: celerite.terms.RealTerm
.. autoclass:: celerite.terms.ComplexTerm
.. autoclass:: celerite.terms.SHOTerm
.. autoclass:: celerite.terms.Matern32Term
.. autoclass:: celerite.terms.JitterTerm
