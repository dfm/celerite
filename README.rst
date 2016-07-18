Python bindings to the
`sivaramambikasaran/ESS <https://github.com/sivaramambikasaran/ESS>`_
implementation of the generalized Rybicki Press algorithm for solving
matrices of the form::

    K_{ij} = sum_p a_p exp(-b_p |t_i - t_j|)

This interface allows complex bs and the parameters are specified as
log-amplitudes, log-Q-factors, and frequencies. A frequency can be set to
``None`` if it is meant to be non-periodic.

A simple benchmark shows that this scales as O(N):

.. code-block:: python

    import time
    import numpy as np
    import matplotlib.pyplot as pl

    from ess import GRPSolver

    solver = GRPSolver(
        np.log([10.0, 5.0]),  # log-amplitudes
        np.log([0.1, 10.0]),  # log-Q-factors
        [None, 50.0],         # frequencies
    )

    N = 2**np.arange(5, 20)
    times = np.empty((len(N), 3))

    t = np.random.rand(np.max(N))
    yerr = np.random.uniform(0.1, 0.2, len(t))
    b = np.random.randn(len(t))

    for i, n in enumerate(N):
        strt = time.time()
        solver.compute(t[:n], yerr[:n])
        times[i, 0] = time.time() - strt

        strt = time.time()
        solver.log_determinant
        times[i, 1] = time.time() - strt

        strt = time.time()
        solver.apply_inverse(b[:n])
        times[i, 2] = time.time() - strt

.. image:: https://raw.github.com/dfm/ess/master/demo.png
