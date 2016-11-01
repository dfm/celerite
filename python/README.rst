A simple benchmark shows that this scales as O(N):

.. code-block:: python

    import time
    import numpy as np
    import matplotlib.pyplot as pl

    from genrp import GP

    gp = GP()
    gp.add_term(1.0, 0.1)
    gp.add_term(0.1, 2.0, 1.6)

    N = 2**np.arange(5, 20)
    K = np.maximum((N.max() / N), 5*np.ones_like(N)).astype(int)
    times = np.empty((len(N), 2))

    t = np.sort(np.random.rand(np.max(N)))
    yerr = np.random.uniform(0.1, 0.2, len(t))
    y = np.sin(t)

    for i, n in enumerate(N):
        strt = time.time()
        for k in range(K[i]):
            gp.compute(t[:n], yerr[:n])
        times[i, 0] = (time.time() - strt) / K[i]

        strt = time.time()
        for k in range(K[i]):
            gp.log_likelihood(y[:n])
        times[i, 1] = (time.time() - strt) / K[i]

.. image:: https://raw.github.com/dfm/ess/master/python/demo.png
