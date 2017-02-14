
.. module:: celerite

.. note:: This tutorial was generated from an IPython notebook that can be
          downloaded `here <../../_static/notebooks/normalization.ipynb>`_.

.. _normalization:


PSD Normalization
=================

The crux of many time series analysis problems is the question of where
all the factors of :math:`N` and :math:`2\,\pi` enter. In this tutorial,
we'll look at how the PSD returned by celerite should be compared to an
estimate made using NumPy's FFT library or to an estimate made using a
Lomb-Scargle periodogram. To make this comparison, we'll sample many
realizations from a celerite GP and compute the empirical power spectrum
using the standard methods and compare this (numerically) to the true
power spectrum as defined by celerite.

To start, here's a function that simulates :math:`K` random datasets
with :math:`N` data points from a celerite model and computes the mean
FFT and Lomb-Scargle estimators of the power spectrum.

.. code:: python

    import numpy as np
    import matplotlib.pyplot as plt
    from astropy.stats import LombScargle
    
    import celerite
    from celerite import terms
    
    def simulate_and_compute_psds(N, K=1500, norm=False):
        # Set up a simple celerite model
        kernel = terms.RealTerm(0.1, 0.5) + terms.ComplexTerm(0.5, 0.1, 2.0)
        gp = celerite.GP(kernel)
        
        # Simulate K datasets with N points
        t = np.linspace(0, 10, N)
        np.random.seed(42)
        y = gp.sample(t, size=K)
        
        # Compute the mean FFT based power spectrum estimate
        f = np.fft.rfftfreq(len(t), t[1] - t[0])
        fft = np.array(list(map(np.fft.rfft, y)))
        fft *= np.conj(fft)
        if norm:
            fft /= N**2
        q_fft = np.percentile(fft.real, [16, 50, 84], axis=0)
    
        # Compute the mean LS based power spectrum estimate
        power_ls = []
        for y0 in y:
            model = LombScargle(t, y0)
            power_ls.append(model.power(f[1:-1], method="fast", normalization="psd"))
        power_ls = np.array(power_ls)
        if norm:
            power_ls /= N
        q_ls = np.percentile(power_ls, [16, 50, 84], axis=0)
        
        # Compute the true power spectrum
        # NOTE: the 2*pi enters because celerite computes the PSD in _angular_ frequency
        power_true = kernel.get_psd(2*np.pi*f)
        if norm:
            power_true /= 2*np.pi
        
        # Let's plot the estimates of the PSD
        plt.plot(f, power_true, label="truth")
        plt.plot(f, q_fft[1], "--", label="FFT")
        plt.plot(f[1:-1], q_ls[1], ":", label="LS")
        plt.yscale("log")
        plt.xscale("log")
        plt.xlim(f.min(), f.max())
        plt.ylabel("power")
        plt.xlabel("frequency")
        plt.legend()
        
    simulate_and_compute_psds(1000, norm=True)



.. image:: normalization_files/normalization_2_0.png


