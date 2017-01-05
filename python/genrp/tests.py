# -*- coding: utf-8 -*-

from __future__ import division, print_function

import pytest
import numpy as np

from . import Solver, GP, kernels
from ._genrp import get_kernel_value

__all__ = ["test_invalid_parameters", "test_log_determinant", "test_solve",
           "test_nyquist_singularity"]


@pytest.mark.skip(reason="solver no longer checks for ordering")
def test_invalid_parameters(seed=42):
    np.random.seed(seed)
    t = np.random.rand(50)

    alpha_real = np.array([1.0, 2.0])
    beta_real = np.array([1.0, 0.5])
    alpha_complex_real = np.array([1.0])
    alpha_complex_imag = np.array([0.0])
    beta_complex_real = np.array([1.0])
    beta_complex_imag = np.array([1.0])
    with pytest.raises(ValueError):
        Solver(alpha_real, beta_real, alpha_complex_real, alpha_complex_imag,
               beta_complex_real, beta_complex_imag, t)
    t = np.sort(t)
    Solver(alpha_real, beta_real, alpha_complex_real, alpha_complex_imag,
           beta_complex_real, beta_complex_imag, t)


def test_log_determinant(seed=42):
    np.random.seed(seed)
    t = np.sort(np.random.rand(5))
    diag = np.random.uniform(0.1, 0.5, len(t))

    alpha_real = np.array([1.5, 0.1])
    beta_real = np.array([1.0, 0.3])
    alpha_complex_real = np.array([1.0])
    alpha_complex_imag = np.array([0.1])
    beta_complex_real = np.array([1.0])
    beta_complex_imag = np.array([1.0])

    solver = Solver()
    solver.compute(
        alpha_real, beta_real, alpha_complex_real, alpha_complex_imag,
        beta_complex_real, beta_complex_imag, t, diag
    )
    K = get_kernel_value(
        alpha_real, beta_real, alpha_complex_real, alpha_complex_imag,
        beta_complex_real, beta_complex_imag, t[:, None] - t[None, :]
    )
    K[np.diag_indices_from(K)] += diag
    assert np.allclose(solver.log_determinant(), np.linalg.slogdet(K)[1])


def test_solve(seed=42):
    np.random.seed(seed)
    t = np.sort(np.random.rand(500))
    diag = np.random.uniform(0.1, 0.5, len(t))

    alpha_real = np.array([1.3, 1.5])
    beta_real = np.array([0.5, 0.2])
    alpha_complex_real = np.array([1.0])
    alpha_complex_imag = np.array([0.1])
    beta_complex_real = np.array([1.0])
    beta_complex_imag = np.array([1.0])

    solver = Solver()
    solver.compute(
        alpha_real, beta_real, alpha_complex_real, alpha_complex_imag,
        beta_complex_real, beta_complex_imag, t, diag
    )
    K = get_kernel_value(
        alpha_real, beta_real, alpha_complex_real, alpha_complex_imag,
        beta_complex_real, beta_complex_imag, t[:, None] - t[None, :]
    )
    K[np.diag_indices_from(K)] += diag
    b = np.random.randn(len(t))
    assert np.allclose(solver.solve(b).T, np.linalg.solve(K, b))

    b = np.random.randn(len(t), 5)
    print(b)
    print(solver.solve(b))
    assert np.allclose(solver.solve(b), np.linalg.solve(K, b))

#  def test_kernel_params():
#      gp = GP()
#      terms = [(-0.5, 0.1), (-0.6, 0.7, 1.0)]
#      for term in terms:
#          gp.add_term(*term)

#      alpha_real = []
#      beta_real = []
#      alpha_complex = []
#      beta_complex_real = []
#      beta_complex_imag = []
#      for term in terms:
#          if len(term) == 2:
#              a, q = np.exp(term)
#              alpha_real.append(4.0 * np.pi**2 * a * q)
#              beta_real.append(2.0 * np.pi * q)
#              continue
#          a, q, f = np.exp(term)
#          alpha_complex.append(4.0 * np.pi**2 * a * q)
#          beta_complex_real.append(2.0 * np.pi * q)
#          beta_complex_imag.append(2.0 * np.pi * f)

#      assert np.allclose(gp.alpha_real, alpha_real)
#      assert np.allclose(gp.beta_real, beta_real)
#      assert np.allclose(gp.alpha_complex, alpha_complex)
#      assert np.allclose(gp.beta_complex_real, beta_complex_real)
#      assert np.allclose(gp.beta_complex_imag, beta_complex_imag)


#  def test_kernel_value(seed=42):
#      gp = GP()
#      terms = [(-0.3, 0.1), (-0.5, 0.1), (-0.7, 0.1),
#               (-0.6, 0.7, 1.0), (-0.8, 0.6, 0.1)]
#      for term in terms:
#          gp.add_term(*term)

#      np.random.seed(seed)
#      x1 = np.sort(np.random.rand(1000))
#      K0 = gp.get_matrix(x1)
#      dt = np.abs(x1[:, None] - x1[None, :])
#      K = np.zeros_like(K0)
#      for term in terms:
#          if len(term) == 2:
#              amp, q = np.exp(term[:2])
#              K += 4*amp*np.pi**2*q*np.exp(-2*np.pi*q*dt)
#              continue
#          amp, q, f = np.exp(term)
#          K += 4*amp*np.pi**2*q*np.exp(-2*np.pi*q*dt)*np.cos(2*np.pi*f*dt)

#      assert np.allclose(K, K0)

#      x2 = np.sort(np.random.rand(5))
#      K0 = gp.get_matrix(x1, x2)
#      dt = np.abs(x1[:, None] - x2[None, :])
#      K = np.zeros_like(K0)
#      for term in terms:
#          if len(term) == 2:
#              amp, q = np.exp(term[:2])
#              K += 4*amp*np.pi**2*q*np.exp(-2*np.pi*q*dt)
#              continue
#          amp, q, f = np.exp(term)
#          K += 4*amp*np.pi**2*q*np.exp(-2*np.pi*q*dt)*np.cos(2*np.pi*f*dt)
#      assert np.allclose(K, K0)


#  def test_build_gp(seed=42):
#      gp = GP()
#      terms = [(-0.5, 0.1), (-0.6, 0.7, 1.0)]
#      for term in terms:
#          gp.add_term(*term)

#      # assert len(gp.terms) == len(terms)
#      # assert all(np.allclose(t1, t2) for t1, t2 in zip(gp.terms, terms))
#      assert len(gp) == 5
#      assert np.allclose(gp.params, [-0.5, 0.1, -0.6, 0.7, 1.0])

#      gp.params = [0.5, 0.8, -0.6, 0.7, 2.0]
#      assert np.allclose(gp.params, [0.5, 0.8, -0.6, 0.7, 2.0])

#      with pytest.raises(ValueError):
#          gp.params = [0.5, 0.8, -0.6]

#      with pytest.raises(ValueError):
#          gp.params = "face"

#  def test_log_likelihood(seed=42):
#      np.random.seed(seed)
#      x = np.sort(np.random.rand(10))
#      yerr = np.random.uniform(0.1, 0.5, len(x))
#      y = np.sin(x)

#      gp = GP()
#      with pytest.raises(RuntimeError):
#          gp.log_likelihood(y)
#      for term in [(-0.5, 0.1), (-0.6, 0.7, 1.0)]:
#          gp.add_term(*term)

#          assert gp.computed is False

#          with pytest.raises(ValueError):
#              gp.compute(np.random.rand(len(x)), yerr)

#          gp.compute(x, yerr)
#          assert gp.computed is True

#          ll = gp.log_likelihood(y)
#          K = gp.get_matrix(x)
#          K[np.diag_indices_from(K)] += yerr**2
#          ll0 = -0.5 * np.dot(y, np.linalg.solve(K, y))
#          ll0 -= 0.5 * np.linalg.slogdet(K)[1]
#          ll0 -= 0.5 * len(x) * np.log(2*np.pi)
#          assert np.allclose(ll, ll0)

#      # Check that changing the parameters "un-computes" the likelihood.
#      gp.params = gp.params
#      with pytest.raises(RuntimeError):
#          gp.log_likelihood(y)

#      # Check that changing the parameters changes the likelihood.
#      gp.compute(x, yerr)
#      ll1 = gp.log_likelihood(y)
#      params = gp.params
#      params[0] += 0.1
#      gp.params = params
#      gp.compute(x, yerr)
#      ll2 = gp.log_likelihood(y)
#      assert not np.allclose(ll1, ll2)

#      gp[1] += 0.1
#      gp.compute(x, yerr)
#      ll3 = gp.log_likelihood(y)
#      assert not np.allclose(ll2, ll3)


#  def test_psd():
#      gp = GP()
#      terms = [(-0.5, 0.1), (-0.6, 0.7, 1.0)]
#      for term in terms:
#          gp.add_term(*term)

#      freqs = np.exp(np.linspace(np.log(0.1), np.log(10), 1000))
#      psd0 = np.zeros_like(freqs)
#      for term in terms:
#          if len(term) == 2:
#              amp, q = np.exp(term)
#              psd0 += 2.0 * amp / (1 + (freqs/q)**2)
#              continue
#          amp, q, f = np.exp(term)
#          psd0 += amp / (1 + ((freqs - f)/q)**2)
#          psd0 += amp / (1 + ((freqs + f)/q)**2)

#      assert np.allclose(psd0, gp.get_psd(freqs))

# Test whether the GP can properly handle the case where the Lorentzian has a
# very large quality factor and the time samples are almost exactly at Nyquist
# sampling.  This can frustrate Green's-function-based CARMA solvers.
def test_nyquist_singularity(seed=4220):
    np.random.seed(seed)

    kernel = kernels.ComplexTerm(1.0, np.log(1e-6), np.log(1.0))
    gp = GP(kernel)

    # Samples are very close to Nyquist with f = 1.0
    ts = np.array([0.0, 0.5, 1.0, 1.5])
    ts[1] = ts[1]+1e-9*np.random.randn()
    ts[2] = ts[2]+1e-8*np.random.randn()
    ts[3] = ts[3]+1e-7*np.random.randn()

    yerr = np.random.uniform(low=0.1, high=0.2, size=len(ts))
    y = np.random.randn(len(ts))

    gp.compute(ts, yerr)
    llgp = gp.log_likelihood(y)

    K = gp.get_matrix(ts)
    K[np.diag_indices_from(K)] += yerr**2.0

    ll = (-0.5*np.dot(y, np.linalg.solve(K, y)) - 0.5*np.linalg.slogdet(K)[1] -
          0.5*len(y)*np.log(2.0*np.pi))

    assert np.allclose(ll, llgp)
