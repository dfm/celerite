# -*- coding: utf-8 -*-

from __future__ import division, print_function

import pytest
import numpy as np

try:
    import cPickle as pickle
except ImportError:
    import pickle

from celerite import Solver, GP, terms
from celerite.solver import get_kernel_value, CARMASolver

__all__ = ["test_carma", "test_log_determinant", "test_solve", "test_dot",
           "test_pickle", "test_build_gp", "test_log_likelihood",
           "test_predict", "test_nyquist_singularity"]


def test_carma(seed=42):
    np.random.seed(seed)
    t = np.sort(np.random.uniform(0, 5, 100))
    yerr = 0.1 + np.zeros_like(t)
    y = np.sin(t) + yerr * np.random.randn(len(t))

    carma_solver = CARMASolver(-0.5, np.array([0.1, 0.05, 0.01]),
                               np.array([0.2, 0.1]))
    carma_ll = carma_solver.log_likelihood(t, y, yerr)
    params = carma_solver.get_celerite_coeffs()
    print(params)

    solver = Solver()
    solver.compute(
        params[0], params[1], params[2], params[3], params[4], params[5],
        t, yerr**2
    )
    celerite_ll = -0.5*(
        solver.dot_solve(y) + solver.log_determinant() + len(t)*np.log(2*np.pi)
    )

    print(carma_ll, celerite_ll)
    assert np.allclose(carma_ll, celerite_ll)


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
    b = np.random.randn(len(t))

    alpha_real = np.array([1.3, 1.5])
    beta_real = np.array([0.5, 0.2])
    alpha_complex_real = np.array([1.0])
    alpha_complex_imag = np.array([0.1])
    beta_complex_real = np.array([1.0])
    beta_complex_imag = np.array([1.0])

    solver = Solver()
    with pytest.raises(RuntimeError):
        solver.log_determinant()
    with pytest.raises(RuntimeError):
        solver.dot_solve(b)

    solver.compute(
        alpha_real, beta_real, alpha_complex_real, alpha_complex_imag,
        beta_complex_real, beta_complex_imag, t, diag
    )
    K = get_kernel_value(
        alpha_real, beta_real, alpha_complex_real, alpha_complex_imag,
        beta_complex_real, beta_complex_imag, t[:, None] - t[None, :]
    )
    K[np.diag_indices_from(K)] += diag
    assert np.allclose(solver.solve(b).T, np.linalg.solve(K, b))

    b = np.random.randn(len(t), 5)
    assert np.allclose(solver.solve(b), np.linalg.solve(K, b))


def test_dot(seed=42):
    np.random.seed(seed)
    t = np.sort(np.random.rand(300))
    b = np.random.randn(len(t), 5)

    alpha_real = np.array([1.3, 0.2])
    beta_real = np.array([0.5, 0.8])
    alpha_complex_real = np.array([0.1])
    alpha_complex_imag = np.array([0.3])
    beta_complex_real = np.array([0.5])
    beta_complex_imag = np.array([3.0])

    K = get_kernel_value(
        alpha_real, beta_real, alpha_complex_real, alpha_complex_imag,
        beta_complex_real, beta_complex_imag, t[:, None] - t[None, :]
    )
    x0 = np.dot(K, b)

    solver = Solver()
    x = solver.dot(
        alpha_real, beta_real, alpha_complex_real, alpha_complex_imag,
        beta_complex_real, beta_complex_imag, t, b
    )
    assert np.allclose(x0, x)


def test_pickle(seed=42):
    np.random.seed(seed)
    t = np.sort(np.random.rand(500))
    diag = np.random.uniform(0.1, 0.5, len(t))
    y = np.sin(t)

    alpha_real = np.array([1.3, 1.5])
    beta_real = np.array([0.5, 0.2])
    alpha_complex_real = np.array([1.0])
    alpha_complex_imag = np.array([0.1])
    beta_complex_real = np.array([1.0])
    beta_complex_imag = np.array([1.0])

    def compare(solver1, solver2):
        assert solver1.computed() == solver2.computed()
        if not solver1.computed():
            return
        assert np.allclose(solver1.log_determinant(),
                           solver2.log_determinant())
        assert np.allclose(solver1.dot_solve(y),
                           solver2.dot_solve(y))

    solver1 = Solver()
    solver2 = pickle.loads(pickle.dumps(solver1, -1))
    compare(solver1, solver2)

    solver1.compute(
        alpha_real, beta_real, alpha_complex_real, alpha_complex_imag,
        beta_complex_real, beta_complex_imag, t, diag
    )
    solver2 = pickle.loads(pickle.dumps(solver1, -1))
    compare(solver1, solver2)

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


def test_build_gp(seed=42):
    kernel = terms.RealTerm(0.5, 0.1)
    kernel += terms.ComplexTerm(0.6, 0.7, 1.0)
    gp = GP(kernel)

    assert gp.vector_size == 5
    p = gp.get_parameter_vector()
    assert np.allclose(p, [0.5, 0.1, 0.6, 0.7, 1.0])

    gp.set_parameter_vector([0.5, 0.8, 0.6, 0.7, 2.0])
    p = gp.get_parameter_vector()
    assert np.allclose(p, [0.5, 0.8, 0.6, 0.7, 2.0])

    with pytest.raises(ValueError):
        gp.set_parameter_vector([0.5, 0.8, -0.6])

    with pytest.raises(ValueError):
        gp.set_parameter_vector("face1")

def test_log_likelihood(seed=42):
    np.random.seed(seed)
    x = np.sort(np.random.rand(10))
    yerr = np.random.uniform(0.1, 0.5, len(x))
    y = np.sin(x)

    kernel = terms.RealTerm(0.1, 0.5)
    gp = GP(kernel)
    with pytest.raises(RuntimeError):
        gp.log_likelihood(y)

    for term in [(0.6, 0.7, 1.0)]:
        kernel += terms.ComplexTerm(*term)
        gp = GP(kernel)

        assert gp.computed is False

        with pytest.raises(ValueError):
            gp.compute(np.random.rand(len(x)), yerr)

        gp.compute(x, yerr)
        assert gp.computed is True
        assert gp.dirty is False

        ll = gp.log_likelihood(y)
        K = gp.get_matrix(include_diagonal=True)
        ll0 = -0.5 * np.dot(y, np.linalg.solve(K, y))
        ll0 -= 0.5 * np.linalg.slogdet(K)[1]
        ll0 -= 0.5 * len(x) * np.log(2*np.pi)
        assert np.allclose(ll, ll0)

    # Check that changing the parameters "un-computes" the likelihood.
    gp.set_parameter_vector(gp.get_parameter_vector())
    assert gp.dirty is True
    assert gp.computed is False

    # Check that changing the parameters changes the likelihood.
    gp.compute(x, yerr)
    ll1 = gp.log_likelihood(y)
    params = gp.get_parameter_vector()
    params[0] += 0.1
    gp.set_parameter_vector(params)
    gp.compute(x, yerr)
    ll2 = gp.log_likelihood(y)
    assert not np.allclose(ll1, ll2)

    print(gp.get_parameter_dict())
    gp[1] += 0.1
    print(gp.get_parameter_dict())
    assert gp.dirty is True
    gp.compute(x, yerr)
    ll3 = gp.log_likelihood(y)
    assert not np.allclose(ll2, ll3)

def test_predict(seed=42):
    np.random.seed(seed)
    x = np.sort(np.random.rand(10))
    yerr = np.random.uniform(0.1, 0.5, len(x))
    y = np.sin(x)

    kernel = terms.RealTerm(0.1, 0.5)
    for term in [(0.6, 0.7, 1.0)]:
        kernel += terms.ComplexTerm(*term)
    gp = GP(kernel)
    gp.compute(x, yerr)

    mu0, cov0 = gp.predict(y, x)
    mu, cov = gp.predict(y)
    assert np.allclose(mu0, mu)
    assert np.allclose(cov0, cov)

# Test whether the GP can properly handle the case where the Lorentzian has a
# very large quality factor and the time samples are almost exactly at Nyquist
# sampling.  This can frustrate Green's-function-based CARMA solvers.
def test_nyquist_singularity(seed=4220):
    np.random.seed(seed)

    kernel = terms.ComplexTerm(1.0, np.log(1e-6), np.log(1.0))
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
