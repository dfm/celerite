# -*- coding: utf-8 -*-

from __future__ import division, print_function

import pytest
import numpy as np
from itertools import product

try:
    import cPickle as pickle
except ImportError:
    import pickle

import celerite
from celerite import get_solver, GP, terms
from celerite.solver import get_kernel_value, CARMASolver, SingleSolver

__all__ = ["test_carma", "test_log_determinant", "test_solve", "test_dot",
           "test_pickle", "test_build_gp", "test_log_likelihood",
           "test_predict", "test_nyquist_singularity"]


method_switch = pytest.mark.parametrize(
    "method",
    [
        "simple",
        "cholesky",
        pytest.mark.skipif(celerite.__with_lapack__ is False,
                           "lapack",
                           reason="LAPACK support not enabled"),
        pytest.mark.skipif(celerite.__with_sparse__ is False,
                           "sparse",
                           reason="Sparse support not sparse"),
    ]
)


def test_kernel(seed=42):
    np.random.seed(seed)
    t = np.sort(np.random.uniform(0, 5, 100))
    tau = t[:, None] - t[None, :]

    k1 = terms.RealTerm(log_a=0.1, log_c=0.5)
    k2 = terms.ComplexTerm(0.2, -3.0, 0.5, 0.01)
    k3 = terms.SHOTerm(1.0, 0.2, 3.0)

    K1 = k1.get_value(tau)
    K2 = k2.get_value(tau)
    K3 = k3.get_value(tau)

    assert np.allclose((k1 + k2).get_value(tau), K1 + K2)
    assert np.allclose((k3 + k2).get_value(tau), K3 + K2)
    assert np.allclose((k1 + k2 + k3).get_value(tau), K1 + K2 + K3)

    for (a, b), (A, B) in zip(product((k1, k2, k3, k1+k2, k1+k3, k2+k3),
                                      (k1, k2, k3)),
                              product((K1, K2, K3, K1+K2, K1+K3, K2+K3),
                                      (K1, K2, K3))):
        assert np.allclose((a * b).get_value(tau), A*B)

@method_switch
def test_carma(method, seed=42):
    solver = get_solver(method)
    np.random.seed(seed)
    t = np.sort(np.random.uniform(0, 5, 100))
    yerr = 0.1 + np.zeros_like(t)
    y = np.sin(t) + yerr * np.random.randn(len(t))

    carma_solver = CARMASolver(-0.5, np.array([0.1, 0.05, 0.01]),
                               np.array([0.2, 0.1]))
    carma_ll = carma_solver.log_likelihood(t, y, yerr)
    params = carma_solver.get_celerite_coeffs()

    solver.compute(
        params[0], params[1], params[2], params[3], params[4], params[5],
        t, yerr**2
    )
    celerite_ll = -0.5*(
        solver.dot_solve(y) + solver.log_determinant() + len(t)*np.log(2*np.pi)
    )
    assert np.allclose(carma_ll, celerite_ll)

def test_single(seed=42):
    if not celerite.__with_lapack__:
        with pytest.raises(RuntimeError):
            solver = SingleSolver()
        return

    alpha_real = np.array([1.5])
    beta_real = np.array([0.3])
    alpha_complex_real = np.array([])
    alpha_complex_imag = np.array([])
    beta_complex_real = np.array([])
    beta_complex_imag = np.array([])

    solver = SingleSolver()
    np.random.seed(seed)
    t = np.sort(np.random.uniform(0, 10, 5))
    diag = np.random.uniform(0.1, 0.8, len(t))

    flag = solver.compute(
        alpha_real, beta_real, alpha_complex_real, alpha_complex_imag,
        beta_complex_real, beta_complex_imag, t, diag
    )
    assert flag == 0
    K0 = get_kernel_value(
        alpha_real, beta_real, alpha_complex_real, alpha_complex_imag,
        beta_complex_real, beta_complex_imag, t[:, None] - t[None, :]
    )
    K = np.array(K0)
    K[np.diag_indices_from(K)] += diag
    assert np.allclose(solver.log_determinant(), np.linalg.slogdet(K)[1])

    b = np.random.randn(len(t), 2)
    assert np.allclose(solver.solve(b), np.linalg.solve(K, b))

    d1 = solver.dot(
        alpha_real, beta_real, alpha_complex_real, alpha_complex_imag,
        beta_complex_real, beta_complex_imag, t, b
    )
    d2 = np.dot(K0, b)
    assert np.allclose(d1, d2)

def _test_log_determinant(alpha_real, beta_real, alpha_complex_real,
                          alpha_complex_imag, beta_complex_real,
                          beta_complex_imag, method, seed=42):
    solver = get_solver(method)
    np.random.seed(seed)
    t = np.sort(np.random.rand(5))
    diag = np.random.uniform(0.1, 0.5, len(t))

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

@method_switch
def test_log_determinant(method, seed=42):
    alpha_real = np.array([1.5, 0.1])
    beta_real = np.array([1.0, 0.3])
    alpha_complex_real = np.array([1.0])
    alpha_complex_imag = np.array([0.1])
    beta_complex_real = np.array([1.0])
    beta_complex_imag = np.array([1.0])
    _test_log_determinant(alpha_real, beta_real, alpha_complex_real,
                          alpha_complex_imag, beta_complex_real,
                          beta_complex_imag, method, seed=seed)

    alpha_real = np.array([1.5, 0.1, 0.6, 0.3, 0.8, 0.7])
    beta_real = np.array([1.0, 0.3, 0.05, 0.01, 0.1, 0.2])
    alpha_complex_real = np.array([1.0, 2.0])
    alpha_complex_imag = np.array([0.1, 0.5])
    beta_complex_real = np.array([1.0, 1.0])
    beta_complex_imag = np.array([1.0, 1.0])
    _test_log_determinant(alpha_real, beta_real, alpha_complex_real,
                          alpha_complex_imag, beta_complex_real,
                          beta_complex_imag, method, seed=seed)


def _test_solve(alpha_real, beta_real, alpha_complex_real, alpha_complex_imag,
                beta_complex_real, beta_complex_imag, method, seed=42):
    solver = get_solver(method)
    np.random.seed(seed)
    t = np.sort(np.random.rand(500))
    diag = np.random.uniform(0.1, 0.5, len(t))
    b = np.random.randn(len(t))

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

@method_switch
def test_solve(method, seed=42):
    alpha_real = np.array([1.5, 0.1])
    beta_real = np.array([1.0, 0.3])
    alpha_complex_real = np.array([1.0])
    alpha_complex_imag = np.array([0.1])
    beta_complex_real = np.array([1.0])
    beta_complex_imag = np.array([1.0])
    _test_solve(alpha_real, beta_real, alpha_complex_real, alpha_complex_imag,
                beta_complex_real, beta_complex_imag, method, seed=seed)

    alpha_real = np.array([1.5, 0.1, 0.6, 0.3, 0.8, 0.7])
    beta_real = np.array([1.0, 0.3, 0.05, 0.01, 0.1, 0.2])
    alpha_complex_real = np.array([1.0, 2.0])
    alpha_complex_imag = np.array([0.1, 0.5])
    beta_complex_real = np.array([1.0, 1.0])
    beta_complex_imag = np.array([1.0, 1.0])
    _test_solve(alpha_real, beta_real, alpha_complex_real, alpha_complex_imag,
                beta_complex_real, beta_complex_imag, method, seed=seed)


@method_switch
def test_dot(method, seed=42):
    solver = get_solver(method)
    np.random.seed(seed)
    t = np.sort(np.random.rand(500))
    b = np.random.randn(len(t), 5)

    alpha_real = np.array([1.3, 0.2])
    beta_real = np.array([0.5, 0.8])
    alpha_complex_real = np.array([0.1])
    alpha_complex_imag = np.array([0.0])
    beta_complex_real = np.array([1.5])
    beta_complex_imag = np.array([0.1])

    K = get_kernel_value(
        alpha_real, beta_real, alpha_complex_real, alpha_complex_imag,
        beta_complex_real, beta_complex_imag, t[:, None] - t[None, :]
    )
    x0 = np.dot(K, b)

    x = solver.dot(
        alpha_real, beta_real, alpha_complex_real, alpha_complex_imag,
        beta_complex_real, beta_complex_imag, t, b
    )
    assert np.allclose(x0, x)

def test_dot_L(method="cholesky", seed=42):
    solver = get_solver(method)
    np.random.seed(seed)
    t = np.sort(np.random.rand(5))
    b = np.random.randn(len(t), 5)
    yerr = np.random.uniform(0.1, 0.5, len(t))

    alpha_real = np.array([1.3, 0.2])
    beta_real = np.array([0.5, 0.8])
    alpha_complex_real = np.array([0.1])
    alpha_complex_imag = np.array([0.0])
    beta_complex_real = np.array([1.5])
    beta_complex_imag = np.array([0.1])

    K = get_kernel_value(
        alpha_real, beta_real, alpha_complex_real, alpha_complex_imag,
        beta_complex_real, beta_complex_imag, t[:, None] - t[None, :]
    )
    K[np.diag_indices_from(K)] += yerr**2
    L = np.linalg.cholesky(K)
    x0 = np.dot(L, b)

    solver.compute(
        alpha_real, beta_real, alpha_complex_real, alpha_complex_imag,
        beta_complex_real, beta_complex_imag, t, yerr**2)
    x = solver.dot_L(b)
    print(x0 - x)
    assert np.allclose(x0, x)

@method_switch
def test_pickle(method, seed=42):
    solver = get_solver(method)
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

    s = pickle.dumps(solver, -1)
    solver2 = pickle.loads(s)
    compare(solver, solver2)

    if method != "sparse":
        solver.compute(
            alpha_real, beta_real, alpha_complex_real, alpha_complex_imag,
            beta_complex_real, beta_complex_imag, t, diag
        )
        solver2 = pickle.loads(pickle.dumps(solver, -1))
        compare(solver, solver2)

    # Test that models can be pickled too.
    kernel = terms.RealTerm(0.5, 0.1)
    kernel += terms.ComplexTerm(0.6, 0.7, 1.0)
    gp1 = GP(kernel, method=method)
    gp1.compute(t, diag)
    s = pickle.dumps(gp1, -1)
    gp2 = pickle.loads(s)
    assert np.allclose(gp1.log_likelihood(y), gp2.log_likelihood(y))


@method_switch
def test_build_gp(method, seed=42):
    kernel = terms.RealTerm(0.5, 0.1)
    kernel += terms.ComplexTerm(0.6, 0.7, 1.0)
    gp = GP(kernel, method=method)

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


@method_switch
def test_log_likelihood(method, seed=42):
    np.random.seed(seed)
    x = np.sort(np.random.rand(10))
    yerr = np.random.uniform(0.1, 0.5, len(x))
    y = np.sin(x)

    kernel = terms.RealTerm(0.1, 0.5)
    gp = GP(kernel, method=method)
    with pytest.raises(RuntimeError):
        gp.log_likelihood(y)

    for term in [(0.6, 0.7, 1.0)]:
        kernel += terms.ComplexTerm(*term)
        gp = GP(kernel, method=method)

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

    gp[1] += 0.1
    assert gp.dirty is True
    gp.compute(x, yerr)
    ll3 = gp.log_likelihood(y)
    assert not np.allclose(ll2, ll3)

    # Test zero delta t
    ind = len(x) // 2
    x = np.concatenate((x[:ind], [x[ind]], x[ind:]))
    y = np.concatenate((y[:ind], [y[ind]], y[ind:]))
    yerr = np.concatenate((yerr[:ind], [yerr[ind]], yerr[ind:]))
    gp.compute(x, yerr)
    ll = gp.log_likelihood(y)
    K = gp.get_matrix(include_diagonal=True)
    ll0 = -0.5 * np.dot(y, np.linalg.solve(K, y))
    ll0 -= 0.5 * np.linalg.slogdet(K)[1]
    ll0 -= 0.5 * len(x) * np.log(2*np.pi)
    assert np.allclose(ll, ll0)


@method_switch
def test_predict(method, seed=42):
    np.random.seed(seed)
    x = np.sort(np.random.rand(10))
    yerr = np.random.uniform(0.1, 0.5, len(x))
    y = np.sin(x)

    kernel = terms.RealTerm(0.1, 0.5)
    for term in [(0.6, 0.7, 1.0)]:
        kernel += terms.ComplexTerm(*term)
    gp = GP(kernel, method=method)
    gp.compute(x, yerr)

    mu0, cov0 = gp.predict(y, x)
    mu, cov = gp.predict(y)
    assert np.allclose(mu0, mu)
    assert np.allclose(cov0, cov)

# Test whether the GP can properly handle the case where the Lorentzian has a
# very large quality factor and the time samples are almost exactly at Nyquist
# sampling.  This can frustrate Green's-function-based CARMA solvers.
@method_switch
def test_nyquist_singularity(method, seed=4220):
    np.random.seed(seed)

    kernel = terms.ComplexTerm(1.0, np.log(1e-6), np.log(1.0))
    gp = GP(kernel, method=method)

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
