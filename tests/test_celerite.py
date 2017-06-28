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
from celerite import GP, terms
from celerite.solver import get_kernel_value, CARMASolver

__all__ = ["test_carma", "test_log_determinant", "test_solve", "test_dot",
           "test_pickle", "test_build_gp", "test_log_likelihood",
           "test_predict", "test_nyquist_singularity"]

def test_carma(seed=42):
    solver = celerite.CholeskySolver()
    np.random.seed(seed)
    t = np.sort(np.random.uniform(0, 5, 100))
    yerr = 0.1 + np.zeros_like(t)
    y = np.sin(t) + yerr * np.random.randn(len(t))

    carma_solver = CARMASolver(-0.5, np.array([0.1, 0.05, 0.01]),
                               np.array([0.2, 0.1]))
    carma_ll = carma_solver.log_likelihood(t, y, yerr)
    params = carma_solver.get_celerite_coeffs()

    solver.compute(
        0.0, params[0], params[1], params[2], params[3], params[4], params[5],
        np.empty(0), np.empty((0, 0)), np.empty((0, 0)),
        t, yerr**2
    )
    celerite_ll = -0.5*(
        solver.dot_solve(y) + solver.log_determinant() + len(t)*np.log(2*np.pi)
    )
    assert np.allclose(carma_ll, celerite_ll)


def _test_log_determinant(alpha_real, beta_real, alpha_complex_real,
                          alpha_complex_imag, beta_complex_real,
                          beta_complex_imag, seed=42):
    solver = celerite.CholeskySolver()
    np.random.seed(seed)
    t = np.sort(np.random.rand(5))
    diag = np.random.uniform(0.1, 0.5, len(t))

    solver.compute(
        0.0, alpha_real, beta_real, alpha_complex_real, alpha_complex_imag,
        beta_complex_real, beta_complex_imag,
        np.empty(0), np.empty((0, 0)), np.empty((0, 0)),
        t, diag
    )
    K = get_kernel_value(
        alpha_real, beta_real, alpha_complex_real, alpha_complex_imag,
        beta_complex_real, beta_complex_imag, t[:, None] - t[None, :]
    )
    K[np.diag_indices_from(K)] += diag
    assert np.allclose(solver.log_determinant(), np.linalg.slogdet(K)[1])

def test_log_determinant(seed=42):
    alpha_real = np.array([1.5, 0.1])
    beta_real = np.array([1.0, 0.3])
    alpha_complex_real = np.array([1.0])
    alpha_complex_imag = np.array([0.1])
    beta_complex_real = np.array([1.0])
    beta_complex_imag = np.array([1.0])
    _test_log_determinant(alpha_real, beta_real, alpha_complex_real,
                          alpha_complex_imag, beta_complex_real,
                          beta_complex_imag, seed=seed)

    alpha_real = np.array([1.5, 0.1, 0.6, 0.3, 0.8, 0.7])
    beta_real = np.array([1.0, 0.3, 0.05, 0.01, 0.1, 0.2])
    alpha_complex_real = np.array([1.0, 2.0])
    alpha_complex_imag = np.array([0.1, 0.5])
    beta_complex_real = np.array([1.0, 1.0])
    beta_complex_imag = np.array([1.0, 1.0])
    _test_log_determinant(alpha_real, beta_real, alpha_complex_real,
                          alpha_complex_imag, beta_complex_real,
                          beta_complex_imag, seed=seed)


def _test_solve(alpha_real, beta_real, alpha_complex_real, alpha_complex_imag,
                beta_complex_real, beta_complex_imag, seed=42,
                with_general=False):
    solver = celerite.CholeskySolver()
    np.random.seed(seed)
    t = np.sort(np.random.rand(500))
    diag = np.random.uniform(0.1, 0.5, len(t))
    b = np.random.randn(len(t))

    with pytest.raises(RuntimeError):
        solver.log_determinant()
    with pytest.raises(RuntimeError):
        solver.dot_solve(b)

    if with_general:
        U = np.vander(t - np.mean(t), 4).T
        V = U * np.random.rand(4)[:, None]
        A = np.sum(U * V, axis=0) + 1e-8
    else:
        A = np.empty(0)
        U = np.empty((0, 0))
        V = np.empty((0, 0))

    solver.compute(
        0.0, alpha_real, beta_real, alpha_complex_real, alpha_complex_imag,
        beta_complex_real, beta_complex_imag,
        A, U, V, t, diag
    )
    K = get_kernel_value(
        alpha_real, beta_real, alpha_complex_real, alpha_complex_imag,
        beta_complex_real, beta_complex_imag, t[:, None] - t[None, :]
    )
    K[np.diag_indices_from(K)] += diag

    if len(A):
        K[np.diag_indices_from(K)] += A
        K += np.tril(np.dot(U.T, V), -1) + np.triu(np.dot(V.T, U), 1)

    assert np.allclose(solver.solve(b).T, np.linalg.solve(K, b))

    b = np.random.randn(len(t), 5)
    assert np.allclose(solver.solve(b), np.linalg.solve(K, b))

@pytest.mark.parametrize("with_general", [True, False])
def test_solve(with_general, seed=42):
    alpha_real = np.array([1.5, 0.1])
    beta_real = np.array([1.0, 0.3])
    alpha_complex_real = np.array([1.0])
    alpha_complex_imag = np.array([0.1])
    beta_complex_real = np.array([1.0])
    beta_complex_imag = np.array([1.0])
    _test_solve(alpha_real, beta_real, alpha_complex_real, alpha_complex_imag,
                beta_complex_real, beta_complex_imag, seed=seed,
                with_general=with_general)

    alpha_real = np.array([1.5, 0.1, 0.6, 0.3, 0.8, 0.7])
    beta_real = np.array([1.0, 0.3, 0.05, 0.01, 0.1, 0.2])
    alpha_complex_real = np.array([1.0, 2.0])
    alpha_complex_imag = np.array([0.1, 0.5])
    beta_complex_real = np.array([1.0, 1.0])
    beta_complex_imag = np.array([1.0, 1.0])
    _test_solve(alpha_real, beta_real, alpha_complex_real, alpha_complex_imag,
                beta_complex_real, beta_complex_imag, seed=seed,
                with_general=with_general)


@pytest.mark.parametrize("with_general", [True, False])
def test_dot(with_general, seed=42):
    solver = celerite.CholeskySolver()
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

    if with_general:
        U = np.vander(t - np.mean(t), 4).T
        V = U * np.random.rand(4)[:, None]
        A = np.sum(U * V, axis=0) + 1e-8

        K[np.diag_indices_from(K)] += A
        K += np.tril(np.dot(U.T, V), -1) + np.triu(np.dot(V.T, U), 1)
    else:
        A = np.empty(0)
        U = np.empty((0, 0))
        V = np.empty((0, 0))

    x0 = np.dot(K, b)

    x = solver.dot(
        0.0, alpha_real, beta_real, alpha_complex_real, alpha_complex_imag,
        beta_complex_real, beta_complex_imag,
        A, U, V, t, b
    )
    assert np.allclose(x0, x)

@pytest.mark.parametrize("with_general", [True, False])
def test_dot_L(with_general, seed=42):
    solver = celerite.CholeskySolver()
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

    if with_general:
        U = np.vander(t - np.mean(t), 4).T
        V = U * np.random.rand(4)[:, None]
        A = np.sum(U * V, axis=0) + 1e-8

        K[np.diag_indices_from(K)] += A
        K += np.tril(np.dot(U.T, V), -1) + np.triu(np.dot(V.T, U), 1)
    else:
        A = np.empty(0)
        U = np.empty((0, 0))
        V = np.empty((0, 0))

    L = np.linalg.cholesky(K)
    x0 = np.dot(L, b)

    solver.compute(
        0.0, alpha_real, beta_real, alpha_complex_real, alpha_complex_imag,
        beta_complex_real, beta_complex_imag,
        A, U, V, t, yerr**2)
    x = solver.dot_L(b)
    assert np.allclose(x0, x)

@pytest.mark.parametrize("with_general", [True, False])
def test_pickle(with_general, seed=42):
    solver = celerite.CholeskySolver()
    np.random.seed(seed)
    t = np.sort(np.random.rand(500))
    diag = np.random.uniform(0.1, 0.5, len(t))
    y = np.sin(t)

    if with_general:
        U = np.vander(t - np.mean(t), 4).T
        V = U * np.random.rand(4)[:, None]
        A = np.sum(U * V, axis=0) + 1e-8
    else:
        A = np.empty(0)
        U = np.empty((0, 0))
        V = np.empty((0, 0))

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

    solver.compute(
        0.0, alpha_real, beta_real, alpha_complex_real, alpha_complex_imag,
        beta_complex_real, beta_complex_imag,
        A, U, V, t, diag
    )
    solver2 = pickle.loads(pickle.dumps(solver, -1))
    compare(solver, solver2)

    # Test that models can be pickled too.
    kernel = terms.RealTerm(0.5, 0.1)
    kernel += terms.ComplexTerm(0.6, 0.7, 1.0)
    gp1 = GP(kernel)
    gp1.compute(t, diag)
    s = pickle.dumps(gp1, -1)
    gp2 = pickle.loads(s)
    assert np.allclose(gp1.log_likelihood(y), gp2.log_likelihood(y))


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

@pytest.mark.parametrize("with_general", [True, False])
def test_log_likelihood(with_general, seed=42):
    np.random.seed(seed)
    x = np.sort(np.random.rand(10))
    yerr = np.random.uniform(0.1, 0.5, len(x))
    y = np.sin(x)

    if with_general:
        U = np.vander(x - np.mean(x), 4).T
        V = U * np.random.rand(4)[:, None]
        A = np.sum(U * V, axis=0) + 1e-8
    else:
        A = np.empty(0)
        U = np.empty((0, 0))
        V = np.empty((0, 0))

    # Check quiet argument with a non-positive definite kernel.
    class NPDTerm(terms.Term):
        parameter_names = ("par1", )
        def get_real_coefficients(self, params):  # NOQA
            return [params[0]], [0.1]
    gp = GP(NPDTerm(-1.0))
    with pytest.raises(celerite.solver.LinAlgError):
        gp.compute(x, 0.0)
    with pytest.raises(celerite.solver.LinAlgError):
        gp.log_likelihood(y)
    assert np.isinf(gp.log_likelihood(y, quiet=True))
    if terms.HAS_AUTOGRAD:
        assert np.isinf(gp.grad_log_likelihood(y, quiet=True)[0])

    kernel = terms.RealTerm(0.1, 0.5)
    gp = GP(kernel)
    with pytest.raises(RuntimeError):
        gp.log_likelihood(y)

    termlist = [(0.1 + 10./j, 0.5 + 10./j) for j in range(1, 4)]
    termlist += [(1.0 + 10./j, 0.01 + 10./j, 0.5, 0.01) for j in range(1, 10)]
    termlist += [(0.6, 0.7, 1.0), (0.3, 0.05, 0.5, 0.6)]
    for term in termlist:
        if len(term) > 2:
            kernel += terms.ComplexTerm(*term)
        else:
            kernel += terms.RealTerm(*term)
        gp = GP(kernel)

        assert gp.computed is False

        with pytest.raises(ValueError):
            gp.compute(np.random.rand(len(x)), yerr)

        gp.compute(x, yerr, A=A, U=U, V=V)
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
    gp.compute(x, yerr, A=A, U=U, V=V)
    ll1 = gp.log_likelihood(y)
    params = gp.get_parameter_vector()
    params[0] += 10.0
    gp.set_parameter_vector(params)
    gp.compute(x, yerr, A=A, U=U, V=V)
    ll2 = gp.log_likelihood(y)
    assert not np.allclose(ll1, ll2)

    gp[1] += 10.0
    assert gp.dirty is True
    gp.compute(x, yerr, A=A, U=U, V=V)
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


@pytest.mark.parametrize(
    "kernel,with_general",
    product([
        terms.RealTerm(log_a=0.1, log_c=0.5),
        terms.RealTerm(log_a=0.1, log_c=0.5) +
        terms.RealTerm(log_a=-0.1, log_c=0.7),
        terms.ComplexTerm(log_a=0.1, log_c=0.5, log_d=0.1),
        terms.ComplexTerm(log_a=0.1, log_b=-0.2, log_c=0.5, log_d=0.1),
        terms.JitterTerm(log_sigma=0.1),
        terms.SHOTerm(log_S0=0.1, log_Q=-1, log_omega0=0.5) +
        terms.JitterTerm(log_sigma=0.1),
        terms.SHOTerm(log_S0=0.1, log_Q=-1, log_omega0=0.5),
        terms.SHOTerm(log_S0=0.1, log_Q=1.0, log_omega0=0.5),
        terms.SHOTerm(log_S0=0.1, log_Q=1.0, log_omega0=0.5) +
        terms.RealTerm(log_a=0.1, log_c=0.4),
        terms.SHOTerm(log_S0=0.1, log_Q=1.0, log_omega0=0.5) *
        terms.RealTerm(log_a=0.1, log_c=0.4),
    ], [False, True])
)
def test_grad_log_likelihood(kernel, with_general, seed=42, eps=1.34e-7):
    np.random.seed(seed)
    x = np.sort(np.random.rand(100))
    yerr = np.random.uniform(0.1, 0.5, len(x))
    y = np.sin(x)

    if with_general:
        U = np.vander(x - np.mean(x), 4).T
        V = U * np.random.rand(4)[:, None]
        A = np.sum(U * V, axis=0) + 1e-8
    else:
        A = np.empty(0)
        U = np.empty((0, 0))
        V = np.empty((0, 0))

    if not terms.HAS_AUTOGRAD:
        gp = GP(kernel)
        gp.compute(x, yerr, A=A, U=U, V=V)
        with pytest.raises(ImportError):
            _, grad = gp.grad_log_likelihood(y)
        return

    for fit_mean in [True, False]:
        gp = GP(kernel, fit_mean=fit_mean)
        gp.compute(x, yerr, A=A, U=U, V=V)
        _, grad = gp.grad_log_likelihood(y)
        grad0 = np.empty_like(grad)

        v = gp.get_parameter_vector()
        for i, pval in enumerate(v):
            v[i] = pval + eps
            gp.set_parameter_vector(v)
            ll = gp.log_likelihood(y)

            v[i] = pval - eps
            gp.set_parameter_vector(v)
            ll -= gp.log_likelihood(y)

            grad0[i] = 0.5 * ll / eps
            v[i] = pval
        assert np.allclose(grad, grad0)

def test_predict(seed=42):
    np.random.seed(seed)
    x = np.linspace(1, 59, 300)
    t = np.sort(np.random.uniform(10, 50, 100))
    yerr = np.random.uniform(0.1, 0.5, len(t))
    y = np.sin(t)

    kernel = terms.RealTerm(0.1, 0.5)
    for term in [(0.6, 0.7, 1.0), (0.1, 0.05, 0.5, -0.1)]:
        kernel += terms.ComplexTerm(*term)
    gp = GP(kernel)

    gp.compute(t, yerr)
    K = gp.get_matrix(include_diagonal=True)
    Ks = gp.get_matrix(x, t)
    true_mu = np.dot(Ks, np.linalg.solve(K, y))
    true_cov = gp.get_matrix(x, x) - np.dot(Ks, np.linalg.solve(K, Ks.T))

    mu, cov = gp.predict(y, x)

    _, var = gp.predict(y, x, return_var=True)
    assert np.allclose(mu, true_mu)
    assert np.allclose(cov, true_cov)
    assert np.allclose(var, np.diag(true_cov))

    mu0, cov0 = gp.predict(y, t)
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
