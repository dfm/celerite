# -*- coding: utf-8 -*-

from __future__ import division, print_function

import pytest
import numpy as np

from ._genrp import Solver, GP

__all__ = ["test_invalid_parameters", "test_log_determinant", "test_solve"]


def test_invalid_parameters(seed=42):
    np.random.seed(seed)
    t = np.random.rand(50)

    alpha = np.array([1.0, 1.0])
    beta = np.array([1.0 + 1j, 1.0 - 1j])
    with pytest.raises(ValueError):
        Solver(alpha, beta, t)
    t = np.sort(t)
    Solver(alpha, beta, t)

    alpha = np.array([1.0, 5.0])
    beta = np.array([1.0 + 1j, 1.0 - 1j])
    with pytest.raises(ValueError):
        Solver(alpha, beta, t)

    alpha = np.array([1.0, 1.0])
    beta = np.array([1.0 + 1j, 1.0])
    with pytest.raises(ValueError):
        Solver(alpha, beta, t)

    alpha = np.array([1.0])
    beta = np.array([1.0 + 1j, 1.0 - 1j])
    with pytest.raises(ValueError):
        Solver(alpha, beta, t)


def test_log_determinant(seed=42):
    np.random.seed(seed)
    t = np.sort(np.random.rand(10))
    diag = np.random.uniform(0.1, 0.5, len(t))
    alpha = np.array([1.0, 10.0, 10.0])
    beta = np.array([0.5, 1.0 + 1j, 1.0 - 1j])
    solver = Solver(alpha, beta, t, diag)
    K = solver.get_matrix()
    assert np.allclose(solver.log_determinant, np.linalg.slogdet(K)[1])


def test_solve(seed=42):
    np.random.seed(seed)
    t = np.sort(np.random.rand(500))
    diag = np.random.uniform(0.1, 0.5, len(t))
    alpha = np.array([1.0, 10.0, 10.0])
    beta = np.array([0.5, 1.0 + 1j, 1.0 - 1j])
    solver = Solver(alpha, beta, t, diag)
    K = solver.get_matrix()
    b = np.random.randn(len(t))
    assert np.allclose(solver.apply_inverse(b), np.linalg.solve(K, b))


def test_kernel(seed=42):
    gp = GP()
    gp.add_term(-0.5, 0.1)
    gp.add_term(-0.6, 0.7, 1.0)
    assert np.allclose(gp.alpha, [np.exp(-0.5), 0.5*np.exp(-0.6),
                                  0.5*np.exp(-0.6)])
    re = np.exp(-0.7)
    im = 2*np.pi*np.exp(1.0)
    assert np.allclose(gp.beta, [np.exp(-0.1), re+1j*im, re-1j*im])

    gp.add_term(-0.8, 1.0)
    assert np.allclose(gp.alpha, [np.exp(-0.5), np.exp(-0.8), 0.5*np.exp(-0.6),
                                  0.5*np.exp(-0.6)])
    re = np.exp(-0.7)
    im = 2*np.pi*np.exp(1.0)
    assert np.allclose(gp.beta, [np.exp(-0.1), np.exp(-1.0), re+1j*im,
                                 re-1j*im])


def test_build_gp(seed=42):
    gp = GP()
    for term in [(-0.5, 0.1), (-0.6, 0.7, 1.0)]:
        gp.add_term(*term)

    assert len(gp) == 5
    assert np.allclose(gp.params, [-0.5, 0.1, -0.6, 0.7, 1.0])

    gp.params = [0.5, 0.8, -0.6, 0.7, 2.0]
    assert np.allclose(gp.params, [0.5, 0.8, -0.6, 0.7, 2.0])

    with pytest.raises(ValueError):
        gp.params = [0.5, 0.8, -0.6]

    with pytest.raises(ValueError):
        gp.params = "face"

def test_log_likelihood(seed=42):
    np.random.seed(seed)
    x = np.sort(np.random.rand(10))
    yerr = np.random.uniform(0.1, 0.5, len(x))
    y = np.sin(x)

    gp = GP()
    with pytest.raises(RuntimeError):
        gp.log_likelihood(y)
    for term in [(-0.5, 0.1), (-0.6, 0.7, 1.0)]:
        gp.add_term(*term)

        assert gp.computed is False

        with pytest.raises(ValueError):
            gp.compute(np.random.rand(len(x)), yerr)

        gp.compute(x, yerr)
        assert gp.computed is True

        ll = gp.log_likelihood(y)
        K = gp.get_matrix(x)
        K[np.diag_indices_from(K)] += yerr**2
        ll0 = -0.5 * np.dot(y, np.linalg.solve(K, y))
        ll0 -= 0.5 * np.linalg.slogdet(K)[1]
        ll0 -= 0.5 * len(x) * np.log(2*np.pi)
        assert np.allclose(ll, ll0)


# def test_order(seed=42):
#     np.random.seed(seed)
#     t = np.random.rand(500)
#     yerr = np.random.uniform(0.1, 0.5, len(t))
#     b = np.random.randn(len(t))
#     inds = np.argsort(t)

#     solver = GenRPSolver(np.random.randn(3), np.random.randn(3),
#                          np.random.rand(3))

#     solver.compute(t, yerr)
#     K = solver.solver.get_matrix()
#     x1 = solver.apply_inverse(b)
#     x2 = np.empty_like(x1)
#     x2[inds] = np.linalg.solve(K, b[inds])
#     assert np.allclose(x1, x2)

#     t = np.sort(t)
#     solver.compute(t, yerr)
#     K = solver.solver.get_matrix()
#     x1 = solver.apply_inverse(b)
#     x2 = np.linalg.solve(K, b)
#     assert np.allclose(x1, x2)
