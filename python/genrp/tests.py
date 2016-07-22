# -*- coding: utf-8 -*-

from __future__ import division, print_function

import pytest
import numpy as np

# from .genrp import GRPSolver
from ._genrp import CythonGenRPSolver

__all__ = ["test_invalid_parameters", "test_log_determinant", "test_solve",
           "test_order"]


def test_invalid_parameters(seed=42):
    np.random.seed(seed)
    t = np.random.rand(50)

    alpha = np.array([1.0, 1.0])
    beta = np.array([1.0 + 1j, 1.0 - 1j])
    with pytest.raises(ValueError):
        CythonGenRPSolver(alpha, beta, t)
    t = np.sort(t)
    CythonGenRPSolver(alpha, beta, t)

    alpha = np.array([1.0, 5.0])
    beta = np.array([1.0 + 1j, 1.0 - 1j])
    with pytest.raises(ValueError):
        CythonGenRPSolver(alpha, beta, t)

    alpha = np.array([1.0, 1.0])
    beta = np.array([1.0 + 1j, 1.0])
    with pytest.raises(ValueError):
        CythonGenRPSolver(alpha, beta, t)

    alpha = np.array([1.0])
    beta = np.array([1.0 + 1j, 1.0 - 1j])
    with pytest.raises(ValueError):
        CythonGenRPSolver(alpha, beta, t)


def test_log_determinant(seed=42):
    np.random.seed(seed)
    t = np.sort(np.random.rand(10))
    diag = np.random.uniform(0.1, 0.5, len(t))
    alpha = np.array([1.0, 10.0, 10.0])
    beta = np.array([0.5, 1.0 + 1j, 1.0 - 1j])
    solver = CythonGenRPSolver(alpha, beta, t, diag)
    K = solver.get_matrix()
    assert np.allclose(solver.log_determinant, np.linalg.slogdet(K)[1])


def test_solve(seed=42):
    np.random.seed(seed)
    t = np.sort(np.random.rand(500))
    diag = np.random.uniform(0.1, 0.5, len(t))
    alpha = np.array([1.0, 10.0, 10.0])
    beta = np.array([0.5, 1.0 + 1j, 1.0 - 1j])
    solver = CythonGenRPSolver(alpha, beta, t, diag)
    K = solver.get_matrix()
    b = np.random.randn(len(t))
    assert np.allclose(solver.apply_inverse(b), np.linalg.solve(K, b))


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
