# -*- coding: utf-8 -*-

from __future__ import division, print_function

import pytest
import numpy as np

from ._ess import GRPSolver

__all__ = ["test_invalid_parameters", "test_log_determinant"]


def test_invalid_parameters(seed=42):
    np.random.seed(seed)
    t = np.random.rand(50)

    alpha = np.array([1.0, 1.0])
    beta = np.array([1.0 + 1j, 1.0 - 1j])
    with pytest.raises(ValueError):
        GRPSolver(alpha, beta, t)
    t = np.sort(t)
    GRPSolver(alpha, beta, t)

    alpha = np.array([1.0, 5.0])
    beta = np.array([1.0 + 1j, 1.0 - 1j])
    with pytest.raises(ValueError):
        GRPSolver(alpha, beta, t)

    alpha = np.array([1.0, 1.0])
    beta = np.array([1.0 + 1j, 1.0])
    with pytest.raises(ValueError):
        GRPSolver(alpha, beta, t)

    alpha = np.array([1.0])
    beta = np.array([1.0 + 1j, 1.0 - 1j])
    with pytest.raises(ValueError):
        GRPSolver(alpha, beta, t)


def test_log_determinant(seed=42):
    np.random.seed(seed)
    t = np.sort(np.random.rand(500))
    diag = np.random.uniform(0.1, 0.5, len(t))
    alpha = np.array([1.0, 10.0, 10.0])
    beta = np.array([0.5, 1.0 + 1j, 1.0 - 1j])
    solver = GRPSolver(alpha, beta, t, diag)
    K = solver.get_matrix()
    assert np.allclose(solver.log_determinant, np.linalg.slogdet(K)[1])


def test_solve(seed=42):
    np.random.seed(seed)
    t = np.sort(np.random.rand(500))
    diag = np.random.uniform(0.1, 0.5, len(t))
    alpha = np.array([1.0, 10.0, 10.0])
    beta = np.array([0.5, 1.0 + 1j, 1.0 - 1j])
    solver = GRPSolver(alpha, beta, t, diag)
    K = solver.get_matrix()
    b = np.random.randn(len(t))
    assert np.allclose(solver.apply_inverse(b), np.linalg.solve(K, b))
