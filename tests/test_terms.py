# -*- coding: utf-8 -*-

from __future__ import division, print_function

import pytest
import numpy as np
from itertools import product

from celerite import terms

__all__ = ["test_product", "test_jacobian"]

def test_product(seed=42):
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


def test_bounds(seed=42):
    bounds = [(-1.0, 0.3), (-2.0, 5.0)]
    kernel = terms.RealTerm(log_a=0.1, log_c=0.5, bounds=bounds)
    b0 = kernel.get_parameter_bounds()
    assert all(np.allclose(a, b) for a, b in zip(b0, bounds))

    kernel = terms.RealTerm(log_a=0.1, log_c=0.5,
                            bounds=dict(zip(["log_a", "log_c"], bounds)))
    assert all(np.allclose(a, b)
               for a, b in zip(b0, kernel.get_parameter_bounds()))


@pytest.mark.parametrize(
    "k",
    [
        terms.RealTerm(log_a=0.1, log_c=0.5),
        terms.RealTerm(log_a=0.1, log_c=0.5) +
        terms.RealTerm(log_a=-0.1, log_c=0.7),
        terms.ComplexTerm(log_a=0.1, log_c=0.5, log_d=0.1),
        terms.ComplexTerm(log_a=0.1, log_b=-0.2, log_c=0.5, log_d=0.1),
        terms.SHOTerm(log_S0=0.1, log_Q=-1, log_omega0=0.5),
        terms.SHOTerm(log_S0=0.1, log_Q=1.0, log_omega0=0.5),
        terms.SHOTerm(log_S0=0.1, log_Q=1.0, log_omega0=0.5) +
        terms.RealTerm(log_a=0.1, log_c=0.4),
        terms.SHOTerm(log_S0=0.1, log_Q=1.0, log_omega0=0.5) *
        terms.RealTerm(log_a=0.1, log_c=0.4),
    ]
)
def test_jacobian(k, eps=1.34e-7):
    if not terms.HAS_AUTOGRAD:
        with pytest.raises(ImportError):
            jac = k.get_coeffs_jacobian()
        return

    v = k.get_parameter_vector()
    c = np.concatenate(k.coefficients)
    jac = k.get_coeffs_jacobian()
    assert jac.shape == (len(v), len(c))
    jac0 = np.empty_like(jac)
    for i, pval in enumerate(v):
        v[i] = pval + eps
        k.set_parameter_vector(v)
        coeffs = np.concatenate(k.coefficients)

        v[i] = pval - eps
        k.set_parameter_vector(v)
        coeffs -= np.concatenate(k.coefficients)

        jac0[i] = 0.5 * coeffs / eps
        v[i] = pval
    assert np.allclose(jac, jac0)

@pytest.mark.parametrize(
    "k",
    [
        terms.JitterTerm(log_sigma=0.5),
        terms.RealTerm(log_a=0.5, log_c=0.1),
        terms.RealTerm(log_a=0.5, log_c=0.1) + terms.JitterTerm(log_sigma=0.3),
        terms.JitterTerm(log_sigma=0.5) + terms.JitterTerm(log_sigma=0.1),
    ]
)
def test_jitter_jacobian(k, eps=1.34e-7):
    if not terms.HAS_AUTOGRAD:
        with pytest.raises(ImportError):
            jac = k.get_jitter_jacobian()
        return

    v = k.get_parameter_vector()
    jac = k.get_jitter_jacobian()
    assert len(jac) == len(v)
    jac0 = np.empty_like(jac)
    for i, pval in enumerate(v):
        v[i] = pval + eps
        k.set_parameter_vector(v)
        jitter = k.jitter

        v[i] = pval - eps
        k.set_parameter_vector(v)
        jitter -= k.jitter

        jac0[i] = 0.5 * jitter / eps
        v[i] = pval
    assert np.allclose(jac, jac0)
