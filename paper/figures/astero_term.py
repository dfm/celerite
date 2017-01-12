# -*- coding: utf-8 -*-

from __future__ import division, print_function

import numpy as np
from celerite import terms

__all__ = ["AsteroTerm"]

# Set up the Gaussian Process model and find the maximum likelihood parameters:
class AsteroTerm(terms.Term):

    parameter_names = (
        "log_S_g", "log_omega_g", "log_nu_max", "log_delta_nu",
        "epsilon", "log_A", "log_Q", "log_W",
    )

    def __init__(self, *args, **kwargs):
        self.nterms = int(kwargs.pop("nterms", 2))
        super(AsteroTerm, self).__init__(*args, **kwargs)

    def get_complex_coefficients(self):
        alpha = np.exp(self.log_S_g + self.log_omega_g) / np.sqrt(2.0)
        beta = np.exp(self.log_omega_g) / np.sqrt(2.0)
        Q = 0.5 + np.exp(self.log_Q)
        j = np.arange(-self.nterms, self.nterms+1, 1)
        delta = j*np.exp(self.log_delta_nu) + self.epsilon
        omega = 2*np.pi * (np.exp(self.log_nu_max) + delta)
        S = np.exp(self.log_A - 0.5*delta**2*np.exp(2*self.log_W)) / Q**2
        return (
            np.append(alpha, S*omega*Q),
            np.append(alpha, S*omega*Q/np.sqrt(4*Q*Q-1)),
            np.append(beta, 0.5*omega/Q),
            np.append(beta, 0.5*omega/Q*np.sqrt(4*Q*Q-1)),
        )

    def log_prior(self):
        lp = super(AsteroTerm, self).log_prior()
        if not np.isfinite(lp):
            return lp
        return lp - 0.5 * self.epsilon**2
