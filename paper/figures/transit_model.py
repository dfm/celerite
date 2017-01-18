# -*- coding: utf-8 -*-

from __future__ import division, print_function

import transit
import numpy as np
from celerite import terms, modeling

__all__ = ["RotationTerm", "TransitModel"]


class RotationTerm(terms.Term):
    parameter_names = ("log_amp", "log_timescale", "log_period", "log_factor")

    def get_real_coefficients(self):
        f = np.exp(self.log_factor)
        return (
            np.exp(self.log_amp) * (1.0 + f) / (2.0 + f),
            np.exp(-self.log_timescale),
        )

    def get_complex_coefficients(self):
        f = np.exp(self.log_factor)
        return (
            np.exp(self.log_amp) / (2.0 + f),
            0.0,
            np.exp(-self.log_timescale),
            2*np.pi*np.exp(-self.log_period),
        )

class TransitModel(modeling.Model):
    parameter_names = ("mean_flux", "log_period", "log_ror", "log_duration",
                       "t0", "impact", "q1", "q2")

    def __init__(self, texp, *args, **kwargs):
        self.texp = texp
        super(TransitModel, self).__init__(*args, **kwargs)

    def get_value(self, t):
        system = transit.SimpleSystem(
            period=np.exp(self.log_period),
            ror=np.exp(self.log_ror),
            duration=np.exp(self.log_duration),
            t0=self.t0,
            impact=self.impact,
            q1=self.q1,
            q2=self.q2
        )
        lc = system.light_curve(t, texp=self.texp)
        return 1e3 * (lc - 1.0) + self.mean_flux
