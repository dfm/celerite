# -*- coding: utf-8 -*-

from __future__ import division, print_function
import math
import numpy as np

from ._celerite import Solver
from .modeling import ModelSet, ConstantModel

__all__ = ["GP"]


class GP(ModelSet):
    """The main interface to the celerite Gaussian Process solver

    Args:
        kernel: An instance of a subclass of :class:`terms.Kernel`.
        mean (Optional): A simple mean value for the process. This can either
            be a ``float`` or a subclass of :class:`Model`. (default: ``0.0``)
        fit_mean (optional): If ``False``, all of the parameters of ``mean``
            will be frozen. Otherwise, the parameter states are unaffected.
            (default: ``False``)
        log_white_noise (Optional): A white noise model for the process. The
            ``exp`` of this will be added to the diagonal of the matrix in
            :func:`GP.compute`. This can either be a ``float`` or a subclass
            of :class:`Model`. (default: ``-inf``)
        fit_white_noise (optional): If ``False``, all of the parameters of
            ``log_white_noise`` will be frozen. Otherwise, the parameter
            states are unaffected. (default: ``False``)

    """

    def __init__(self,
                 kernel,
                 mean=0.0, fit_mean=False,
                 log_white_noise=-float("inf"), fit_white_noise=False):
        self.solver = None
        self._computed = False
        self._t = None
        self._y_var = None

        # Build up a list of models for the ModelSet
        models = [("kernel", kernel)]

        # Interpret the white noise model
        try:
            float(log_white_noise)
        except TypeError:
            pass
        else:
            log_white_noise = ConstantModel(float(log_white_noise))

        # If this model is supposed to be constant, go through and freeze
        # all of the parameters
        if not fit_white_noise:
            for k in log_white_noise.get_parameter_names():
                log_white_noise.freeze_parameter(k)
        models += [("log_white_noise", log_white_noise)]

        # And the mean model
        try:
            float(mean)
        except TypeError:
            pass
        else:
            mean = ConstantModel(float(mean))

        if not fit_mean:
            for k in mean.get_parameter_names():
                mean.freeze_parameter(k)
        models += [("mean", mean)]

        # Init the superclass
        super(GP, self).__init__(models)

    @property
    def mean(self):
        return self.models["mean"]

    @property
    def log_white_noise(self):
        return self.models["log_white_noise"]

    @property
    def kernel(self):
        return self.models["kernel"]

    @property
    def dirty(self):
        return super(GP, self).dirty or not self._computed

    @dirty.setter
    def dirty(self, value):
        self._computed = not value
        super(GP, self.__class__).dirty.fset(self, value)

    @property
    def computed(self):
        return not self.dirty

    def compute(self, t, yerr=1.123e-12, check_sorted=True):
        t = np.atleast_1d(t)
        if check_sorted and np.any(np.diff(t) < 0.0):
            raise ValueError("the input coordinates must be sorted")
        if check_sorted and len(t.shape) > 1:
            raise ValueError("dimension mismatch")
        self._t = t
        self._yerr = np.empty_like(self._t)
        self._yerr[:] = yerr
        if self.solver is None:
            self.solver = Solver()
        (alpha_real, beta_real, alpha_complex_real, alpha_complex_imag,
         beta_complex_real, beta_complex_imag) = self.kernel.coefficients
        self.solver.compute(
            alpha_real, beta_real,
            alpha_complex_real, alpha_complex_imag,
            beta_complex_real, beta_complex_imag,
            t, self._get_diag()
        )
        self.dirty = False

    def _recompute(self):
        if self.dirty:
            if self._t is None:
                raise RuntimeError("you must call 'compute' first")
            self.compute(self._t, self._yerr, check_sorted=False)

    def _process_input(self, y):
        if self._t is None:
            raise RuntimeError("you must call 'compute' first")
        if len(self._t) != len(y):
            raise ValueError("dimension mismatch")
        return np.ascontiguousarray(y, dtype=float)

    def log_likelihood(self, y, _const=math.log(2.0*math.pi)):
        y = self._process_input(y)
        resid = y - self.mean.get_value(self._t)
        self._recompute()
        if len(y.shape) > 1:
            raise ValueError("dimension mismatch")
        return -0.5 * (self.solver.dot_solve(resid) +
                       self.solver.log_determinant() +
                       len(y) * _const)

    def apply_inverse(self, y):
        self._recompute()
        return self.solver.solve(self._process_input(y))

    def dot(self, y, kernel=None):
        if kernel is None:
            kernel = self.kernel
        (alpha_real, beta_real, alpha_complex_real, alpha_complex_imag,
         beta_complex_real, beta_complex_imag) = kernel.coefficients
        return self.solver.dot(
            alpha_real, beta_real,
            alpha_complex_real, alpha_complex_imag,
            beta_complex_real, beta_complex_imag,
            self._t, self._process_input(y)
        )

    def predict(self, y, t=None, return_cov=True, return_var=False):
        y = self._process_input(y)
        if len(y.shape) > 1:
            raise ValueError("dimension mismatch")

        if t is None:
            xs = self._t
        else:
            xs = np.ascontiguousarray(t, dtype=float)
            if len(xs.shape) > 1:
                raise ValueError("dimension mismatch")

        # Make sure that the model is computed
        self._recompute()

        # Compute the predictive mean.
        resid = y - self.mean.get_value(self._t)
        alpha = self.solver.solve(resid).flatten()

        if t is None:
            alpha = y - self._get_diag() * alpha
        else:
            Kxs = self.get_matrix(xs, self._t)
            alpha = np.dot(Kxs, alpha)

        mu = self.mean.get_value(xs) + alpha
        if not (return_var or return_cov):
            return mu

        # Predictive variance.
        if t is None:
            Kxs = self.get_matrix(xs, self._t)
        KxsT = np.ascontiguousarray(Kxs.T, dtype=np.float64)
        if return_var:
            var = -np.sum(KxsT*self.apply_inverse(KxsT), axis=0)
            var += self.kernel.get_value(0.0)
            return mu, var

        # Predictive covariance
        cov = self.kernel.get_value(xs[:, None] - xs[None, :])
        cov -= np.dot(Kxs, self.apply_inverse(KxsT))
        return mu, cov

    def _get_diag(self):
        return self._yerr**2 + np.exp(self.log_white_noise
                                      .get_value(self._t))

    def get_matrix(self, x1=None, x2=None, include_diagonal=None):
        if x1 is None and x2 is None:
            if self._t is None or not self.computed:
                raise RuntimeError("you must call 'compute' first")
            K = self.kernel.get_value(self._t[:, None] - self._t[None, :])
            if include_diagonal is None or include_diagonal:
                K[np.diag_indices_from(K)] += self._get_diag()
            return K

        incl = False
        x1 = np.ascontiguousarray(x1, dtype=float)
        if x2 is None:
            x2 = x1
            incl = include_diagonal is not None and include_diagonal
        K = self.kernel.get_value(x1[:, None] - x2[None, :])
        if incl:
            K[np.diag_indices_from(K)] += np.exp(self.log_white_noise
                                                 .get_value(x1))
        return K

    def sample(self, x, tiny=1e-12, size=None):
        K = self.get_matrix(x, include_diagonal=True)
        K[np.diag_indices_from(K)] += tiny
        sample = np.random.multivariate_normal(np.zeros_like(x), K, size=size)
        return self.mean.get_value(x) + sample
