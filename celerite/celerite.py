# -*- coding: utf-8 -*-

from __future__ import division, print_function
import math
import warnings
import numpy as np

from . import solver, terms
from .modeling import ModelSet, ConstantModel

__all__ = ["GP"]


class GP(ModelSet):
    """The main interface to the celerite Gaussian Process solver

    Args:
        kernel: An instance of a subclass of :class:`terms.Term`.
        mean (Optional): A simple mean value for the process. This can either
            be a ``float`` or a subclass of :class:`modeling.Model`.
            (default: ``0.0``)
        fit_mean (optional): If ``False``, all of the parameters of ``mean``
            will be frozen. Otherwise, the parameter states are unaffected.
            (default: ``False``)

    """

    def __init__(self,
                 kernel,
                 mean=0.0, fit_mean=False,
                 log_white_noise=None, fit_white_noise=False):
        self._solver = None
        self._computed = False
        self._t = None
        self._y_var = None

        # Backwards compatibility for 'log_white_noise' parameter
        if log_white_noise is not None:
            warnings.warn("The 'log_white_noise' parameter is deprecated. "
                          "Use a 'JitterTerm' instead.")
            k = terms.JitterTerm(log_sigma=float(log_white_noise))
            if not fit_white_noise:
                k.freeze_parameter("log_sigma")
            kernel += k

        # Build up a list of models for the ModelSet
        models = [("kernel", kernel)]

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
    def solver(self):
        if self._solver is None:
            self._solver = solver.CholeskySolver()
        return self._solver

    @property
    def mean(self):
        """The mean :class:`modeling.Model`"""
        return self.models["mean"]

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
        return (
            self._solver is not None and
            self.solver.computed() and
            not self.dirty
        )

    def compute(self, t, yerr=1.123e-12, check_sorted=True,
                A=None, U=None, V=None):
        """
        Compute the extended form of the covariance matrix and factorize

        Args:
            x (array[n]): The independent coordinates of the data points.
                This array must be _sorted_ in ascending order.
            yerr (Optional[float or array[n]]): The measurement uncertainties
                for the data points at coordinates ``x``. These values will be
                added in quadrature to the diagonal of the covariance matrix.
                (default: ``1.123e-12``)
            check_sorted (bool): If ``True``, ``x`` will be checked to make
                sure that it is properly sorted. If ``False``, the coordinates
                will be assumed to be in the correct order.

        Raises:
            ValueError: For un-sorted data or mismatched dimensions.
            solver.LinAlgError: For non-positive definite matrices.

        """
        t = np.atleast_1d(t)
        if check_sorted and np.any(np.diff(t) < 0.0):
            raise ValueError("the input coordinates must be sorted")
        if check_sorted and len(t.shape) > 1:
            raise ValueError("dimension mismatch")
        self._t = t
        self._yerr = np.empty_like(self._t)
        self._yerr[:] = yerr
        (alpha_real, beta_real, alpha_complex_real, alpha_complex_imag,
         beta_complex_real, beta_complex_imag) = self.kernel.coefficients
        self._A = np.empty(0) if A is None else A
        self._U = np.empty((0, 0)) if U is None else U
        self._V = np.empty((0, 0)) if V is None else V
        self.solver.compute(
            self.kernel.jitter,
            alpha_real, beta_real,
            alpha_complex_real, alpha_complex_imag,
            beta_complex_real, beta_complex_imag,
            self._A, self._U, self._V,
            t, self._yerr**2
        )
        self.dirty = False

    def _recompute(self):
        if not self.computed:
            if self._t is None:
                raise RuntimeError("you must call 'compute' first")
            self.compute(self._t, self._yerr, check_sorted=False,
                         A=self._A, U=self._U, V=self._V)

    def _process_input(self, y):
        if self._t is None:
            raise RuntimeError("you must call 'compute' first")
        if len(self._t) != len(y):
            raise ValueError("dimension mismatch")
        return np.ascontiguousarray(y, dtype=float)

    def log_likelihood(self, y, _const=math.log(2.0*math.pi), quiet=False):
        """
        Compute the marginalized likelihood of the GP model

        The factorized matrix from the previous call to :func:`GP.compute` is
        used so ``compute`` must be called first.

        Args:
            y (array[n]): The observations at coordinates ``x`` from
                :func:`GP.compute`.
            quiet (bool): If true, return ``-numpy.inf`` for non-positive
                definite matrices instead of throwing an error.

        Returns:
            float: The marginalized likelihood of the GP model.

        Raises:
            ValueError: For mismatched dimensions.
            solver.LinAlgError: For non-positive definite matrices.

        """
        y = self._process_input(y)
        resid = y - self.mean.get_value(self._t)
        try:
            self._recompute()
        except solver.LinAlgError:
            if quiet:
                return -np.inf
            raise
        if len(y.shape) > 1:
            raise ValueError("dimension mismatch")
        logdet = self.solver.log_determinant()
        if not np.isfinite(logdet):
            return -np.inf
        loglike = -0.5*(self.solver.dot_solve(resid)+logdet+len(y)*_const)
        if not np.isfinite(loglike):
            return -np.inf
        return loglike

    def grad_log_likelihood(self, y, quiet=False):
        """
        Compute the gradient of the marginalized likelihood

        The factorized matrix from the previous call to :func:`GP.compute` is
        used so ``compute`` must be called first. The gradient is taken with
        respect to the parameters returned by :func:`GP.get_parameter_vector`.
        This function requires the `autograd
        <https://github.com/HIPS/autograd>`_ package.

        Args:
            y (array[n]): The observations at coordinates ``x`` from
                :func:`GP.compute`.
            quiet (bool): If true, return ``-numpy.inf`` and a gradient vector
                of zeros for non-positive definite matrices instead of
                throwing an error.

        Returns:
            The gradient of marginalized likelihood with respect to the
            parameter vector.

        Raises:
            ValueError: For mismatched dimensions.
            solver.LinAlgError: For non-positive definite matrices.

        """
        if not solver.has_autodiff():
            raise RuntimeError("celerite must be compiled with autodiff "
                               "support to use the gradient methods")

        if not self.kernel.vector_size:
            return self.log_likelihood(y, quiet=quiet), np.empty(0)

        y = self._process_input(y)
        if len(y.shape) > 1:
            raise ValueError("dimension mismatch")
        resid = y - self.mean.get_value(self._t)

        (alpha_real, beta_real, alpha_complex_real, alpha_complex_imag,
         beta_complex_real, beta_complex_imag) = self.kernel.coefficients
        try:
            val, grad = self.solver.grad_log_likelihood(
                self.kernel.jitter,
                alpha_real, beta_real,
                alpha_complex_real, alpha_complex_imag,
                beta_complex_real, beta_complex_imag,
                self._A, self._U, self._V,
                self._t, resid, self._yerr**2
            )
        except solver.LinAlgError:
            if quiet:
                return -np.inf, np.zeros(self.vector_size)
            raise

        if self.kernel._has_coeffs:
            coeffs_jac = self.kernel.get_coeffs_jacobian()
            full_grad = np.dot(coeffs_jac, grad[1:])
        else:
            full_grad = np.zeros(self.kernel.vector_size)
        if self.kernel._has_jitter:
            jitter_jac = self.kernel.get_jitter_jacobian()
            full_grad += jitter_jac * grad[0]

        if self.mean.vector_size:
            self._recompute()
            alpha = self.solver.solve(resid)
            g = self.mean.get_gradient(self._t)
            full_grad = np.append(full_grad, np.dot(g, alpha))

        return val, full_grad

    def apply_inverse(self, y):
        """
        Apply the inverse of the covariance matrix to a vector or matrix

        Solve ``K.x = y`` for ``x`` where ``K`` is the covariance matrix of
        the GP with the white noise and ``yerr`` components included on the
        diagonal.

        Args:
            y (array[n] or array[n, nrhs]): The vector or matrix ``y``
            described above.

        Returns:
            array[n] or array[n, nrhs]: The solution to the linear system.
            This will have the same shape as ``y``.

        Raises:
            ValueError: For mismatched dimensions.

        """
        self._recompute()
        return self.solver.solve(self._process_input(y))

    def dot(self, y, t=None, A=None, U=None, V=None, kernel=None,
            check_sorted=True):
        """
        Dot the covariance matrix into a vector or matrix

        Compute ``K.y`` where ``K`` is the covariance matrix of the GP without
        the white noise or ``yerr`` values on the diagonal.

        Args:
            y (array[n] or array[n, nrhs]): The vector or matrix ``y``
                described above.
            kernel (Optional[terms.Term]): A different kernel can optionally
                be provided to compute the matrix ``K`` from a different
                kernel than the ``kernel`` property on this object.

        Returns:
            array[n] or array[n, nrhs]: The dot product ``K.y`` as described
            above. This will have the same shape as ``y``.

        Raises:
            ValueError: For mismatched dimensions.

        """
        if kernel is None:
            kernel = self.kernel

        if t is not None:
            t = np.atleast_1d(t)
            if check_sorted and np.any(np.diff(t) < 0.0):
                raise ValueError("the input coordinates must be sorted")
            if check_sorted and len(t.shape) > 1:
                raise ValueError("dimension mismatch")

            A = np.empty(0) if A is None else A
            U = np.empty((0, 0)) if U is None else U
            V = np.empty((0, 0)) if V is None else V
        else:
            if not self.computed:
                raise RuntimeError("you must call 'compute' first")
            t = self._t
            A = self._A
            U = self._U
            V = self._V

        (alpha_real, beta_real, alpha_complex_real, alpha_complex_imag,
         beta_complex_real, beta_complex_imag) = kernel.coefficients

        return self.solver.dot(
            kernel.jitter,
            alpha_real, beta_real,
            alpha_complex_real, alpha_complex_imag,
            beta_complex_real, beta_complex_imag,
            A, U, V, t, np.ascontiguousarray(y, dtype=float)
        )

    def predict(self, y, t=None, return_cov=True, return_var=False):
        """
        Compute the conditional predictive distribution of the model

        You must call :func:`GP.compute` before this method.

        Args:
            y (array[n]): The observations at coordinates ``x`` from
                :func:`GP.compute`.
            t (Optional[array[ntest]]): The independent coordinates where the
                prediction should be made. If this is omitted the coordinates
                will be assumed to be ``x`` from :func:`GP.compute` and an
                efficient method will be used to compute the prediction.
            return_cov (Optional[bool]): If ``True``, the full covariance
                matrix is computed and returned. Otherwise, only the mean
                prediction is computed. (default: ``True``)
            return_var (Optional[bool]): If ``True``, only return the diagonal
                of the predictive covariance; this will be faster to compute
                than the full covariance matrix. This overrides ``return_cov``
                so, if both are set to ``True``, only the diagonal is computed.
                (default: ``False``)

        Returns:
            ``mu``, ``(mu, cov)``, or ``(mu, var)`` depending on the values of
            ``return_cov`` and ``return_var``. These output values are:
            (a) **mu** ``(ntest,)``: mean of the predictive distribution,
            (b) **cov** ``(ntest, ntest)``: the predictive covariance matrix,
            and
            (c) **var** ``(ntest,)``: the diagonal elements of ``cov``.

        Raises:
            ValueError: For mismatched dimensions.

        """
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

        if t is None:
            alpha = self.solver.solve(resid).flatten()
            alpha = resid - (self._yerr**2 + self.kernel.jitter) * alpha
        elif not len(self._A):
            alpha = self.solver.predict(resid, xs)
        else:
            Kxs = self.get_matrix(xs, self._t)
            alpha = np.dot(Kxs, alpha)

        mu = self.mean.get_value(xs) + alpha
        if not (return_var or return_cov):
            return mu

        # Predictive variance.
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

    def get_matrix(self, x1=None, x2=None, include_diagonal=None,
                   include_general=None):
        """
        Get the covariance matrix at given independent coordinates

        Args:
            x1 (Optional[array[n1]]): The first set of independent coordinates.
                If this is omitted, ``x1`` will be assumed to be equal to ``x``
                from a previous call to :func:`GP.compute`.
            x2 (Optional[array[n2]]): The second set of independent
                coordinates. If this is omitted, ``x2`` will be assumed to be
                ``x1``.
            include_diagonal (Optional[bool]): Should the white noise and
                ``yerr`` terms be included on the diagonal?
                (default: ``False``)

        """
        if x1 is None and x2 is None:
            if self._t is None or not self.computed:
                raise RuntimeError("you must call 'compute' first")
            K = self.kernel.get_value(self._t[:, None] - self._t[None, :])
            if include_diagonal is None or include_diagonal:
                K[np.diag_indices_from(K)] += (
                    self._yerr**2 + self.kernel.jitter
                )
            if (include_general is None or include_general) and len(self._A):
                K[np.diag_indices_from(K)] += self._A
                K += np.tril(np.dot(self._U.T, self._V), -1)
                K += np.triu(np.dot(self._V.T, self._U), 1)
            return K

        incl = False
        x1 = np.ascontiguousarray(x1, dtype=float)
        if x2 is None:
            x2 = x1
            incl = include_diagonal is not None and include_diagonal
        K = self.kernel.get_value(x1[:, None] - x2[None, :])
        if incl:
            K[np.diag_indices_from(K)] += self.kernel.jitter
        return K

    def sample(self, size=None):
        """
        Sample from the prior distribution over datasets

        Args:
            size (Optional[int]): The number of samples to draw.

        Returns:
            array[n] or array[size, n]: The samples from the prior
            distribution over datasets.

        """
        self._recompute()
        if size is None:
            n = np.random.randn(len(self._t))
        else:
            n = np.random.randn(len(self._t), size)
        n = self.solver.dot_L(n)
        if size is None:
            return self.mean.get_value(self._t) + n[:, 0]
        return self.mean.get_value(self._t)[None, :] + n.T

    def sample_conditional(self, y, t=None, size=None):
        """
        Sample from the conditional (predictive) distribution

        Note: this method scales as ``O(M^3)`` for large ``M``, where
        ``M == len(t)``.

        Args:
            y (array[n]): The observations at coordinates ``x`` from
                :func:`GP.compute`.
            t (Optional[array[ntest]]): The independent coordinates where the
                prediction should be made. If this is omitted the coordinates
                will be assumed to be ``x`` from :func:`GP.compute` and an
                efficient method will be used to compute the prediction.
            size (Optional[int]): The number of samples to draw.

        Returns:
            array[n] or array[size, n]: The samples from the conditional
            distribution over datasets.

        """
        mu, cov = self.predict(y, t, return_cov=True)
        return np.random.multivariate_normal(mu, cov, size=size)
