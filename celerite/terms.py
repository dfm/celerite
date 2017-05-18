# -*- coding: utf-8 -*-

from __future__ import division, print_function

try:
    import autograd  # NOQA
except ImportError:
    import numpy as np
    HAS_AUTOGRAD = False
else:
    import autograd.numpy as np
    from autograd import jacobian, elementwise_grad
    HAS_AUTOGRAD = True

from itertools import chain, product

from .modeling import Model, ModelSet
from .solver import get_kernel_value, get_psd_value, check_coefficients

__all__ = [
    "Term", "TermProduct", "TermSum",
    "JitterTerm", "RealTerm", "ComplexTerm", "SHOTerm", "Matern32Term",
]


class Term(Model):
    """
    The abstract base "term" that is the superclass of all other terms

    Subclasses should overload the :func:`terms.Term.get_real_coefficients`
    and :func:`terms.Term.get_complex_coefficients` methods.

    """

    _has_jitter = False
    _has_coeffs = True

    @property
    def terms(self):
        """A list of all the terms included in a sum of terms"""
        return [self]

    def get_value(self, tau):
        """
        Compute the value of the term for an array of lags

        Args:
            tau (array[...]): An array of lags where the term should be
                evaluated.

        Returns:
            The value of the term for each ``tau``. This will have the same
            shape as ``tau``.

        """
        tau = np.asarray(tau)
        (alpha_real, beta_real, alpha_complex_real, alpha_complex_imag,
         beta_complex_real, beta_complex_imag) = self.coefficients
        k = get_kernel_value(
            alpha_real, beta_real,
            alpha_complex_real, alpha_complex_imag,
            beta_complex_real, beta_complex_imag,
            tau.flatten(),
        )
        return np.asarray(k).reshape(tau.shape)

    def get_psd(self, omega):
        """
        Compute the PSD of the term for an array of angular frequencies

        Args:
            omega (array[...]): An array of frequencies where the PSD should
                be evaluated.

        Returns:
            The value of the PSD for each ``omega``. This will have the same
            shape as ``omega``.

        """
        w = np.asarray(omega)
        (alpha_real, beta_real, alpha_complex_real, alpha_complex_imag,
         beta_complex_real, beta_complex_imag) = self.coefficients
        p = get_psd_value(
            alpha_real, beta_real,
            alpha_complex_real, alpha_complex_imag,
            beta_complex_real, beta_complex_imag,
            w.flatten(),
        )
        return p.reshape(w.shape)

    def check_parameters(self):
        """
        Check for negative power in the PSD using Sturm's theorem

        Returns:
            ``True`` for valid parameters.

        """
        return check_coefficients(*(self.coefficients))

    def __add__(self, b):
        return TermSum(self, b)

    def __radd__(self, b):
        return TermSum(b, self)

    def __mul__(self, b):
        return TermProduct(self, b)

    def __rmul__(self, b):
        return TermProduct(b, self)

    def get_real_coefficients(self, params):
        """
        Get the arrays ``alpha_real`` and ``beta_real``

        This method should be overloaded by subclasses to return the arrays
        ``alpha_real`` and ``beta_real`` given the current parameter settings.
        By default, this term is empty.

        Returns:
            (array[j_real], array[j_real]): ``alpha_real`` and ``beta_real``
            as described above.

        """
        return np.empty(0), np.empty(0)

    def get_complex_coefficients(self, params):
        """
        Get the arrays ``alpha_complex_*`` and ``beta_complex_*``

        This method should be overloaded by subclasses to return the arrays
        ``alpha_complex_real``, ``alpha_complex_imag``, ``beta_complex_real``,
        and ``beta_complex_imag`` given the current parameter settings. By
        default, this term is empty.

        Returns:
            (array[j_complex], array[j_complex], array[j_complex],
            array[j_complex]): ``alpha_complex_real``, ``alpha_complex_imag``,
            ``beta_complex_real``, and ``beta_complex_imag`` as described
            above. ``alpha_complex_imag`` can be omitted and it will be
            assumed to be zero.

        """
        return np.empty(0), np.empty(0), np.empty(0), np.empty(0)

    def get_all_coefficients(self, params=None):
        if params is None:
            params = self.get_parameter_vector(include_frozen=True)
        r = self.get_real_coefficients(params)
        c = self.get_complex_coefficients(params)
        if len(c) == 3:
            c = (c[0], np.zeros_like(c[0]), c[1], c[2])
        return list(map(np.atleast_1d, chain(r, c)))

    @property
    def coefficients(self):
        """
        All of the coefficient arrays

        This property is the concatenation of the results from
        :func:`terms.Term.get_real_coefficients` and
        :func:`terms.Term.get_complex_coefficients` but it will always return
        a tuple of length 6, even if ``alpha_complex_imag`` was omitted from
        ``get_complex_coefficients``.

        Returns:
            (array[j_real], array[j_real], array[j_complex], array[j_complex],
            array[j_complex], array[j_complex]): ``alpha_real``, ``beta_real``,
            ``alpha_complex_real``, ``alpha_complex_imag``,
            ``beta_complex_real``, and ``beta_complex_imag`` as described
            above.

        Raises:
            ValueError: For invalid dimensions for the coefficients.

        """
        vector = self.get_parameter_vector(include_frozen=True)
        pars = self.get_all_coefficients(vector)
        if len(pars) != 6:
            raise ValueError("there must be 6 coefficient blocks")
        if any(len(p.shape) != 1 for p in pars):
            raise ValueError("coefficient blocks must be 1D")
        if len(pars[0]) != len(pars[1]):
            raise ValueError("coefficient blocks must have the same shape")
        if any(len(pars[2]) != len(p) for p in pars[3:]):
            raise ValueError("coefficient blocks must have the same shape")
        return pars

    def get_jitter(self, params):
        return 0.0

    @property
    def jitter(self):
        return self.get_jitter(self.get_parameter_vector(include_frozen=True))

    def get_jitter_jacobian(self, include_frozen=False):
        if not HAS_AUTOGRAD:
            raise ImportError("'autograd' must be installed to compute "
                              "gradients")
        jac = elementwise_grad(self.get_jitter)
        jac = jac(self.get_parameter_vector(include_frozen=True))
        if include_frozen:
            return jac
        return jac[self.unfrozen_mask]

    def get_coeffs_jacobian(self, include_frozen=False):
        if not HAS_AUTOGRAD:
            raise ImportError("'autograd' must be installed to compute "
                              "gradients")
        jac = jacobian(lambda p: np.concatenate(self.get_all_coefficients(p)))
        jac = jac(self.get_parameter_vector(include_frozen=True)).T
        if include_frozen:
            return jac
        return jac[self.unfrozen_mask]


class TermProduct(Term, ModelSet):

    def __init__(self, k1, k2):
        if k1._has_jitter or k2._has_jitter:
            raise ValueError("Products are not implemented for terms with "
                             "jitter")
        super(TermProduct, self).__init__([("k1", k1), ("k2", k2)])

    def __repr__(self):
        return " * ".join(map("{0}".format, (self.models["k1"],
                                             self.models["k2"])))

    @property
    def terms(self):
        return [self]

    def get_all_coefficients(self, params=None):
        if params is None:
            params = self.get_parameter_vector(include_frozen=True)
        n = self.models["k1"].full_size
        c1 = self.models["k1"].get_all_coefficients(params[:n])
        c2 = self.models["k2"].get_all_coefficients(params[n:])

        # First compute real terms
        ar = []
        cr = []
        gen = product(zip(c1[0], c1[1]), zip(c2[0], c2[1]))
        for i, ((aj, cj), (ak, ck)) in enumerate(gen):
            ar.append(aj * ak)
            cr.append(cj + ck)

        # Then the complex terms
        ac = []
        bc = []
        cc = []
        dc = []

        # real * complex
        gen = product(zip(c1[0], c1[1]), zip(*(c2[2:])))
        gen = chain(gen, product(zip(c2[0], c2[1]), zip(*(c1[2:]))))
        for i, ((aj, cj), (ak, bk, ck, dk)) in enumerate(gen):
            ac.append(aj * ak)
            bc.append(aj * bk)
            cc.append(cj + ck)
            dc.append(dk)

        # complex * complex
        gen = product(zip(*(c1[2:])), zip(*(c2[2:])))
        for i, ((aj, bj, cj, dj), (ak, bk, ck, dk)) in enumerate(gen):
            ac.append(0.5 * (aj * ak + bj * bk))
            bc.append(0.5 * (bj * ak - aj * bk))
            cc.append(cj + ck)
            dc.append(dj - dk)

            ac.append(0.5 * (aj * ak - bj * bk))
            bc.append(0.5 * (bj * ak + aj * bk))
            cc.append(cj + ck)
            dc.append(dj + dk)

        return list(map(np.array, (ar, cr, ac, bc, cc, dc)))


class TermSum(Term, ModelSet):

    def __init__(self, *terms):
        models = []
        for term in terms:
            models += term.terms
        super(TermSum, self).__init__([("terms[{0}]".format(i), t)
                                       for i, t in enumerate(models)])

    def __repr__(self):
        return "(" + " + ".join(map("{0}".format, self.terms)) + ")"

    @property
    def terms(self):
        return list(self.models.values())

    @property
    def _has_jitter(self):
        return any(t._has_jitter for t in self.models.values())

    @property
    def _has_coeffs(self):
        return any(t._has_coeffs for t in self.models.values())

    def get_all_coefficients(self, params=None):
        if params is None:
            params = self.get_parameter_vector(include_frozen=True)
        coeffs = []
        n = 0
        for t in self.models.values():
            d = t.full_size
            coeffs.append(t.get_all_coefficients(params[n:n+d]))
            n += d
        return [np.concatenate(a) for a in zip(*coeffs)]

    def get_jitter(self, params=None):
        if params is None:
            params = self.get_parameter_vector(include_frozen=True)
        jitter = 0.0
        n = 0
        for t in self.models.values():
            d = t.full_size
            jitter += t.get_jitter(params[n:n+d])
            n += d
        return jitter


class JitterTerm(Term):
    r"""
    A diagonal jitter or "white noise" term

    This term has the form

    .. math::

        k(\tau_{n,m}) = \sigma^2\,\delta_{n,m}

    with the parameter ``log_sigma``.

    .. note:: Jitter is never used in :func:`celerite.GP.predict` and or
        :func:`celerite.terms.Term.get_psd`. If you want to compute the jitter
        power, it is ``term.jitter * N``, where ``N`` is the number of data
        points.

    Args:
        log_sigma (float): The log of the amplitude of the white noise.

    """

    _has_jitter = True
    _has_coeffs = False
    parameter_names = ("log_sigma", )

    def __repr__(self):
        return "JitterTerm({0.log_sigma})".format(self)

    def get_jitter(self, params):
        return np.exp(2.0 * params[0])


class RealTerm(Term):
    r"""
    The simplest celerite term

    This term has the form

    .. math::

        k(\tau) = a_j\,e^{-c_j\,\tau}

    with the parameters ``log_a`` and ``log_c``.

    Strictly speaking, for a sum of terms, the parameter ``a`` could be
    allowed to go negative but since it is somewhat subtle to ensure positive
    definiteness, we require that the amplitude be positive through this
    interface. Advanced users can build a custom term that has negative
    coefficients but care should be taken to ensure positivity.

    Args:
        log_a (float): The log of the amplitude of the term.
        log_c (float): The log of the exponent of the term.

    """

    parameter_names = ("log_a", "log_c")

    def __repr__(self):
        return "RealTerm({0.log_a}, {0.log_c})".format(self)

    def get_real_coefficients(self, params):
        log_a, log_c = params
        return np.exp(log_a), np.exp(log_c)


class ComplexTerm(Term):
    r"""
    A general celerite term

    This term has the form

    .. math::

        k(\tau) = \frac{1}{2}\,\left[(a_j + b_j)\,e^{-(c_j+d_j)\,\tau}
         + (a_j - b_j)\,e^{-(c_j-d_j)\,\tau}\right]

    with the parameters ``log_a``, ``log_b``, ``log_c``, and ``log_d``.
    The parameter ``log_b`` can be omitted and it will be assumed to be zero.

    This term will only correspond to a positive definite kernel (on its own)
    if :math:`a_j\,c_j \ge b_j\,d_j` and the ``log_prior`` method checks for
    this constraint.

    Args:
        log_a (float): The log of the real part of amplitude.
        log_b (float): The log of the imaginary part of amplitude.
        log_c (float): The log of the real part of the exponent.
        log_d (float): The log of the imaginary part of exponent.

    """

    def __init__(self, *args, **kwargs):
        if len(args) == 4 or "log_b" in kwargs:
            self.fit_b = True
            self.parameter_names = ("log_a", "log_b", "log_c", "log_d")
        else:
            self.fit_b = False
            self.parameter_names = ("log_a", "log_c", "log_d")
        super(ComplexTerm, self).__init__(*args, **kwargs)

    def __repr__(self):
        if not self.fit_b:
            return "ComplexTerm({0.log_a}, {0.log_c}, {0.log_d})".format(self)
        return ("ComplexTerm({0.log_a}, {0.log_b}, {0.log_c}, {0.log_d})"
                .format(self))

    def get_complex_coefficients(self, params):
        if not self.fit_b:
            log_a, log_c, log_d = params
            return (
                np.exp(log_a), 0.0, np.exp(log_c), np.exp(log_d)
            )
        log_a, log_b, log_c, log_d = params
        return (
            np.exp(log_a), np.exp(log_b), np.exp(log_c), np.exp(log_d)
        )

    def log_prior(self):
        # Constraint required for term to be positive definite. Can be relaxed
        # with multiple terms but must be treated carefully.
        if self.fit_b and self.log_a + self.log_c < self.log_b + self.log_d:
            return -np.inf
        return super(ComplexTerm, self).log_prior()

    # def get_dcoeffs_dparams(self):
    #     if self.fit_b:
    #         return np.diag(self.get_complex_coefficients())
    #     c = self.get_complex_coefficients()
    #     result = np.zeros((3, 4))
    #     result[0, 0] = c[0]
    #     result[1, 2] = c[2]
    #     result[2, 3] = c[3]
    #     return result


class SHOTerm(Term):
    r"""
    A term representing a stochastically-driven, damped harmonic oscillator

    The PSD of this term is

    .. math::

        S(\omega) = \sqrt{\frac{2}{\pi}} \frac{S_0\,\omega_0^4}
        {(\omega^2-{\omega_0}^2)^2 + {\omega_0}^2\,\omega^2/Q^2}

    with the parameters ``log_S0``, ``log_Q``, and ``log_omega0``.

    Args:
        log_S0 (float): The log of the parameter :math:`S_0`.
        log_Q (float): The log of the parameter :math:`Q`.
        log_omega0 (float): The log of the parameter :math:`\omega_0`.

    """

    parameter_names = ("log_S0", "log_Q", "log_omega0")

    def __repr__(self):
        return "SHOTerm({0.log_S0}, {0.log_Q}, {0.log_omega0})".format(self)

    def get_real_coefficients(self, params):
        log_S0, log_Q, log_omega0 = params
        Q = np.exp(log_Q)
        if Q >= 0.5:
            return np.empty(0), np.empty(0)

        S0 = np.exp(log_S0)
        w0 = np.exp(log_omega0)
        f = np.sqrt(1.0 - 4.0 * Q**2)
        return (
            0.5*S0*w0*Q*np.array([1.0+1.0/f, 1.0-1.0/f]),
            0.5*w0/Q*np.array([1.0-f, 1.0+f])
        )

    def get_complex_coefficients(self, params):
        log_S0, log_Q, log_omega0 = params
        Q = np.exp(log_Q)
        if Q < 0.5:
            return np.empty(0), np.empty(0), np.empty(0), np.empty(0)

        S0 = np.exp(log_S0)
        w0 = np.exp(log_omega0)
        f = np.sqrt(4.0 * Q**2-1)
        return (
            S0 * w0 * Q,
            S0 * w0 * Q / f,
            0.5 * w0 / Q,
            0.5 * w0 / Q * f,
        )


class Matern32Term(Term):
    r"""
    A term that approximates a Matern-3/2 function

    This term is defined as

    .. math::

        k(\tau) = \sigma^2\,\left[
            \left(1+1/\epsilon\right)\,e^{-(1-\epsilon)\sqrt{3}\,\tau/\rho}
            \left(1-1/\epsilon\right)\,e^{-(1+\epsilon)\sqrt{3}\,\tau/\rho}
        \right]

    with the parameters ``log_sigma`` and ``log_rho``. The parameter ``eps``
    controls the quality of the approximation since, in the limit
    :math:`\epsilon \to 0` this becomes the Matern-3/2 function

    .. math::

        \lim_{\epsilon \to 0} k(\tau) = \sigma^2\,\left(1+
        \frac{\sqrt{3}\,\tau}{\rho}\right)\,
        \exp\left(-\frac{\sqrt{3}\,\tau}{\rho}\right)

    Args:
        log_sigma (float): The log of the parameter :math:`\sigma`.
        log_rho (float): The log of the parameter :math:`\rho`.
        eps (Optional[float]): The value of the parameter :math:`\epsilon`.
            (default: `0.01`)

    """

    parameter_names = ("log_sigma", "log_rho")

    def __init__(self, *args, **kwargs):
        eps = kwargs.pop("eps", 0.01)
        super(Matern32Term, self).__init__(*args, **kwargs)
        self.eps = eps

    def __repr__(self):
        return "Matern32Term({0.log_sigma}, {0.log_rho}, eps={0.eps})" \
            .format(self)

    def get_complex_coefficients(self, params):
        log_sigma, log_rho = params
        w0 = np.sqrt(3.0) * np.exp(-log_rho)
        S0 = np.exp(2.0 * log_sigma) / w0
        return (w0*S0, w0*w0*S0/self.eps, w0, self.eps)
