#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>

#include <Eigen/Core>

#ifndef NO_AUTODIFF
#include <cmath>
#ifdef USE_STAN_MATH
#include <vector>
#include <stan/math.hpp>
#else
#include <cfloat>
#include <unsupported/Eigen/AutoDiff>
#endif
#endif

#include "celerite/celerite.h"
#include "celerite/carma.h"


namespace py = pybind11;

//
// By sub-classing CholeskySolver here, we can make it picklable but not make
// C++11 a requirement for the C++ library.
//
// The pybind11 pickle docs are here:
// http://pybind11.readthedocs.io/en/master/advanced/classes.html#pickling-support
// but the gist is that class just needs to expose __getstate__ and __setstate__.
// These will just directly call serialize and deserialize below.
//
class PicklableCholeskySolver : public celerite::solver::CholeskySolver<double> {
public:
  PicklableCholeskySolver () : celerite::solver::CholeskySolver<double>() {};

  auto serialize () const {
    return std::make_tuple(
      this->computed_, this->N_, this->J_, this->log_det_,
      this->phi_, this->u_, this->W_, this->D_
    );
  };

  void deserialize (
      bool computed, int n, int J, double log_det,
      Eigen::MatrixXd phi,
      Eigen::MatrixXd u,
      Eigen::MatrixXd W,
      Eigen::VectorXd D) {
    this->computed_ = computed;
    this->N_        = n;
    this->J_        = J;
    this->log_det_  = log_det;
    this->phi_      = phi;
    this->u_        = u;
    this->W_        = W;
    this->D_        = D;
  };
};

//
// Below is the boilerplate code for the pybind11 extension module.
//
PYBIND11_PLUGIN(solver) {
  typedef Eigen::MatrixXd matrix_t;
  typedef Eigen::VectorXd vector_t;

  py::module m("solver", R"delim(
This is the low-level interface to the C++ implementation of the celerite
algorithm. These methods do most of the heavy lifting but most users shouldn't
need to call these directly. This interface was built using `pybind11
<http://pybind11.readthedocs.io/>`_.

)delim");

  m.def("get_library_version", []() { return CELERITE_VERSION_STRING; },
        "The version of the linked C++ library");

  m.def("has_autodiff", []() {
#ifdef NO_AUTODIFF
    return false;
#else
    return true;
#endif
  }, "Returns True if celerite was compiled with autodiff support");

  py::register_exception<celerite::linalg_exception>(m, "LinAlgError");

  m.def("get_kernel_value",
    [](
      const vector_t& a_real,
      const vector_t& c_real,
      const vector_t& a_comp,
      const vector_t& b_comp,
      const vector_t& c_comp,
      const vector_t& d_comp,
      py::array_t<double> tau
    ) {
      auto get_kernel_value_closure = [a_real, c_real, a_comp, b_comp, c_comp, d_comp] (double t) {
        return celerite::get_kernel_value(
          a_real, c_real, a_comp, b_comp, c_comp, d_comp, t
        );
      };
      return py::vectorize(get_kernel_value_closure)(tau);
    },
    R"delim(
Get the value of the kernel for given parameters and lags

Args:
    a_real (array[j_real]): The coefficients of the real terms.
    c_real (array[j_real]): The exponents of the real terms.
    a_comp (array[j_complex]): The real part of the coefficients of the
        complex terms.
    b_comp (array[j_complex]): The imaginary part of the coefficients of
        the complex terms.
    c_comp (array[j_complex]): The real part of the exponents of the
        complex terms.
    d_comp (array[j_complex]): The imaginary part of the exponents of the
        complex terms.
    tau (array[n]): The time lags where the kernel should be evaluated.

Returns:
    array[n]: The kernel evaluated at ``tau``.

)delim");

  m.def("get_psd_value",
    [](
      const vector_t& a_real,
      const vector_t& c_real,
      const vector_t& a_comp,
      const vector_t& b_comp,
      const vector_t& c_comp,
      const vector_t& d_comp,
      py::array_t<double> omega
    ) {
      auto get_psd_value_closure = [a_real, c_real, a_comp, b_comp, c_comp, d_comp] (double t) {
        return celerite::get_psd_value(
          a_real, c_real, a_comp, b_comp, c_comp, d_comp, t
        );
      };
      return py::vectorize(get_psd_value_closure)(omega);
    },
    R"delim(
Get the PSD of the kernel for given parameters and angular frequencies

Args:
    a_real (array[j_real]): The coefficients of the real terms.
    c_real (array[j_real]): The exponents of the real terms.
    a_comp (array[j_complex]): The real part of the coefficients of the
        complex terms.
    b_comp (array[j_complex]): The imaginary part of the coefficients of
        the complex terms.
    c_comp (array[j_complex]): The real part of the exponents of the
        complex terms.
    d_comp (array[j_complex]): The imaginary part of the exponents of the
        complex terms.
    omega (array[n]): The frequencies where the PSD should be evaluated.

Returns:
    array[n]: The PSD evaluated at ``omega``.

)delim");

  m.def("check_coefficients",
    [](
      const vector_t& a_real,
      const vector_t& c_real,
      const vector_t& a_comp,
      const vector_t& b_comp,
      const vector_t& c_comp,
      const vector_t& d_comp
    ) {
      return celerite::check_coefficients(a_real, c_real, a_comp, b_comp, c_comp, d_comp);
    },
    R"delim(
Apply Sturm's theorem to check if parameters yield a positive PSD

Args:
    a_real (array[j_real]): The coefficients of the real terms.
    c_real (array[j_real]): The exponents of the real terms.
    a_comp (array[j_complex]): The real part of the coefficients of the
        complex terms.
    b_comp (array[j_complex]): The imaginary part of the coefficients of
        the complex terms.
    c_comp (array[j_complex]): The real part of the exponents of the
        complex terms.
    d_comp (array[j_complex]): The imaginary part of the exponents of the
        complex terms.

Returns:
    bool: ``True`` if the PSD is everywhere positive.

)delim");


  //
  // ------ CARMA ------
  //
  py::class_<celerite::carma::CARMASolver> carma_solver(m, "CARMASolver", R"delim(
A thin wrapper around the C++ CARMASolver class

This solver is parameterized following carma_pack:
https://github.com/brandonckelly/carma_pack

Args:
    log_sigma (float): The log of the variance of the process.
    arparams (array[p]): The parameters of the autoregressive component.
    maparams (array[q]): The parameters of the moving average component.

)delim");
  carma_solver.def(py::init<double, vector_t, vector_t>());

  carma_solver.def("log_likelihood",
    [](celerite::carma::CARMASolver& solver, const vector_t& t, const vector_t& y, const vector_t& yerr) {
      return solver.log_likelihood(t, y, yerr);
    }, R"delim(
Compute the log likelihood using a Kalman filter

Args:
    t (array[n]): The input coordinates of the observations.
    y (array[n]): The observations.
    yerr (array[n]): The measurement uncertainties of the observations.

)delim");

  carma_solver.def("get_celerite_coeffs",
    [](celerite::carma::CARMASolver& solver) {
      vector_t a_real, c_real, a_comp, b_comp, c_comp, d_comp;
      solver.get_celerite_coeffs(a_real, c_real, a_comp, b_comp, c_comp, d_comp);
      return std::make_tuple(a_real, c_real, a_comp, b_comp, c_comp, d_comp);
    }, R"delim(
Compute the coefficients of the celerite model for the given CARMA model

)delim");


  //
  // ------ CHOLESKY ------
  //
  py::class_<PicklableCholeskySolver> cholesky_solver(m, "CholeskySolver", R"delim(
A thin wrapper around the C++ CholeskySolver class
)delim");
  cholesky_solver.def(py::init<>());

#ifdef USE_STAN_MATH
  cholesky_solver.def("grad_log_likelihood",
    [](
        PicklableCholeskySolver& nothing,
        double jitter,
        const vector_t& a_real,
        const vector_t& c_real,
        const vector_t& a_comp,
        const vector_t& b_comp,
        const vector_t& c_comp,
        const vector_t& d_comp,
        const vector_t& A,
        const matrix_t& U,
        const matrix_t& V,
        const vector_t& x,
        const vector_t& y,
        const vector_t& diag
    ) {

#ifndef NO_AUTODIFF
      typedef stan::math::var g_t;
      typedef Eigen::Matrix<g_t, Eigen::Dynamic, 1> v_t;

      int J_real = a_real.rows();
      int J_comp = a_comp.rows();

      // Set up the solver to track the gradients
      celerite::solver::CholeskySolver<g_t> solver;
      g_t jitter_ = g_t(jitter);
      v_t a_real_(J_real), c_real_(J_real),
          a_comp_(J_comp), b_comp_(J_comp), c_comp_(J_comp), d_comp_(J_comp);
      a_real_ << a_real;
      c_real_ << c_real;
      a_comp_ << a_comp;
      b_comp_ << b_comp;
      c_comp_ << c_comp;
      d_comp_ << d_comp;

      // Factorize the matrix while propagating the gradients
      solver.compute(
        jitter_, a_real_, c_real_, a_comp_, b_comp_, c_comp_, d_comp_,
        A, U, V, x, diag
      );

      // Compute the likelihood
      g_t ll = -0.5 * (solver.dot_solve(y) + solver.log_determinant() + M_PI * log(x.rows()));
      double ll_val = ll.val();

      // Evaluate the backpropagated gradients
      std::vector<g_t> params;
      params.push_back(jitter_);
      for (int i = 0; i < J_real; ++i) params.push_back(a_real_(i));
      for (int i = 0; i < J_real; ++i) params.push_back(c_real_(i));
      for (int i = 0; i < J_comp; ++i) params.push_back(a_comp_(i));
      for (int i = 0; i < J_comp; ++i) params.push_back(b_comp_(i));
      for (int i = 0; i < J_comp; ++i) params.push_back(c_comp_(i));
      for (int i = 0; i < J_comp; ++i) params.push_back(d_comp_(i));
      std::vector<double> g;
      ll.grad(params, g);

      // Copy the results to a numpy array
      auto result = py::array_t<double>(g.size());
      auto buf = result.request();
      double* ptr = (double *) buf.ptr;
      for (size_t i = 0; i < g.size(); ++i) ptr[i] = g[i];

      // Tell stan that we don't need these gradients anymore
      stan::math::recover_memory();

      return std::make_tuple(ll_val, result);
#else
      throw std::exception();
#endif
    }, R"delim(
Compute the gradient of the log likelihood of the model using autodiff

The returned gradient is with respect to the jitter and the coefficients.

Args:
    jitter (float): The jitter of the kernel.
    a_real (array[j_real]): The coefficients of the real terms.
    c_real (array[j_real]): The exponents of the real terms.
    a_comp (array[j_complex]): The real part of the coefficients of the
        complex terms.
    b_comp (array[j_complex]): The imaginary part of the coefficients of
        the complex terms.
    c_comp (array[j_complex]): The real part of the exponents of the
        complex terms.
    d_comp (array[j_complex]): The imaginary part of the exponents of the
        complex terms.
    x (array[n]): The _sorted_ array of input coordinates.
    y (array[n]): The observations at ``x``.
    diag (array[n]): An array that should be added to the diagonal of the
        matrix. This often corresponds to measurement uncertainties and in
        that case, ``diag`` should be the measurement _variance_
        (i.e. sigma^2).

)delim");

#else

  cholesky_solver.def("grad_log_likelihood",
    [](
      PicklableCholeskySolver& nothing,
      double jitter,
      const vector_t& a_real,
      const vector_t& c_real,
      const vector_t& a_comp,
      const vector_t& b_comp,
      const vector_t& c_comp,
      const vector_t& d_comp,
      const vector_t& A,
      const matrix_t& U,
      const matrix_t& V,
      const vector_t& x,
      const vector_t& y,
      const vector_t& diag
    ) {

#ifndef NO_AUTODIFF
      typedef Eigen::AutoDiffScalar<Eigen::VectorXd> g_t;
      typedef Eigen::Matrix<g_t, Eigen::Dynamic, 1> v_t;

      int J_real = a_real.rows();
      int J_comp = a_comp.rows();
      int g_tot = 2 * J_real + 4 * J_comp;

      celerite::solver::CholeskySolver<g_t> solver;
      v_t a_real_(J_real), c_real_(J_real),
          a_comp_(J_comp), b_comp_(J_comp), c_comp_(J_comp), d_comp_(J_comp);

      // This hack is needed because, if jitter is zero, we end up with
      // numerically unstable gradients in all dimensions.
      bool compute_jitter = false;
      int i0 = 0, i = 0;
      g_t jitter_ = g_t(jitter);
      if (jitter > DBL_EPSILON) {
        compute_jitter = true;
        i0 = 1;
        g_tot ++;
        jitter_ = g_t(jitter, g_tot, 0);
      } else {
        jitter_ = g_t(jitter);
      }

      // Keep track of the coordinates of each gradient
      if (J_real) {
        for (i = 0; i < J_real; ++i) a_real_(i) = g_t(a_real(i), g_tot, i0+i);
        i0 += i;
        for (i = 0; i < J_real; ++i) c_real_(i) = g_t(c_real(i), g_tot, i0+i);
        i0 += i;
      }
      if (J_comp) {
        for (i = 0; i < J_comp; ++i) a_comp_(i) = g_t(a_comp(i), g_tot, i0+i);
        i0 += i;
        for (i = 0; i < J_comp; ++i) b_comp_(i) = g_t(b_comp(i), g_tot, i0+i);
        i0 += i;
        for (i = 0; i < J_comp; ++i) c_comp_(i) = g_t(c_comp(i), g_tot, i0+i);
        i0 += i;
        for (i = 0; i < J_comp; ++i) d_comp_(i) = g_t(d_comp(i), g_tot, i0+i);
      }

      // Factorize and track the gradients
      solver.compute(
        jitter_, a_real_, c_real_, a_comp_, b_comp_, c_comp_, d_comp_,
        A, U, V, x, diag
      );

      // Compute the likelihood and the gradients
      g_t ll = -0.5 * (solver.dot_solve(y) + solver.log_determinant() + M_PI * log(x.rows()));
      double ll_val = ll.value();

      // Deal with our zero jitter hack
      Eigen::VectorXd g;
      if (compute_jitter) {
        g = ll.derivatives();
      } else {
        g.resize(g_tot + 1);
        g(0) = 0.0;
        g.tail(g_tot) = ll.derivatives();
      }

      // Copy the result to a numpy array
      auto result = py::array_t<double>(g.size());
      auto buf = result.request();
      double* ptr = (double *) buf.ptr;
      for (int i = 0; i < g.rows(); ++i) ptr[i] = g(i);

      return std::make_tuple(ll_val, result);
#else
      throw std::exception();
#endif

    }, R"delim(
Compute the gradient of the log likelihood of the model using autodiff

The returned gradient is with respect to the jitter and the coefficients.

Args:
    jitter (float): The jitter of the kernel.
    a_real (array[j_real]): The coefficients of the real terms.
    c_real (array[j_real]): The exponents of the real terms.
    a_comp (array[j_complex]): The real part of the coefficients of the
        complex terms.
    b_comp (array[j_complex]): The imaginary part of the coefficients of
        the complex terms.
    c_comp (array[j_complex]): The real part of the exponents of the
        complex terms.
    d_comp (array[j_complex]): The imaginary part of the exponents of the
        complex terms.
    x (array[n]): The _sorted_ array of input coordinates.
    y (array[n]): The observations at ``x``.
    diag (array[n]): An array that should be added to the diagonal of the
        matrix. This often corresponds to measurement uncertainties and in
        that case, ``diag`` should be the measurement _variance_
        (i.e. sigma^2).

)delim");

#endif

  cholesky_solver.def("compute", [](PicklableCholeskySolver& solver,
      double jitter,
      const vector_t& a_real,
      const vector_t& c_real,
      const vector_t& a_comp,
      const vector_t& b_comp,
      const vector_t& c_comp,
      const vector_t& d_comp,
      const vector_t& A,
      const matrix_t& U,
      const matrix_t& V,
      const vector_t& x,
      const vector_t& diag) {
    return solver.compute(
      jitter, a_real, c_real, a_comp, b_comp, c_comp, d_comp, A, U, V, x, diag
    );
  },
  R"delim(
Compute the Cholesky factorization of the matrix

Args:
    jitter (float): The jitter of the kernel.
    a_real (array[j_real]): The coefficients of the real terms.
    c_real (array[j_real]): The exponents of the real terms.
    a_comp (array[j_complex]): The real part of the coefficients of the
        complex terms.
    b_comp (array[j_complex]): The imaginary part of the coefficients of
        the complex terms.
    c_comp (array[j_complex]): The real part of the exponents of the
        complex terms.
    d_comp (array[j_complex]): The imaginary part of the exponents of the
        complex terms.
    x (array[n]): The _sorted_ array of input coordinates.
    diag (array[n]): An array that should be added to the diagonal of the
        matrix. This often corresponds to measurement uncertainties and in
        that case, ``diag`` should be the measurement _variance_
        (i.e. sigma^2).

)delim");

  cholesky_solver.def("solve", [](PicklableCholeskySolver& solver, const matrix_t& b) {
    return solver.solve(b);
  },
  R"delim(
Solve a linear system for the matrix defined in ``compute``

A previous call to :func:`solver.CholeskySolver.compute` defines a matrix
``A`` and this method solves for ``x`` in the matrix equation ``A.x = b``.

Args:
    b (array[n] or array[n, nrhs]): The right hand side of the linear system.

Returns:
    array[n] or array[n, nrhs]: The solution of the linear system.

Raises:
    ValueError: For mismatched dimensions.

)delim");

  cholesky_solver.def("dot_solve", [](PicklableCholeskySolver& solver, const matrix_t& b) {
    return solver.dot_solve(b);
  },
  R"delim(
Solve the system ``b^T . A^-1 . b``

A previous call to :func:`solver.Solver.compute` defines a matrix ``A``
and this method solves ``b^T . A^-1 . b`` for a vector ``b``.

Args:
    b (array[n]): The right hand side of the linear system.

Returns:
    float: The solution of ``b^T . A^-1 . b``.

Raises:
    ValueError: For mismatched dimensions.

)delim");

  cholesky_solver.def("dot_L", [](PicklableCholeskySolver& solver, const matrix_t& z) {
    return solver.dot_L(z);
  },
  R"delim(
Compute the dot product of the square root of a ``celerite`` matrix

This method computes ``L.z`` where ``A = L.L^T`` is the matrix defined in
``compute``.

Args:
    z (array[n] or array[n, neq]): The matrix ``z`` described above.

Returns:
    array[n] or array[n, neq]: The dot product ``L.b`` as described above.

Raises:
    ValueError: For mismatched dimensions.

)delim");

  cholesky_solver.def("dot", [](PicklableCholeskySolver& solver,
      double jitter,
      const vector_t& a_real,
      const vector_t& c_real,
      const vector_t& a_comp,
      const vector_t& b_comp,
      const vector_t& c_comp,
      const vector_t& d_comp,
      const vector_t& A,
      const matrix_t& U,
      const matrix_t& V,
      const vector_t& x,
      const matrix_t& b) {
    return solver.dot(jitter, a_real, c_real, a_comp, b_comp, c_comp, d_comp, A, U, V, x, b);
  },
  R"delim(
Compute the dot product of a ``celerite`` matrix and another arbitrary matrix

This method computes ``A.b`` where ``A`` is defined by the parameters and
``b`` is an arbitrary matrix of the correct shape.

Args:
    jitter (float): The jitter of the kernel.
    a_real (array[j_real]): The coefficients of the real terms.
    c_real (array[j_real]): The exponents of the real terms.
    a_comp (array[j_complex]): The real part of the coefficients of the
        complex terms.
    b_comp (array[j_complex]): The imaginary part of the coefficients of
        the complex terms.
    c_comp (array[j_complex]): The real part of the exponents of the
        complex terms.
    d_comp (array[j_complex]): The imaginary part of the exponents of the
        complex terms.
    x (array[n]): The _sorted_ array of input coordinates.
    b (array[n] or array[n, neq]): The matrix ``b`` described above.

Returns:
    array[n] or array[n, neq]: The dot product ``A.b`` as described above.

Raises:
    ValueError: For mismatched dimensions.

)delim");

  cholesky_solver.def("predict", [](PicklableCholeskySolver& solver,
      const vector_t& y,
      const vector_t& x) {
    return solver.predict(y, x);
  },
  R"delim(

)delim");

  cholesky_solver.def("log_determinant", [](PicklableCholeskySolver& solver) {
    return solver.log_determinant();
  },
  R"delim(
Get the log-determinant of the matrix defined by ``compute``

Returns:
    float: The log-determinant of the matrix defined by
    :func:`solver.CholeskySolver.compute`.

)delim");

  cholesky_solver.def("computed", [](PicklableCholeskySolver& solver) {
      return solver.computed();
  },
  R"delim(
A flag that indicates if ``compute`` has been executed

Returns:
    bool: ``True`` if :func:`solver.CholeskySolver.compute` was previously
    executed successfully.

)delim");

  cholesky_solver.def("__getstate__", [](const PicklableCholeskySolver& solver) {
    return solver.serialize();
  });

  cholesky_solver.def("__setstate__", [](PicklableCholeskySolver& solver, py::tuple t) {
    if (t.size() != 8) throw std::runtime_error("Invalid state!");
    new (&solver) PicklableCholeskySolver();
    solver.deserialize(
      t[0].cast<bool>(),
      t[1].cast<int>(),
      t[2].cast<int>(),
      t[3].cast<double>(),

      t[4].cast<matrix_t>(),
      t[5].cast<matrix_t>(),
      t[6].cast<matrix_t>(),

      t[7].cast<vector_t>()
    );
  });

  return m.ptr();
}
