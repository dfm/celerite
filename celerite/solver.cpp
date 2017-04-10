#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>

#include <Eigen/Core>

#include "celerite/celerite.h"
#include "celerite/carma.h"

namespace py = pybind11;

//
// By sub-classing CholeskySolver here, we can make it picklable but not make C++11 a requirement for the C++ library.
//
// The pybind11 pickle docs are here: http://pybind11.readthedocs.io/en/master/advanced/classes.html#pickling-support
// but the gist is that class just needs to expose __getstate__ and __setstate__. These will just directly call
// serialize and deserialize below.
//
class PicklableCholeskySolver : public celerite::solver::CholeskySolver<double> {
public:
  PicklableCholeskySolver () : celerite::solver::CholeskySolver<double>() {};

  auto serialize () const {
    return std::make_tuple(
      this->computed_, this->N_, this->J_, this->log_det_,
      this->phi_, this->u_, this->X_, this->D_
    );
  };

  void deserialize (
      bool computed, int n, int J, double log_det,
      Eigen::MatrixXd phi,
      Eigen::MatrixXd u,
      Eigen::MatrixXd X,
      Eigen::VectorXd D) {
    this->computed_ = computed;
    this->N_        = n;
    this->J_        = J;
    this->log_det_  = log_det;
    this->phi_      = phi;
    this->u_        = u;
    this->X_        = X;
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

  m.def("get_library_version", []() { return CELERITE_VERSION_STRING; }, "The version of the linked C++ library");

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
)delim");
  carma_solver.def(py::init<double, Eigen::VectorXd, Eigen::VectorXd>());
  carma_solver.def("log_likelihood", [](celerite::carma::CARMASolver& solver, const Eigen::VectorXd& t, const Eigen::VectorXd& y, const Eigen::VectorXd& yerr) {
    return solver.log_likelihood(t, y, yerr);
  });
  carma_solver.def("get_celerite_coeffs", [](celerite::carma::CARMASolver& solver) {
    Eigen::VectorXd alpha_real, beta_real, alpha_complex_real, alpha_complex_imag, beta_complex_real, beta_complex_imag;
    solver.get_celerite_coeffs(
      alpha_real, beta_real, alpha_complex_real, alpha_complex_imag, beta_complex_real, beta_complex_imag
    );
    return std::make_tuple(
      alpha_real, beta_real, alpha_complex_real, alpha_complex_imag, beta_complex_real, beta_complex_imag
    );
  });


  //
  // ------ CHOLESKY ------
  //
  py::class_<PicklableCholeskySolver> cholesky_solver(m, "CholeskySolver", R"delim(
A thin wrapper around the C++ CholeskySolver class
)delim");
  cholesky_solver.def(py::init<>());

  cholesky_solver.def("compute", [](PicklableCholeskySolver& solver,
      const vector_t& a_real,
      const vector_t& c_real,
      const vector_t& a_comp,
      const vector_t& b_comp,
      const vector_t& c_comp,
      const vector_t& d_comp,
      const vector_t& x,
      const vector_t& diag) {
    return solver.compute(
      a_real, c_real, a_comp, b_comp, c_comp, d_comp, x, diag
    );
  },
  R"delim(
Assemble the extended matrix and perform the banded LU decomposition

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
    x (array[n]): The _sorted_ array of input coordinates.
    diag (array[n]): An array that should be added to the diagonal of the
        matrix. This often corresponds to measurement uncertainties and in
        that case, ``diag`` should be the measurement _variance_
        (i.e. sigma^2).

Returns:
    int: ``1`` if the dimensions are inconsistent and ``0`` otherwise. No
    attempt is made to confirm that the matrix is positive definite. If
    it is not positive definite, the ``solve`` and ``log_determinant``
    methods will return incorrect results.

)delim");

  cholesky_solver.def("solve", [](PicklableCholeskySolver& solver, const matrix_t& b) {
    return solver.solve(b);
  },
  R"delim(
Solve a linear system for the matrix defined in ``compute``

A previous call to :func:`solver.Solver.compute` defines a matrix ``A``
and this method solves for ``x`` in the matrix equation ``A.x = b``.

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
      const vector_t& a_real,
      const vector_t& c_real,
      const vector_t& a_comp,
      const vector_t& b_comp,
      const vector_t& c_comp,
      const vector_t& d_comp,
      const vector_t& x,
      const matrix_t& b) {
    return solver.dot(a_real, c_real, a_comp, b_comp, c_comp, d_comp, x, b);
  },
  R"delim(
Compute the dot product of a ``celerite`` matrix and another arbitrary matrix

This method computes ``A.b`` where ``A`` is defined by the parameters and
``b`` is an arbitrary matrix of the correct shape.

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
    x (array[n]): The _sorted_ array of input coordinates.
    b (array[n] or array[n, neq]): The matrix ``b`` described above.

Returns:
    array[n] or array[n, neq]: The dot product ``A.b`` as described above.

Raises:
    ValueError: For mismatched dimensions.

)delim");

  cholesky_solver.def("log_determinant", [](PicklableCholeskySolver& solver) {
    return solver.log_determinant();
  },
  R"delim(
Get the log-determinant of the matrix defined by ``compute``

Returns:
    float: The log-determinant of the matrix defined by
    :func:`solver.Solver.compute`.

)delim");

  cholesky_solver.def("computed", [](PicklableCholeskySolver& solver) {
      return solver.computed();
  },
  R"delim(
A flag that indicates if ``compute`` has been executed

Returns:
    bool: ``True`` if :func:`solver.Solver.compute` was previously executed
    successfully.

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
