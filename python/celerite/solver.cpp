#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>

#include <Eigen/Core>

#include "celerite/celerite.h"

namespace py = pybind11;

//
// By sub-classing BandSolver here, we can make it picklable but not make C++11 a requirement for the C++ library.
//
// The pybind11 pickle docs are here: http://pybind11.readthedocs.io/en/master/advanced/classes.html#pickling-support
// but the gist is that class just needs to expose __getstate__ and __setstate__. These will just directly call
// serialize and deserialize below.
//
class PicklableBandSolver : public celerite::solver::BandSolver<double> {
public:
  auto serialize () const {
    return std::make_tuple(this->computed_, this->n_, this->p_real_, this->p_complex_, this->log_det_,
                           this->a_, this->al_, this->ipiv_);
  };

  void deserialize (bool computed, int n, int p_real, int p_complex, double log_det,
                    Eigen::MatrixXd a, Eigen::MatrixXd al, Eigen::VectorXi ipiv) {
    this->computed_ = computed;
    this->n_ = n;
    this->p_real_ = p_real;
    this->p_complex_ = p_complex;
    this->log_det_ = log_det;
    this->a_ = a;
    this->al_ = al;
    this->ipiv_ = ipiv;
  };
};

//
// Below is the boilerplate code for the pybind11 extension module.
//
PYBIND11_PLUGIN(solver) {
  py::module m("solver", "celerite extension");

  m.def("get_library_version", []() {
      return CELERITE_VERSION_STRING;
    }
  );

  m.def("get_kernel_value",
    [](
      const Eigen::VectorXd& alpha_real,
      const Eigen::VectorXd& beta_real,
      const Eigen::VectorXd& alpha_complex_real,
      const Eigen::VectorXd& alpha_complex_imag,
      const Eigen::VectorXd& beta_complex_real,
      const Eigen::VectorXd& beta_complex_imag,
      py::array_t<double> tau
    ) {
      auto get_kernel_value_closure = [
        alpha_real, beta_real, alpha_complex_real, alpha_complex_imag, beta_complex_real, beta_complex_imag
      ] (double t) {
        return celerite::get_kernel_value(
          alpha_real, beta_real, alpha_complex_real, alpha_complex_imag, beta_complex_real, beta_complex_imag, t
        );
      };
      return py::vectorize(get_kernel_value_closure)(tau);
    }
  );

  m.def("get_psd_value",
    [](
      const Eigen::VectorXd& alpha_real,
      const Eigen::VectorXd& beta_real,
      const Eigen::VectorXd& alpha_complex_real,
      const Eigen::VectorXd& alpha_complex_imag,
      const Eigen::VectorXd& beta_complex_real,
      const Eigen::VectorXd& beta_complex_imag,
      py::array_t<double> omega
    ) {
      auto get_psd_value_closure = [
        alpha_real, beta_real, alpha_complex_real, alpha_complex_imag, beta_complex_real, beta_complex_imag
      ] (double t) {
        return celerite::get_psd_value(
          alpha_real, beta_real, alpha_complex_real, alpha_complex_imag, beta_complex_real, beta_complex_imag, t
        );
      };
      return py::vectorize(get_psd_value_closure)(omega);
    }
  );

  m.def("check_coefficients",
    [](
      const Eigen::VectorXd& alpha_real,
      const Eigen::VectorXd& beta_real,
      const Eigen::VectorXd& alpha_complex_real,
      const Eigen::VectorXd& alpha_complex_imag,
      const Eigen::VectorXd& beta_complex_real,
      const Eigen::VectorXd& beta_complex_imag
    ) {
      return celerite::check_coefficients(
        alpha_real,
        beta_real,
        alpha_complex_real,
        alpha_complex_imag,
        beta_complex_real,
        beta_complex_imag
      );
    }
  );

  py::class_<PicklableBandSolver> solver(m, "Solver");
  solver.def(py::init<>());

  solver.def("compute", [](PicklableBandSolver& solver,
      const Eigen::VectorXd& alpha_real,
      const Eigen::VectorXd& beta_real,
      const Eigen::VectorXd& alpha_complex_real,
      const Eigen::VectorXd& alpha_complex_imag,
      const Eigen::VectorXd& beta_complex_real,
      const Eigen::VectorXd& beta_complex_imag,
      const Eigen::VectorXd& x,
      const Eigen::VectorXd& diag) {
    return solver.compute(
      alpha_real,
      beta_real,
      alpha_complex_real,
      alpha_complex_imag,
      beta_complex_real,
      beta_complex_imag,
      x,
      diag
    );
  });

  solver.def("solve", [](PicklableBandSolver& solver, const Eigen::MatrixXd& b) {
    return solver.solve(b);
  });

  solver.def("dot", [](PicklableBandSolver& solver,
      const Eigen::VectorXd& alpha_real,
      const Eigen::VectorXd& beta_real,
      const Eigen::VectorXd& alpha_complex_real,
      const Eigen::VectorXd& alpha_complex_imag,
      const Eigen::VectorXd& beta_complex_real,
      const Eigen::VectorXd& beta_complex_imag,
      const Eigen::VectorXd& x,
      const Eigen::MatrixXd& b) {
    return solver.dot(
      alpha_real,
      beta_real,
      alpha_complex_real,
      alpha_complex_imag,
      beta_complex_real,
      beta_complex_imag,
      x,
      b
    );
  });

  solver.def("dot_solve", [](PicklableBandSolver& solver, const Eigen::MatrixXd& b) {
    return solver.dot_solve(b);
  });

  solver.def("log_determinant", [](PicklableBandSolver& solver) {
    return solver.log_determinant();
  });

  solver.def("computed", [](PicklableBandSolver& solver) {
      return solver.computed();
  });

  solver.def("__getstate__", [](const PicklableBandSolver& solver) {
    return solver.serialize();
  });

  solver.def("__setstate__", [](PicklableBandSolver& solver, py::tuple t) {
    if (t.size() != 8)
    throw std::runtime_error("Invalid state!");

    new (&solver) PicklableBandSolver();

    solver.deserialize(
      t[0].cast<bool>(),
      t[1].cast<int>(),
      t[2].cast<int>(),
      t[3].cast<int>(),
      t[4].cast<double>(),
      t[5].cast<Eigen::MatrixXd>(),
      t[6].cast<Eigen::MatrixXd>(),
      t[7].cast<Eigen::VectorXi>()
    );
  });

  return m.ptr();
}
