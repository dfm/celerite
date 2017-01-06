#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>

#include <Eigen/Core>

#include "genrp/genrp.h"

namespace py = pybind11;

PYBIND11_PLUGIN(_genrp) {
  py::module m("_genrp", "GenRP extension");

  m.def("get_library_version", []() {
      return GENRP_VERSION_STRING;
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
        return genrp::get_kernel_value(
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
        return genrp::get_psd_value(
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
      return genrp::check_coefficients(
        alpha_real,
        beta_real,
        alpha_complex_real,
        alpha_complex_imag,
        beta_complex_real,
        beta_complex_imag
      );
    }
  );

  py::class_<genrp::solver::BandSolver<double> > solver(m, "Solver");
  solver.def(py::init<>());

  solver.def("compute",
    [](
      genrp::solver::BandSolver<double>& solver,
      const Eigen::VectorXd& alpha_real,
      const Eigen::VectorXd& beta_real,
      const Eigen::VectorXd& alpha_complex_real,
      const Eigen::VectorXd& alpha_complex_imag,
      const Eigen::VectorXd& beta_complex_real,
      const Eigen::VectorXd& beta_complex_imag,
      const Eigen::VectorXd& x,
      const Eigen::VectorXd& diag
    ) {
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
    }
  );

  solver.def("solve",
    [](
      genrp::solver::BandSolver<double>& solver,
      const Eigen::MatrixXd& b
    ) {
      return solver.solve(b);
    }
  );

  solver.def("dot_solve",
    [](
      genrp::solver::BandSolver<double>& solver,
      const Eigen::MatrixXd& b
    ) {
      return solver.dot_solve(b);
    }
  );

  solver.def("log_determinant",
    [](
      genrp::solver::BandSolver<double>& solver
    ) {
      return solver.log_determinant();
    }
  );

  return m.ptr();
}
