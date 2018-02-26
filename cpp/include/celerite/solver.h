#ifndef _CELERITE_SOLVER_H_
#define _CELERITE_SOLVER_H_

#include <Eigen/Core>

#include "celerite/engine.h"

namespace celerite {

  template <typename T, int Size>
  class CeleriteSolver {

    typename Eigen::Matrix<T, Eigen::Dynamic, 1> Vector;
    typename Eigen::Matrix<T, Eigen::Dynamic, Size, Eigen::RowMajor> MatrixJ;

    public:

      template <typename Derived>
      void compute (
        const T& jitter,
        const Eigen::Base<Derived>& a_real,
        const Eigen::Base<Derived>& c_real,
        const Eigen::Base<Derived>& a_comp,
        const Eigen::Base<Derived>& b_comp,
        const Eigen::Base<Derived>& c_comp,
        const Eigen::Base<Derived>& d_comp,
        const Eigen::Base<Derived>& x,
        const Eigen::Base<Derived>& diag
      ) {
        int J_real = a_real.rows(), J_comp = a_comp.rows();
        J_ = J_real + 2*J_comp;
        N_ = x.rows();
        U_.resize(N_, J_);
        P_.resize(N_-1, J_);
        D_.resize(N_);
        W_.resize(N_, J_);

        for (int n = 0; n < N_; ++n) {
          U.block(n, 0, 1, J_real) = a_real;

          U.block(n) = a_real;

          if (n < N_-1) {
            auto dx = x(n+1) - x(n);
            P.block(n, 0, 1, J_real) = exp(-c_real(j) * dx);
            auto arg = exp(-c_comp(j) * dx);
            P.block(n, J_real, 1, J_comp) = arg;
            P.block(n, J_real+J_comp, 1, J_comp) = arg;
          }
        }

        //for (int j = 0, j < J_real; ++j) {
        //  U.col(j).setConstant(a_real(j));
        //  P.col(j) = exp(-c_real(j) * dx.array())
        //}
        //for (int j = J_real, j < J_; j += 2) {
        //  U.col(2*j)   = a_comp(j) * cos(d_comp(j) * x.array()) + b_comp(j) * sin(d_comp(j) * x.array());
        //  U.col(2*j+1) = a_comp(j) * sin(d_comp(j) * x.array()) - b_comp(j) * cos(d_comp(j) * x.array());
        //}
      };

    private:

      int N_, J_;
      Matrix U_, W_, P_;
      Vector D_, a_real_, c_real_, a_comp_, b_comp_, c_comp_, d_comp_, t_;

  };

};

#endif
