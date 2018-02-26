#include <Eigen/Core>

namespace celerite {

template <typename T1, typename T2, typename T3, typename T4, typename T5>
int factor (
  const Eigen::MatrixBase<T1>& U,  // (N, J)
  const Eigen::MatrixBase<T2>& P,  // (N-1, J)
  Eigen::MatrixBase<T3>& d,        // (N);    initially set to A
  Eigen::MatrixBase<T4>& W,        // (N, J); initially set to V
  Eigen::MatrixBase<T5>& S         // (J, J)
) {
  int N = U.rows(), J = U.cols();

  Eigen::Matrix<typename T4::Scalar, 1, T4::ColsAtCompileTime> tmp(1, J);

  // First row
  S.setZero();
  W.row(0) /= d(0);

  // The rest of the rows
  for (int n = 1; n < N; ++n) {
    // Update S = diag(P) * (S + d*W*W.T) * diag(P)
    S.noalias() += d(n-1) * W.row(n-1).transpose() * W.row(n-1);
    S.array() *= (P.row(n-1).transpose() * P.row(n-1)).array();

    // Update d = a - U * S * U.T
    tmp = U.row(n) * S;
    d(n) -= tmp * U.row(n).transpose();
    if (d(n) <= 0.0) return n;

    // Update W = (V - U * S) / d
    W.row(n).noalias() -= tmp;
    W.row(n) /= d(n);
  }

  return 0;
}

template <typename T1, typename T2, typename T3, typename T4, typename T5, typename T6, typename T7>
void factor_grad (
  const Eigen::MatrixBase<T1>& U,   // (N, J)
  const Eigen::MatrixBase<T2>& P,   // (N-1, J)
  const Eigen::MatrixBase<T3>& d,   // (N)
  const Eigen::MatrixBase<T1>& W,   // (N, J)
  const Eigen::MatrixBase<T4>& S,   // (J, J)

  const Eigen::MatrixBase<T4>& bS,  // (J, J)

  Eigen::MatrixBase<T5>& bU,        // (N, J)
  Eigen::MatrixBase<T6>& bP,        // (N-1, J)
  Eigen::MatrixBase<T7>& ba,        // (N)
  Eigen::MatrixBase<T5>& bV         // (N, J)
) {
  int N = U.rows();

  // Make local copies of the gradients that we need.
  Eigen::Matrix<typename T4::Scalar, T4::RowsAtCompileTime, T4::ColsAtCompileTime, T4::IsRowMajor> bS_ = bS, S_ = S;
  Eigen::Matrix<typename T1::Scalar, T1::ColsAtCompileTime, 1> bSWT;

  bV.array().colwise() /= d.array();

  for (int n = N-1; n > 0; --n) {
    // Step 6
    ba(n) -= W.row(n) * bV.row(n).transpose();
    bU.row(n).noalias() -= (bV.row(n) + 2.0 * ba.row(n) * U.row(n)) * S_;
    bS_.noalias() -= U.row(n).transpose() * (bV.row(n) + ba(n) * U.row(n));

    // Step 4
    S_ *= P.row(n-1).asDiagonal().inverse();
    bP.row(n-1).noalias() += (bS_ * S_ + S_.transpose() * bS_).diagonal();

    // Step 3
    bS_ = P.row(n-1).asDiagonal() * bS_ * P.row(n-1).asDiagonal();
    bSWT = bS_ * W.row(n-1).transpose();
    ba(n-1) += W.row(n-1) * bSWT;
    bV.row(n-1).noalias() += W.row(n-1) * (bS_ + bS_.transpose());

    // Downdate S
    S_ = P.row(n-1).asDiagonal().inverse() * S_;
    S_.noalias() -= d(n-1) * W.row(n-1).transpose() * W.row(n-1);
  }

  ba(0) -= bV.row(0) * W.row(0).transpose();
}

template <typename T1, typename T2, typename T3, typename T4, typename T5>
void solve (
  const Eigen::MatrixBase<T1>& U,  // (N, J)
  const Eigen::MatrixBase<T2>& P,  // (N-1, J)
  const Eigen::MatrixBase<T3>& d,  // (N)
  const Eigen::MatrixBase<T1>& W,  // (N, J)
  Eigen::MatrixBase<T4>& Z,        // (N, Nrhs); initially set to Y
  Eigen::MatrixBase<T5>& F,        // (J, Nrhs)
  Eigen::MatrixBase<T5>& G         // (J, Nrhs)
) {
  int N = U.rows();

  F.setZero();
  G.setZero();

  for (int n = 1; n < N; ++n) {
    F.noalias() += W.row(n-1).transpose() * Z.row(n-1);
    F = P.row(n-1).asDiagonal() * F;
    Z.row(n).noalias() -= U.row(n) * F;
  }

  Z.array().colwise() /= d.array();

  for (int n = N-2; n >= 0; --n) {
    G.noalias() += U.row(n+1).transpose() * Z.row(n+1);
    G = P.row(n).asDiagonal() * G;
    Z.row(n).noalias() -= W.row(n) * G;
  }
}

template <typename T1, typename T2, typename T3, typename T4, typename T5, typename T6, typename T7, typename T8, typename T9>
void solve_grad (
  const Eigen::MatrixBase<T1>& U,  // (N, J)
  const Eigen::MatrixBase<T2>& P,  // (N-1, J)
  const Eigen::MatrixBase<T3>& d,  // (N)
  const Eigen::MatrixBase<T1>& W,  // (N, J)
  const Eigen::MatrixBase<T4>& Z,  // (N, Nrhs)
  const Eigen::MatrixBase<T5>& F,  // (J, Nrhs)
  const Eigen::MatrixBase<T5>& G,  // (J, Nrhs)
  const Eigen::MatrixBase<T4>& bZ, // (N, Nrhs)
  const Eigen::MatrixBase<T5>& bF, // (J, Nrhs)
  const Eigen::MatrixBase<T5>& bG, // (J, Nrhs)
  Eigen::MatrixBase<T6>& bU,       // (N, J)
  Eigen::MatrixBase<T7>& bP,       // (N-1, J)
  Eigen::MatrixBase<T8>& bd,       // (N)
  Eigen::MatrixBase<T6>& bW,       // (N, J)
  Eigen::MatrixBase<T9>& bY        // (N, Nrhs)
) {
  int N = U.rows();

  Eigen::Matrix<typename T5::Scalar, T5::RowsAtCompileTime, T5::ColsAtCompileTime, T5::IsRowMajor>
    bF_ = bF, F_ = F, bG_ = bG, G_ = G;
  Eigen::Matrix<typename T4::Scalar, T4::RowsAtCompileTime, T4::ColsAtCompileTime, T4::IsRowMajor>
    Z_ = Z, bZ_ = bZ;

  // TODO: can replace bZ_ by bY if bY initially assumed to be zero.

  for (int n = 0; n <= N-2; ++n) {
    // Grad of: Z.row(n).noalias() -= W.row(n) * G;
    bW.row(n).noalias() -= bZ_.row(n) * G_.transpose();
    bG_.noalias() -= W.row(n).transpose() * bZ_.row(n);

    // Inverse of: Z.row(n).noalias() -= W.row(n) * G;
    Z_.row(n).noalias() += W.row(n) * G_;

    // Inverse of: G = P.row(n).asDiagonal() * G;
    G_ = P.row(n).asDiagonal().inverse() * G_;

    // Grad of: g = P.row(n).asDiagonal() * G;
    bP.row(n).noalias() += (bG_ * G_.transpose()).diagonal();
    bG_ = P.row(n).asDiagonal() * bG_;

    // Inverse of: g.noalias() += U.row(n+1).transpose() * Z.row(n+1);
    G_.noalias() -= U.row(n+1).transpose() * Z_.row(n+1);

    // Grad of: g.noalias() += U.row(n+1).transpose() * Z.row(n+1);
    bU.row(n+1).noalias() += Z_.row(n+1) * bG_.transpose();
    bZ_.row(n+1).noalias() += U.row(n+1) * bG_;
  }

  bZ_.array().colwise() /= d.array();
  bd.array() -= (Z_.array() * bZ_.array()).rowwise().sum();

  // Inverse of: Z.array().colwise() /= d.array();
  Z_.array().colwise() *= d.array();

  for (int n = N-1; n >= 1; --n) {
    // Grad of: Z.row(n).noalias() -= U.row(n) * f;
    bU.row(n).noalias() -= bZ_.row(n) * F_.transpose();
    bF_.noalias() -= U.row(n).transpose() * bZ_.row(n);

    // Inverse of: F = P.row(n-1).asDiagonal() * F;
    F_ = P.row(n-1).asDiagonal().inverse() * F_;

    // Grad of: F = P.row(n-1).asDiagonal() * F;
    bP.row(n-1).noalias() += (bF_ * F_.transpose()).diagonal();
    bF_ = P.row(n-1).asDiagonal() * bF_;

    // Inverse of: F.noalias() += W.row(n-1).transpose() * Z.row(n-1);
    F_.noalias() -= W.row(n-1).transpose() * Z_.row(n-1);

    // Grad of: F.noalias() += W.row(n-1).transpose() * Z.row(n-1);
    bW.row(n-1).noalias() += Z_.row(n-1) * bF_.transpose();
    bZ_.row(n-1).noalias() += W.row(n-1) * bF_;
  }

  bY.array() += bZ_.array();
}

}

