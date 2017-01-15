#ifndef _CERERITE_SBAND_H_
#define _CERERITE_SBAND_H_

#include <cmath>
#include <Eigen/Core>

namespace celerite {

template <typename T>
class SymmetricBandMatrix {

public:
  SymmetricBandMatrix (int n, int p) : n_(n), p_(p), zero_(0.0), data_(p, n) {};

  int rows () const { return n_; };
  int cols () const { return n_; };
  T& operator () (int i, int j) {
    int r = i, c = j;
    if (c < r) {
      c = i;
      r = j;
    }
    if (c - r > p_) return zero_;
    return data_(c - r, r);
  };

  Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> toDenseMatrix () const {
    Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> result(n_, n_);
    result.setZero();
    for (int i = 0; i < n_; ++i) {
      result(i, i) = data_(0, i);
      for (int j = i+1; j < std::min(i+p_, n_); ++j) {
        T val = data_(j - i, i);
        result(i, j) = val;
        result(j, i) = val;
      }
    }
    return result;
  };

  void chofactor () {
    for (int j = 0; j < n_; ++j) {
      for (int k = std::max(0, j-p_); k < j; ++k) {
        int l = std::min(k+p_, n_);
        for (int m = j; m < l; ++m) {
          data_(m-j, j) -= data_(j-k, k) * data_(m-k, k) * data_(0, k);
        }
      }
      int l = std::min(j+p_, n_);
      for (int m = j+1; m < l; ++m) {
        data_(m-j, j) /= data_(0, j);
      }
    }
  };

  template <typename Derived>
  void chosolve (Derived& b) const {
    // Forward
    for (int j = 0; j < n_; ++j) {
      for (int i = j+1; i < std::min(j+p_, n_); ++i) {
        b.row(i) -= data_(i-j, j) * b.row(j);
      }
    }

    // Diagonal
    for (int j = 0; j < n_; ++j) {
      b.row(j) /= data_(0, j);
    }

    // Backward
    for (int j = n_-1; j >= 0; --j) {
      for (int i = std::max(0, j-p_+1); i < j; ++i) {
        b.row(i) -= data_(j-i, i) * b.row(j);
      }
    }
  }

private:
  T zero_;
  int n_, p_;
  Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> data_;

};

};

#endif
