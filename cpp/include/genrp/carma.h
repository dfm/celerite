#ifndef _GENRP_CARMA_
#define _GENRP_CARMA_

#include <cmath>
#include <complex>

#include <Eigen/Dense>

namespace genrp {
  namespace carma {

#define CARMA_SOLVER_UNSTABLE 1

struct Prediction {
  double expectation;
  double variance;
};

struct State {
  double time;
  std::complex<double> x;
  std::complex<double> P;
};

//
// This class evaluates the log likelihood of a CARMA(1,0) model using a Kalman filter.
//
class CARMA_1_0 {
public:

  CARMA_1_0 (double sigma2, std::complex<double> alpha)
    : sigma2_(sigma2)
    , alpha_(alpha)
    , lambda_base_(exp(-alpha))
  {
    V_ = sigma2_ / (alpha_ + std::conj(alpha_));
    std::cout << "V = " << V_ << std::endl;
  };

  void reset (double t) {
    // Step 2 from Kelly et al.
    state_.time = t;
    state_.x = 0.0;
    state_.P = V_;
  };

  Prediction predict (double yerr) const {
    // Steps 3 and 9 from Kelly et al.
    Prediction pred;
    pred.expectation = state_.x.real();
    pred.variance = yerr * yerr + state_.P.real();

    // Check the variance value for instability.
    if (pred.variance < 0.0) throw CARMA_SOLVER_UNSTABLE;
    return pred;
  };

  void update_state (const Prediction& pred, double y) {
    // Steps 4-6 and 10-12 from Kelly et al.
    std::complex<double> K = state_.P / pred.variance;
    state_.x += (y - pred.expectation) * K;
    state_.P -= pred.variance * K * std::conj(K);
  };

  void advance_time (double dt) {
    // Steps 7 and 8 from Kelly et al.
    std::complex<double> lam = pow(lambda_base_, dt);
    state_.time += dt;
    state_.x *= lam;
    state_.P = lam * (state_.P - V_) * std::conj(lam) + V_;
  };

  double log_likelihood (Eigen::VectorXd t, Eigen::VectorXd y, Eigen::VectorXd yerr) {
    unsigned n = t.rows();
    double r, ll = n * log(2.0 * M_PI);
    Prediction pred;

    reset(t(0));
    for (unsigned i = 0; i < n; ++i) {
      // Integrate the Kalman filter.
      pred = predict(yerr(i));
      update_state(pred, y(i));
      if (i < n - 1) advance_time(t(i+1) - t(i));

      // Update the likelihood evaluation.
      r = y(i) - pred.expectation;
      ll += r * r / pred.variance + log(pred.variance);
    }

    return -0.5 * ll;
  };

  double psd (double f) const {
    std::complex<double> w(0.0, 2.0 * M_PI * f);
    return sigma2_ / std::norm(alpha_ + w);
  };

private:

  double sigma2_;
  std::complex<double> alpha_, b_, V_;

  std::complex<double> lambda_base_;
  State state_;

};

};  // namespace carma
};  // namespace genrp

#endif
