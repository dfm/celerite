// GRP.hpp
// Created by Sivaram Ambikasaran on September 2nd, 2014
// Modifications by Dan Foreman-Mackey
//
// Original license:
//
// Copyright © 2014 New York University.
// All Rights Reserved.
//
// A license, solely to non-commercial parties, to use and copy this software and
// its documentation solely for your non-commercial purposes, without fee and
// without a signed licensing agreement, is hereby granted upon your download of
// the software, through which you agree to the following: 1) the above copyright
// notice, this paragraph and the following three paragraphs will prominently
// appear in all internal copies and modifications; 2) no rights to sublicense or
// further distribute this software for commercial purposes are granted; and 3) no
// rights to assign this license are granted. You may modify or otherwise generate
// derivatives of the software, provided that such derivatives are for
// non-commercial purposes only. The use of all such derivatives shall remain
// subject to the terms contained herein. Please contact The Office of Industrial
// Liaison, New York University, One Park Avenue, 6th Floor, New York, NY 10016,
// (212) 263-8178, for commercial licensing opportunities, or for further
// distribution, modification or license rights.
//
// Created by Sivaram Ambikasaran.
//
// IN NO EVENT SHALL NYU, OR ITS EMPLOYEES, OFFICERS, AGENTS OR TRUSTEES
// (“COLLECTIVELY “NYU PARTIES”) BE LIABLE TO ANY PARTY FOR DIRECT, INDIRECT,
// SPECIAL, INCIDENTAL, OR CONSEQUENTIAL DAMAGES OF ANY KIND, INCLUDING LOST
// PROFITS, ARISING OUT OF ANY CLAIM RESULTING FROM YOUR USE OF THIS SOFTWARE
// AND ITS DOCUMENTATION, EVEN IF ANY OF NYU PARTIES HAS BEEN ADVISED OF THE
// POSSIBILITY OF SUCH CLAIM OR DAMAGE.
//
// NYU SPECIFICALLY DISCLAIMS ANY WARRANTIES OF ANY KIND REGARDING THE SOFTWARE,
// INCLUDING, BUT NOT LIMITED TO, NON-INFRINGEMENT, THE IMPLIED WARRANTIES OF
// MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE, OR THE ACCURACY OR
// USEFULNESS, OR COMPLETENESS OF THE SOFTWARE. THE SOFTWARE AND ACCOMPANYING
// DOCUMENTATION, IF ANY, PROVIDED HEREUNDER IS PROVIDED COMPLETELY "AS IS".
// NYU HAS NO OBLIGATION TO PROVIDE FURTHER DOCUMENTATION, MAINTENANCE,
// SUPPORT, UPDATES, ENHANCEMENTS, OR MODIFICATIONS.
//
//
// Please cite Generalized Rybicki Press algorithm and the code, if you use the
// code, ESS, in your research.
//
// Citation for the article:
//
// @article{ambikasaran2014generalized,
//   title={Generalized Rybicki Press algorithm},
//   author={Ambikasaran, Sivaram},
//   journal={arXiv preprint arXiv:1409.7852},
//   year={2014}
// }
//
// Citation for the code:
//
// @MISC{ESSweb,
//   author = {Sivaram Ambikasaran},
//   title = {ESS},
//   howpublished = {https://github.com/sivaramambikasaran/ESS},
//   year = {2014}
//  }
//

#ifndef __GRP_HPP__
#define __GRP_HPP__

#include <cmath>
#include <vector>
#include <Eigen/Dense>
#include <Eigen/Sparse>

#define GRP_DIMENSION_MISMATCH 1

class GRP {
public:

  GRP(const Eigen::VectorXd alpha, const Eigen::VectorXcd beta)
  : alpha_(alpha)
  , beta_(beta)
  , m_(alpha.rows())
  {};

  GRP(size_t m, double* alpha, std::complex<double>* beta) {
    m_ = m;
    alpha_ = Eigen::Map<Eigen::MatrixXd>(alpha, m, 1);
    beta_ = Eigen::Map<Eigen::MatrixXcd>(beta, m, 1);
  };

  void compute (const Eigen::VectorXd t, const Eigen::VectorXd d);
  void compute (size_t N, double* t, double* d) {
    compute(Eigen::Map<Eigen::MatrixXd>(t, N, 1),
            Eigen::Map<Eigen::MatrixXd>(d, N, 1));
  };

  Eigen::VectorXd solve (const Eigen::VectorXd rhs) const;
  void solve(double* rhs, double* solution) const;
  double get_log_determinant () const;

private:
  Eigen::VectorXd alpha_;
  Eigen::VectorXcd beta_;

  // Number of unknowns.
  size_t N_;

  // Rank of the separable part.
  size_t m_;

  // Size of the extended sparse matrix.
  size_t M_;

  // Locations of the blocks.
  std::vector<size_t> nBlockStart_;

  // The extended sparse matrix.
  Eigen::SparseLU<Eigen::SparseMatrix<std::complex<double> >,
                  Eigen::COLAMDOrdering<int> > factor_;
};

void GRP::compute (const Eigen::VectorXd t, const Eigen::VectorXd d) {
  if (t.rows() != d.rows())
    throw GRP_DIMENSION_MISMATCH;
  N_ = t.rows();

  Eigen::MatrixXcd gamma = Eigen::MatrixXcd(m_, N_-1);
  for (size_t i = 0; i < m_; ++i)
    for (size_t j = 0; j < N_-1; ++j)
      gamma(i,j) = exp(-beta_(i) * fabs(t(j) - t(j+1)));

  // Declare 2*m as a variable, since this will be used frequently.
  size_t twom = 2 * m_;

  // Declare the blocksize, which is the repeating structure along the
  // diagonal.
  size_t nBlockSize = twom+1;

  // Size of the extended sparse matrix.
  M_ = N_*nBlockSize-twom;

  // Obtain the starting index of each of the block.
  for (size_t k = 0; k < N_; ++k) nBlockStart_.push_back(k*nBlockSize);

  // Assembles block by block except the identity matrices on the
  // supersuperdiagonal.
  std::vector<Eigen::Triplet<std::complex<double> > > triplets;
  for (size_t nBlock = 0; nBlock < N_-1; ++nBlock) {
    // The starting row and column for the blocks.
    // Assemble the diagonal first.
    triplets.push_back(
        Eigen::Triplet<std::complex<double> >(
          nBlockStart_[nBlock], nBlockStart_[nBlock], d(nBlock)));
    for (size_t k=0; k < m_; ++k) {
      triplets.push_back(
          Eigen::Triplet<std::complex<double> >(
            nBlockStart_[nBlock]+k+1, nBlockStart_[nBlock], gamma(k,nBlock)));
      triplets.push_back(
          Eigen::Triplet<std::complex<double> >(
            nBlockStart_[nBlock], nBlockStart_[nBlock]+k+1, gamma(k,nBlock)));
      triplets.push_back(
          Eigen::Triplet<std::complex<double> >(
            nBlockStart_[nBlock]+m_+k+1, nBlockStart_[nBlock]+twom+1, alpha_(k)));
      triplets.push_back(
          Eigen::Triplet<std::complex<double> >(
            nBlockStart_[nBlock]+twom+1, nBlockStart_[nBlock]+m_+k+1, alpha_(k)));
      triplets.push_back(
          Eigen::Triplet<std::complex<double> >(
            nBlockStart_[nBlock]+k+1, nBlockStart_[nBlock]+k+m_+1, -1.0));
      triplets.push_back(
          Eigen::Triplet<std::complex<double> >(
            nBlockStart_[nBlock]+k+m_+1, nBlockStart_[nBlock]+k+1, -1.0));
    }
  }
  triplets.push_back(Eigen::Triplet<std::complex<double> >(M_-1, M_-1, d(N_-1)));

  // Assembles the supersuperdiagonal identity blocks.
  for (size_t nBlock = 0; nBlock < N_-2; ++nBlock) {
    for (size_t k = 0; k < m_; ++k) {
      triplets.push_back(
          Eigen::Triplet<std::complex<double> >(
            nBlockStart_[nBlock]+k+m_+1, nBlockStart_[nBlock]+twom+k+2, gamma(k,nBlock+1)));
      triplets.push_back(
          Eigen::Triplet<std::complex<double> >(
            nBlockStart_[nBlock]+twom+k+2, nBlockStart_[nBlock]+k+m_+1, gamma(k,nBlock+1)));
    }
  }

  // Assemble the matrix from triplets.
  Eigen::SparseMatrix<std::complex<double> > Aex;
  Aex.resize(M_, M_);
  Aex.setFromTriplets(triplets.begin(), triplets.end());
  factor_.compute(Aex);
}

Eigen::VectorXd GRP::solve (const Eigen::VectorXd rhs) const {
  // Assemble the extended right hand side `rhsex'
  Eigen::VectorXcd rhsex = Eigen::VectorXcd::Zero(M_);
  for (size_t nBlock = 0; nBlock < N_; ++nBlock)
      rhsex(nBlockStart_[nBlock]) = rhs(nBlock);

  Eigen::VectorXcd solutionex = factor_.solve(rhsex);

  Eigen::VectorXd solution = Eigen::VectorXd(N_);
  for (size_t nBlock = 0; nBlock < N_; ++nBlock)
    solution(nBlock) = solutionex(nBlockStart_[nBlock]).real();

  return solution;
}

void GRP::solve(double* rhs, double* solution) const {
  Eigen::Map<Eigen::VectorXd> rhs_(rhs, N_, 1);
  Eigen::VectorXd solution_ = solve(rhs_);
  for (int i = 0; i < N_; ++i) solution[i] = solution_[i];
}

double GRP::get_log_determinant () const {
  return factor_.logAbsDeterminant().real();
}

#endif /* defined(__GRP_HPP__) */
