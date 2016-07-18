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

class GRP {
private:
    // Number of unknowns.
    int N;

    // Rank of the separable part.
    int m;

    // The semi-separable matrix is of the form diag(d) + triu(U*V,1) + tril((U*V)',-1).
    Eigen::VectorXcd alpha;
    Eigen::VectorXcd beta;
    Eigen::VectorXd t;

    // Diagonal entries of the matrix.
    Eigen::VectorXd d;

    // Size of the extended sparse matrix.
    int M;

    // Number of non-zeros per block.
    int blocknnz;

    // Total number of non-zeros.
    int nnz;

    // Size of each block, will be 2m+1.
    int nBlockSize;

    // Starting index of each of the blocks.
    std::vector<int> nBlockStart;
    // Vector of triplets used to store the sparse matrix.
    std::vector<Eigen::Triplet<std::complex<double> > > triplets;
    // The extended sparse matrix.
    Eigen::SparseMatrix<std::complex<double> > Aex;

    // Stores the factorization.
    Eigen::SparseLU<Eigen::SparseMatrix<std::complex<double> >, Eigen::COLAMDOrdering<int> > factorize;

public:

    GRP(
        const int N,
        const int m,
        const Eigen::VectorXcd alpha,
        const Eigen::VectorXcd beta,
        const Eigen::VectorXd t,
        const Eigen::VectorXd d
    );
    GRP(
        const int N,
        const int m,
        std::complex<double>* alpha,
        std::complex<double>* beta,
        double* t,
        double* d
    );
    void assemble_Extended_Matrix();
    void factorize_Extended_Matrix();
    void obtain_Solution(
        const Eigen::VectorXd rhs,
        Eigen::VectorXd& solution,
        Eigen::VectorXcd& solutionex
    );
    void obtain_Solution(
        double* rhs,
        double* solution
    );
    double obtain_Determinant();

    // Obtain error, i.e., ||Ax-b||_{\inf}
    /* double obtain_Error(const Eigen::VectorXd rhs, const Eigen::VectorXd& solex); */
};

GRP::GRP (const int N, const int m, const Eigen::VectorXcd alpha, const Eigen::VectorXcd beta, const Eigen::VectorXd t, const Eigen::VectorXd d) {
    this->N     = N;
    this->m     = m;
    this->alpha = alpha;
    this->beta  = beta;
    this->t     = t;
    this->d     = d;
}

GRP::GRP (
    const int N,
    const int m,
    std::complex<double>* alpha,
    std::complex<double>* beta,
    double* t,
    double* d
) {
    this->N     = N;
    this->m     = m;
    this->alpha = Eigen::Map<Eigen::MatrixXcd>(alpha, m, 1);
    this->beta  = Eigen::Map<Eigen::MatrixXcd>(beta, m, 1);
    this->t     = Eigen::Map<Eigen::MatrixXd>(t, N, 1);
    this->d     = Eigen::Map<Eigen::MatrixXd>(d, N, 1);
}

void GRP::assemble_Extended_Matrix() {
    Eigen::MatrixXcd gamma = Eigen::MatrixXcd(m,N-1);
    for (int i=0; i<m; ++i) {
        for (int j=0; j<N-1; ++j) {
            gamma(i,j)  =   exp(-beta(i)*fabs(t(j)-t(j+1)));
        }
    }
    //  Declare 2*m as a variable, since this will be used frequently.
    int twom = 2*m;

    //  Number of non-zeros per matrix block in the extended sparse matrix.
    //  This includes the '1' on the diagonal, the two negative identity
    //  matrices above and below the diagonal, the vectors u, u', v and v'.
    blocknnz = 6*m+1;

    //  Declare the blocksize, which is the repeating structure along the diagonal.
    nBlockSize = twom+1;

    //  Size of the extended sparse matrix.
    M = N*nBlockSize-twom;

    //  Number of non-zero entries in the matrix. The identity matrices on the
    //  supersuperdiagonals which were not included in the blocknnz has been
    //  accounted for here.
    //  This number is correct and has been checked with MATLAB.
    nnz = (N-1)*blocknnz+(N-2)*twom+1;

    //  Obtain the starting index of each of the block.
    for (int k=0; k<N; ++k) {
        nBlockStart.push_back(k*nBlockSize);
    }

    //  Assembles block by block except the identity matrices on the supersuperdiagonal.
    for (int nBlock=0; nBlock<N-1; ++nBlock) {
        //  The starting row and column for the blocks.
        //  Assemble the diagonal first.
        triplets.push_back(
            Eigen::Triplet<std::complex<double> >(
                nBlockStart[nBlock], nBlockStart[nBlock], d(nBlock)));
        for (int k=0; k<m; ++k) {
            triplets.push_back(
                Eigen::Triplet<std::complex<double> >(
                    nBlockStart[nBlock]+k+1,nBlockStart[nBlock],gamma(k,nBlock)));
            triplets.push_back(
                Eigen::Triplet<std::complex<double> >(
                    nBlockStart[nBlock],nBlockStart[nBlock]+k+1,gamma(k,nBlock)));
            triplets.push_back(
                Eigen::Triplet<std::complex<double> >(
                    nBlockStart[nBlock]+m+k+1,nBlockStart[nBlock]+twom+1,alpha(k)));
            triplets.push_back(
                Eigen::Triplet<std::complex<double> >(
                    nBlockStart[nBlock]+twom+1,nBlockStart[nBlock]+m+k+1,alpha(k)));
            triplets.push_back(
                Eigen::Triplet<std::complex<double> >(
                    nBlockStart[nBlock]+k+1,nBlockStart[nBlock]+k+m+1,-1.0));
            triplets.push_back(
                Eigen::Triplet<std::complex<double> >(
                    nBlockStart[nBlock]+k+m+1,nBlockStart[nBlock]+k+1,-1.0));
        }
    }
    triplets.push_back(Eigen::Triplet<std::complex<double> >(M-1,M-1,d(N-1)));

    //  Assebmles the supersuperdiagonal identity blocks.
    for (int nBlock=0; nBlock<N-2; ++nBlock) {
        for (int k=0; k<m; ++k) {
            triplets.push_back(
                Eigen::Triplet<std::complex<double> >(
                    nBlockStart[nBlock]+k+m+1,nBlockStart[nBlock]+twom+k+2,gamma(k,nBlock+1)));
            triplets.push_back(
                Eigen::Triplet<std::complex<double> >(
                    nBlockStart[nBlock]+twom+k+2,nBlockStart[nBlock]+k+m+1,gamma(k,nBlock+1)));
        }
    }

    //  Set the size of the extended sparse matrix.
    Aex.resize(M,M);

    //  Assemble the matrix from triplets.
    Aex.setFromTriplets(triplets.begin(), triplets.end());
}

void GRP::factorize_Extended_Matrix() {
    //  Compute the sparse LU factorization of matrix `Aex'
    factorize.compute(Aex);
}

void GRP::obtain_Solution(const Eigen::VectorXd rhs, Eigen::VectorXd& solution, Eigen::VectorXcd& solutionex) {
    //  Assemble the extended right hand side `rhsex'
    Eigen::VectorXcd rhsex = Eigen::VectorXcd::Zero(M);
    for (int nBlock=0; nBlock<N; ++nBlock) {
        rhsex(nBlockStart[nBlock]) = rhs(nBlock);
    }

    //  Obtain the solution
    solutionex = factorize.solve(rhsex);

    //  Desired solution vector
    solution = Eigen::VectorXd(N);
    for (int nBlock=0; nBlock<N; ++nBlock) {
        solution(nBlock) = solutionex(nBlockStart[nBlock]).real();
    }
}

void GRP::obtain_Solution(double* rhs, double* solution) {
    Eigen::Map<Eigen::VectorXd> rhs_(rhs, this->N, 1);
    Eigen::VectorXd solution_;
    Eigen::VectorXcd solex;
    this->obtain_Solution(rhs_, solution_, solex);
    for (int i = 0; i < N; ++i) solution[i] = solution_[i];
}

double GRP::obtain_Determinant() {
    return factorize.logAbsDeterminant().real();
}

#endif /* defined(__GRP_HPP__) */
