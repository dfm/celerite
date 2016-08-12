function lorentz_likelihood_hermitian_band_init_derivative(alpha::Float64,beta::Complex{Float64},
                            w::Float64,t::Vector{Float64})
# Initializes the likelihood of a sum of Exponential/Cosine kernels (which have a
# Lorentzian power spectrum) utilizing the approach of Ambikasaran (2015)
#   Numer. Linear Algebra Appl. 2015; 22:1102-1114 DOI: 10.1002/nla
# Uses (2.30) from Rasmussen & Williams (2006) to compute the log likelihood.
# The form of the kernel is:
#
#   K(t_i,t_j) = w_i \delta_{ij} + \sum_{k=1}^{p} \alpha_k \exp{-\beta_k |t_i-t_j|}
#
# When we take the derivative with respect to alpha_l or beta_l, then only the
# k=l component is relevant.  So, we can write down the derivative of the extended matrix
# in terms of alpha & beta. Now, we want to take derivative wrt the real & complex
# components of beta, so we have 3 derivative matrices to compute.

#tic()
# We only have one component since the derivative wrt the other alphas/beta terms is zero.
p = 1
n = length(t)
# There are p+1 sub-diagonals, p+1 super-diagonals + diagonal
# for a total of 2*p+3 non-zero diagonals:
nex = (2p+1)*n-2p
aex_alpha = zeros(Complex{Float64},nex,2*p+3)
aex_betar = zeros(Complex{Float64},nex,2*p+3)
aex_betai = zeros(Complex{Float64},nex,2*p+3)
gamma = zeros(Complex{Float64},p)
for i=1:n
# Compute actual indices:
  irow =(i-1)*(1+2p)+1
# Diagonal noise:
  jcol = p+2
  aex_alpha[irow,jcol] = 1.0
  for j=1:p
# Equation (61):
    if i > 1
      jcol = 1+j
      aex[irow,jcol] = 1.0
    end
    if i < n
      dt = t[i+1]-t[i]
      gamma[j] = exp(-beta[j]*dt)
      jcol = j+p+2
      aex_betar[irow,jcol] = -dt*gamma[j]
      aex_betai[irow,jcol] = -im*dt*gamma[j]
# Equation (60):
      if i > 1
        jcol = 1
        aex_betar[irow+j,jcol] = -dt*complex(real(gamma[j]),-imag(gamma[j]))
        aex_betai[irow+j,jcol] = im*dt*complex(real(gamma[j]),-imag(gamma[j]))
      end
      jcol = p+2-j
      aex_betar[irow+j,jcol] = -dt*complex(real(gamma[j]),-imag(gamma[j]))
      aex_betai[irow+j,jcol] = im*dt*complex(real(gamma[j]),-imag(gamma[j]))
      jcol = 2p+2
# Equation for r (59):
      jcol = 2
      jcol = 2p+3-j
      aex_alpha[irow+p+j,jcol] = 1.0
      if i < n-1
        dt = t[i+2]-t[i+1]
        jcol = 2p+3
        aex_betar[irow+p+j,jcol] = -dt*exp(-beta[j]*dt)
        aex_betai[irow+p+j,jcol] = -im*dt*exp(-beta[j]*dt)
      end
    end
  end
end

# Specify the number of bands below & above the diagonal:
m1 = p+1
m2 = p+1
# Set up matrix & vector needed for band-diagonal solver:
al_small = zeros(Complex{Float64},nex,m1)
indx = collect(1:nex)
# Do the band-diagonal LU decomposition (indx is a permutation vector for
# pivoting; d gives the sign of the determinant based on the number of pivots): 
d=bandec(aex,nex,m1,m2,al_small,indx)
# Solve the equation A^{-1} y = b using band-diagonal LU back-substitution on
# the extended equations: A_{ex}^{-1} y_{ex} = b_{ex}:
# Now select solution to compute the log likelihood:
# The equation A^{-1} y = b has been solved in bex (the extended vector).
# So, I need to pick out the b portion from bex, and take the dot product
# with y (which is the residuals of the data minus model, which is correlated
# noise that we are modeling with the multi-Lorentzian covariance function):
# Next compute the determinant of A_{ex}:
logdeta = 0.0
for i=1:nex
  logdeta += log(abs(aex[i,1]))
end
return aex,al_small,indx,logdeta
end
