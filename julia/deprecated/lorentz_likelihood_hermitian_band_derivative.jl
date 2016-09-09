function lorentz_likelihood_hermitian_band_derivative(p,y::Vector{Float64},t::Vector{Float64},aex,al_small,indx,logdeta)
# Computes the derivative of the log likelihood of a sum of Exponential/Cosine kernels (which have a
# Lorentzian power spectrum) utilizing the approach of Ambikasaran (2015)
#   Numer. Linear Algebra Appl. 2015; 22:1102-1114 DOI: 10.1002/nla
# Uses (2.30) from Rasmussen & Williams (2006) to compute the log likelihood.
# The derivative is computed with respect to the kernel parameters: alpha, beta_real, beta_imaginary.
# The form of the kernel is:
#
#   K(t_i,t_j) = w_i \delta_{ij} + \sum_{k=1}^{p} \alpha_k \exp{-\beta_k |t_i-t_j|}
#
# This is represented by the matrix A_{ij} = K(t_i,t_j).
# The log likelihood is given by:
# ln(L) = -1/2 y^T A^{-1} y - 1/2 |det(A)|
# The derivative of the log likelihood is given by:
# dln(L)/dx = -1/2 sum_i dA_ii/dx /A_ii - 1/2 y^T A^{-1} dA/dx A^{-1} y 
# 
# Note: if a \beta_k is complex, then its complex conjugate *must* be included as well.
#
# First, define band-diagonal matrix for kernel:
# [x_i,{r_{i,1},...,r_{i,p}},{l_{i,1},...,l_{i,p}}]

#tic()
n = length(t)
@assert (length(y) == n)
# There are p+1 sub-diagonals, p+1 super-diagonals + diagonal
# for a total of 2*p+3 non-zero diagonals:
nex = (2p+1)*n-2p
bex = zeros(Complex{Float64},nex)
for i=1:n
# Compute actual indices:
  irow =(i-1)*(1+2p)+1
  bex[irow] = y[i]
# Diagonal noise:
  for j=1:p
    if i < n
      bex[irow+  j] = 0.0
      bex[irow+p+j] = 0.0
    end
  end
end

# Specify the number of bands below & above the diagonal:
m1 = p+1
m2 = p+1
# Solve the equation A^{-1} y = b using band-diagonal LU back-substitution on
# the extended equations: A_{ex}^{-1} y_{ex} = b_{ex}:
banbks(aex,nex,m1,m2,al_small,indx,bex)
ainv_y = zeros(n)
for i=1:n
  ainv_y[i] = real(bex[(i-1)*(2p+1)+1])
end

# Next, substitute this into the derivative matrix.  For now we will just
# compute the alpha derivatives:
dlog_like_dalpha = zeros(p)
# First add the identity component:
for k=1:p
# Next, recursively compute the lower component:
  zl = zeros(n)
  for i=2:n
    zl[i] += (zl[i-1]+y[i-1])*exp(-beta[k]*(t[i]-t[i-1])
  end
# Next, recursively compute the lower component:
  zu = zeros(n)
  for i=n-1:-1:1
    zu[i] += (zu[i+1]+y[i+1])*exp(-beta[k]*(t[i+1]-t[i])
  end
# Finally, add together & substitute back into K^{-1}:
  for i=1:n
# Compute actual indices:
    irow =(i-1)*(1+2p)+1
    bex[irow] = ainv_y[i]+zu[i]+zl[i]
  end
  banbks(aex,nex,m1,m2,al_small,indx,bex)
  for i=1:n
    dlog_like_dalpha[k] +=  0.5*bex[(i-1)*(2p+1)+1]*y[i]
# Add in the trace term:
    dlog_like_dalpha[k] += -0.5/alpha[k]
  end
end
# That should be it!
return dlog_like_dalpha
end
