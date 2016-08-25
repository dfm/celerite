function compute_likelihood01(p::Int64,y::Vector,aex::Array,al_small::Array,indx::Vector{Int64},logdeta::Real)
# Computes the likelihood of a sum of Exponential/Cosine kernels (which have a
# Lorentzian power spectrum) utilizing the approach of Ambikasaran (2015)
#   Numer. Linear Algebra Appl. 2015; 22:1102-1114 DOI: 10.1002/nla
# Uses (2.30) from Rasmussen & Williams (2006) to compute the log likelihood.
# The form of the kernel is:
#
#   K(t_i,t_j) = w_i \delta_{ij} + \sum_{k=1}^{p} \alpha_k \exp{-\beta_k |t_i-t_j|}
#
# This is represented by the matrix A_{ij} = K(t_i,t_j).
# The log likelihood is given by:
# ln(L) = -1/2 y^T A^{-1} y - 1/2 |det(A)|
# Note: if a \beta_k is complex, then its complex conjugate *must* be included as well.
#
# First, define band-diagonal matrix for kernel:
# [x_i,{r_{i,1},...,r_{i,p}},{l_{i,1},...,l_{i,p}}]

#tic()
n = length(y)
nex = (n-1)*(4p+1)+1

# There are 2(p+1) sub-diagonals, 2(p+1) super-diagonals + diagonal
# for a total of 4*p+5 non-zero diagonals:
bex = zeros(eltype(aex),nex)
czero = zero(eltype(aex))
for i=1:n
# Compute actual indices:
  irow =(i-1)*(1+4p)+1
  bex[irow] = y[i]/2.0

# Diagonal noise:
  for j=1:p
    if i < n
      bex[irow+  j] = czero
      bex[irow+p+j] = czero
    end
  end
end

# Specify the number of bands below & above the diagonal:
m1 = 2p+2
m2 = m1
# Solve the equation A^{-1} y = b using band-diagonal LU back-substitution on
# the extended equations: A_{ex}^{-1} y_{ex} = b_{ex}:
#@code_warntype banbks(aex,nex,m1,m2,al_small,indx,bex)
banbks_trans(aex,nex,m1,m2,al_small,indx,bex)
# Now select solution to compute the log likelihood:
# The equation A^{-1} y = b has been solved in bex (the extended vector).
# So, I need to pick out the b portion from bex, and take the dot product
# with y (which is the residuals of the data minus model, which is correlated
# noise that we are modeling with the multi-Lorentzian covariance function):
log_like = zero(eltype(aex))
for i=1:n
  log_like += bex[(i-1)*(4p+1)+1]*y[i]
end
# Convert this to log likelihood:
#log_like = -0.5*(log_like+logdeta)
log_like = -0.5*log_like
return log_like
end
