function lorentz_likelihood_real2_band_init(alpha::Vector,beta_real::Vector,beta_imag::Vector,
                            w::Real,t::Vector,
       nex::Int64,aex1::Array,aex2::Array,al_small1::Array,al_small2::Array,indx::Vector{Int64})
# Initializes the likelihood of a sum of Exponential/Cosine kernels (which have a
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
p::Int64 = length(alpha)
width = 2*p+3
n::Int64 = length(t)
czero = zero(eltype(alpha))
cone = one(eltype(alpha))
atot = one(eltype(alpha))*sum(alpha)
#@assert(length(w) == n)
#@assert(length(y) == n)
#@assert(length(beta_real) == p)
#@assert(length(beta_imag) == p)
#@assert(length(indx) == nex)
#@assert(size(aex1) == (width,nex))
#@assert(size(aex2) == (width,nex))
# There are p+1 sub-diagonals, p+1 super-diagonals + diagonal
# for a total of 2*p+3 non-zero diagonals:
#nex::Int64 = (2p+1)*n-2p
#aex = zeros(Complex{Float64},nex,2*p+3)
#bex = zeros(Complex{Float64},nex)
gamma1 = zeros(eltype(alpha),p)
gamma2 = zeros(eltype(alpha),p)
irow::Int64 = 1
jcol::Int64 = 1
phi = czero
ebt = czero
for i=1:n
# Compute actual indices:
  irow =(i-1)*(1+2p)+1
#  bex[irow] = y[i]
# Diagonal noise:
  jcol = p+2
  aex1[jcol,irow] = w + atot
  aex2[jcol,irow] = czero
  for j=1:p
# Equation (61):
    if i > 1
      jcol = 1+j
      aex1[jcol,irow] = alpha[j]
      aex2[jcol,irow] = czero
    end
    if i < n
      phi = beta_imag[j]*(t[i+1]-t[i])
      ebt = exp(-beta_real[j]*(t[i+1]-t[i]))
      gamma1[j] =  ebt*cos(phi)
      gamma2[j] = -ebt*sin(phi)
#      println(gamma[j])
      jcol = j+p+2
      aex1[jcol,irow] = gamma1[j]
      aex2[jcol,irow] = gamma2[j]
# Equation (60):
      if i > 1
        jcol = 1
        aex1[jcol,irow+j] =  gamma1[j]
        aex2[jcol,irow+j] = -gamma2[j]
      end
      jcol = p+2-j
      aex1[jcol,irow+j] =  gamma1[j]
      aex2[jcol,irow+j] = -gamma2[j]
      jcol = 2p+2
      aex1[jcol,irow+j] = -cone
      aex2[jcol,irow+j] =  czero
# Equation for r (59):
      jcol = 2
      aex1[jcol,irow+p+j] = -cone
      aex2[jcol,irow+p+j] =  czero
      jcol = 2p+3-j
      aex1[jcol,irow+p+j] = alpha[j]
      aex2[jcol,irow+p+j] = czero
      if i < n-1
        phi = beta_imag[j]*(t[i+2]-t[i+1])
        ebt = exp(-beta_real[j]*(t[i+2]-t[i+1]))
        jcol = 2p+3
        aex1[jcol,irow+p+j] =  ebt*cos(phi)
        aex2[jcol,irow+p+j] = -ebt*sin(phi)
      end
    end
  end
end

# Specify the number of bands below & above the diagonal:
m1 = p+1
m2 = p+1
# Set up matrix & vector needed for band-diagonal solver:
#al_small = zeros(Complex{Float64},nex,m1)
#indx::Vector{Int64} = collect(1:nex)
# Do the band-diagonal LU decomposition (indx is a permutation vector for
# pivoting; d gives the sign of the determinant based on the number of pivots): 
#@code_warntype bandec(aex1,aex2,nex,m1,m2,al_small1,al_small2,indx)
d=bandec_real2_trans(aex1,aex2,nex,m1,m2,al_small1,al_small2,indx)
# Solve the equation A^{-1} y = b using band-diagonal LU back-substitution on
# the extended equations: A_{ex}^{-1} y_{ex} = b_{ex}:
#banbks(aex,nex,m1,m2,al_small,indx,bex)
# Now select solution to compute the log likelihood:
# The equation A^{-1} y = b has been solved in bex (the extended vector).
# So, I need to pick out the b portion from bex, and take the dot product
# with y (which is the residuals of the data minus model, which is correlated
# noise that we are modeling with the multi-Lorentzian covariance function):
#log_like = 0.0
#for i=1:n
#  log_like += real(bex[(i-1)*(2p+1)+1])*y[i]
#end
# Convert this to log likelihood:
#log_like = -0.5*log_like
# Next compute the determinant of A_{ex}:
logdeta = czero
for i=1:nex
  logdeta += 0.5*log(aex1[1,i]*aex1[1,i]+aex2[1,i]*aex2[1,i])
end
#println("Log determinant of A_{ex}: ",logdeta)
# Add determinant to the likelihood function:
#log_like += -0.5*logdeta
#toc()
# Return the log likelihood:
#return log_like,logdeta
#return aex,al_small,indx,logdeta
return logdeta
end
