function lorentz_likelihood_real_band_init(alpha::Vector{Float64},beta_real::Vector{Float64},
                            beta_imag::Vector{Float64},w::Float64,t::Vector{Float64},
       nex::Int64,aex::Array{Float64,2},al_small::Array{Float64,2},indx::Vector{Int64})

#function lorentz_likelihood_real_band_init(alpha::Vector{Float64},beta_real::Vector{Float64},
#                            beta_imag::Vector{Float64},w::Float64,t::Vector{Float64})
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
n::Int64 = length(t)
atot::Float64 = sum(alpha)
#@assert(length(w) == n)
#@assert(length(y) == n)
@assert(length(beta_real) == p)
@assert(length(beta_imag) == p)
# There are 2*(p+1)+1=2p+3 sub-diagonals, 2*(p+1)+1=2p+3 super-diagonals + diagonal
# for a total of 4*p+7 non-zero diagonals:
#nex:: = 2*((2p+1)*n-2p)
#aex = zeros(Float64,nex,4*p+7)
#aex = zeros(Real,nex,4*p+7)
#gamma_cos = zeros(Float64,p)
gamma_cos = zeros(Real,p)
#gamma_sin = zeros(Float64,p)
gamma_sin = zeros(Real,p)
# Specify the number of bands below & above the diagonal:
m1 = 2*p+3
m2 = 2*p+3

for i=1:n
# Compute actual indices:
  irow =2*(i-1)*(1+2p)+1
#  bex[irow] = y[i]
# Diagonal noise:
  jcol = m1+1
# Each element is now a 2x2 matrix:
  aex[irow  ,jcol] = w + atot
  aex[irow+1,jcol] = w + atot
  for j=1:p
# Equation (61):
    if i > 1
      jcol = 2*j+2
      aex[irow  ,jcol] = alpha[j]
      aex[irow+1,jcol] = alpha[j]
    end
    if i < n
      ebt = exp(-beta_real[j]*(t[i+1]-t[i]))
      phi = beta_imag[j]*(t[i+1]-t[i])
      gamma_cos[j] = ebt*cos(phi)
      gamma_sin[j] = ebt*sin(phi)
#      println(gamma[j])
      jcol = 2*(j+p+1)+2
      aex[irow  ,jcol  ] =  gamma_cos[j]
      aex[irow  ,jcol+1] = -gamma_sin[j]
      aex[irow+1,jcol-1] =  gamma_sin[j]
      aex[irow+1,jcol  ] =  gamma_cos[j]
# Equation (60):
      if i > 1
        jcol = 2
        aex[irow+2*j  ,jcol  ] =  gamma_cos[j]
        aex[irow+2*j  ,jcol+1] =  gamma_sin[j]
        aex[irow+2*j+1,jcol-1] = -gamma_sin[j]
        aex[irow+2*j+1,jcol  ] =  gamma_cos[j]
      end
      jcol = 2*(p+1-j)+2
      aex[irow+2*j  ,jcol  ] =  gamma_cos[j]
      aex[irow+2*j  ,jcol+1] =  gamma_sin[j]
      aex[irow+2*j+1,jcol-1] = -gamma_sin[j]
      aex[irow+2*j+1,jcol  ] =  gamma_cos[j]
      jcol = 4p+4
      aex[irow+2*j  ,jcol] = -1
      aex[irow+2*j+1,jcol] = -1
# Equation for r (59):
      jcol = 4
      aex[irow+2*(p+j)  ,jcol] = -1
      aex[irow+2*(p+j)+1,jcol] = -1
      jcol = 4p+6-2j
      aex[irow+2*(p+j)  ,jcol] = alpha[j]
      aex[irow+2*(p+j)+1,jcol] = alpha[j]
      if i < n-1
        ebt = exp(-beta_real[j]*(t[i+2]-t[i+1]))
        phi = beta_imag[j]*(t[i+2]-t[i+1])
        gamma_cos[j] = ebt*cos(phi)
        gamma_sin[j] = ebt*sin(phi)
        jcol = 4p+6
        aex[irow+2*(p+j)  ,jcol  ] =  gamma_cos[j]
        aex[irow+2*(p+j)  ,jcol+1] = -gamma_sin[j]
        aex[irow+2*(p+j)+1,jcol-1] =  gamma_sin[j]
        aex[irow+2*(p+j)+1,jcol  ] =  gamma_cos[j]
      end
    end
  end
end

# Set up matrix & vector needed for band-diagonal solver:
#al_small = zeros(Float64,nex,m1)
#al_small = zeros(Real,nex,m1)
#indx = collect(1:nex)
# Do the band-diagonal LU decomposition (indx is a permutation vector for
# pivoting; d gives the sign of the determinant based on the number of pivots): 
d=bandec(aex,nex,m1,m2,al_small,indx)
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
logdeta = 0.0
for i=1:nex
  logdeta += log(abs(aex[i,1]))
end
#println("Log determinant of A_{ex}: ",logdeta)
# Add determinant to the likelihood function:
#log_like += -0.5*logdeta
#toc()
# Return the log likelihood:
#return log_like,logdeta
return aex,al_small,indx,logdeta
end


function lorentz_likelihood_real_band_init(alpha::Vector,beta_real::Vector,
                            beta_imag::Vector,w::Real,t::Vector,
       nex::Int64,aex,al_small,indx::Vector{Int64})

#function lorentz_likelihood_real_band_init(alpha::Vector{Float64},beta_real::Vector{Float64},
#                            beta_imag::Vector{Float64},w::Float64,t::Vector{Float64})
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
n::Int64 = length(t)
atot::eltype(alpha) = sum(alpha)
#@assert(length(w) == n)
#@assert(length(y) == n)
@assert(length(beta_real) == p)
@assert(length(beta_imag) == p)
# There are 2*(p+1)+1=2p+3 sub-diagonals, 2*(p+1)+1=2p+3 super-diagonals + diagonal
# for a total of 4*p+7 non-zero diagonals:
#nex:: = 2*((2p+1)*n-2p)
#aex = zeros(Float64,nex,4*p+7)
#aex = zeros(Real,nex,4*p+7)
#gamma_cos = zeros(Float64,p)
gamma_cos = zeros(Real,p)
#gamma_sin = zeros(Float64,p)
gamma_sin = zeros(Real,p)
# Specify the number of bands below & above the diagonal:
m1 = 2*p+3
m2 = 2*p+3

for i=1:n
# Compute actual indices:
  irow =2*(i-1)*(1+2p)+1
#  bex[irow] = y[i]
# Diagonal noise:
  jcol = m1+1
# Each element is now a 2x2 matrix:
  aex[irow  ,jcol] = w + atot
  aex[irow+1,jcol] = w + atot
  for j=1:p
# Equation (61):
    if i > 1
      jcol = 2*j+2
      aex[irow  ,jcol] = alpha[j]
      aex[irow+1,jcol] = alpha[j]
    end
    if i < n
      ebt = exp(-beta_real[j]*(t[i+1]-t[i]))
      phi = beta_imag[j]*(t[i+1]-t[i])
      gamma_cos[j] = ebt*cos(phi)
      gamma_sin[j] = ebt*sin(phi)
#      println(gamma[j])
      jcol = 2*(j+p+1)+2
      aex[irow  ,jcol  ] =  gamma_cos[j]
      aex[irow  ,jcol+1] = -gamma_sin[j]
      aex[irow+1,jcol-1] =  gamma_sin[j]
      aex[irow+1,jcol  ] =  gamma_cos[j]
# Equation (60):
      if i > 1
        jcol = 2
        aex[irow+2*j  ,jcol  ] =  gamma_cos[j]
        aex[irow+2*j  ,jcol+1] =  gamma_sin[j]
        aex[irow+2*j+1,jcol-1] = -gamma_sin[j]
        aex[irow+2*j+1,jcol  ] =  gamma_cos[j]
      end
      jcol = 2*(p+1-j)+2
      aex[irow+2*j  ,jcol  ] =  gamma_cos[j]
      aex[irow+2*j  ,jcol+1] =  gamma_sin[j]
      aex[irow+2*j+1,jcol-1] = -gamma_sin[j]
      aex[irow+2*j+1,jcol  ] =  gamma_cos[j]
      jcol = 4p+4
      aex[irow+2*j  ,jcol] = -1
      aex[irow+2*j+1,jcol] = -1
# Equation for r (59):
      jcol = 4
      aex[irow+2*(p+j)  ,jcol] = -1
      aex[irow+2*(p+j)+1,jcol] = -1
      jcol = 4p+6-2j
      aex[irow+2*(p+j)  ,jcol] = alpha[j]
      aex[irow+2*(p+j)+1,jcol] = alpha[j]
      if i < n-1
        ebt = exp(-beta_real[j]*(t[i+2]-t[i+1]))
        phi = beta_imag[j]*(t[i+2]-t[i+1])
        gamma_cos[j] = ebt*cos(phi)
        gamma_sin[j] = ebt*sin(phi)
        jcol = 4p+6
        aex[irow+2*(p+j)  ,jcol  ] =  gamma_cos[j]
        aex[irow+2*(p+j)  ,jcol+1] = -gamma_sin[j]
        aex[irow+2*(p+j)+1,jcol-1] =  gamma_sin[j]
        aex[irow+2*(p+j)+1,jcol  ] =  gamma_cos[j]
      end
    end
  end
end

# Set up matrix & vector needed for band-diagonal solver:
#al_small = zeros(Float64,nex,m1)
#al_small = zeros(Real,nex,m1)
#indx = collect(1:nex)
# Do the band-diagonal LU decomposition (indx is a permutation vector for
# pivoting; d gives the sign of the determinant based on the number of pivots): 
d=bandec(aex,nex,m1,m2,al_small,indx)
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
logdeta = 0.0
for i=1:nex
  logdeta += log(abs(aex[i,1]))
end
#println("Log determinant of A_{ex}: ",logdeta)
# Add determinant to the likelihood function:
#log_like += -0.5*logdeta
#toc()
# Return the log likelihood:
#return log_like,logdeta
return aex,al_small,indx,logdeta
end
