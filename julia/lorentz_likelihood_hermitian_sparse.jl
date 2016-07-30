function lorentz_likelihood_hermitian_sparse(alpha::Vector{Float64},beta::Vector{Complex{Float64}},
                            w::Float64,t::Vector{Float64},y::Vector{Float64})
# Computes the likelihood of a sum of Exponential/Cosine kernels (which have a
# Lorentzian power spectrum) utilizing the approach of Ambikasaran (2015)
#   Numer. Linear Algebra Appl. 2015; 22:1102-1114 DOI: 10.1002/nla
# Uses (2.30) from Rasmussen & Williams (2006) to compute the log likelihood.
# The form of the kernel is:
#
#   K(t_i,t_j) = w_i \delta_{ij} + \sum_{k=1}^{p} \alpha_k \exp{-\beta_k |t_i-t_j|}
#
# Note: if a \beta_k is complex, then its complex conjugate *must* be included as well.
#
# First, define band-diagonal matrix for kernel:
# [x_i,{r_{i,1},...,r_{i,p}},{l_{i,1},...,l_{i,p}}]

p = length(alpha)
n = length(t)
atot = sum(alpha)
#@assert(length(w) == n)
@assert(length(y) == n)
@assert(length(beta) == p)
#aex = Array(Complex{Float64},(2p+1)*n-2p,2*p+3)
nex = (2p+1)*n-2p
#aex = Array(Complex{Float64},(2p+1)*n-2p,(2p+1)*n-2p)
#aex = zeros(Complex{Float64},(2p+1)*n-2p,(2p+1)*n-2p)
#bex = zeros(Complex{Float64},(2p+1)*n-2p)
gamma = zeros(Complex{Float64},p)
ib=Int64[]
bex=Float64[]
iarow=Int64[]
iacol=Int64[]
aex=Complex{Float64}[]
# I'm going to store this as a sparse matrix:
for i=1:n
# Compute actual indices:
  irow =(i-1)*(1+2p)+1
#  bex[irow] = y[i]
  push!(ib,irow)
  push!(bex,y[i])
# Diagonal noise:
#  aex[irow,irow] = w + atot
  push!(iarow,irow)
  push!(iacol,irow)
  push!(aex,complex(w+atot,0.0))
  for j=1:p
    if i < n
#      bex[irow+  j] = 0.0
#      bex[irow+p+j] = 0.0
    end
# Equation (61):
    if i > 1
#      aex[irow,irow-p-1+j] = alpha[j]
      push!(iarow,irow)
      push!(iacol,irow-p-1+j)
      push!(aex,complex(alpha[j],0.0))
    end
    if i < n
      phi = imag(beta[j])*(t[i+1]-t[i])
      gamma[j] = exp(-real(beta[j])*(t[i+1]-t[i]))*complex(cos(phi),-sin(phi))
#      println(gamma[j])
#      aex[irow,irow+j] = gamma[j]
      push!(iarow,irow)
      push!(iacol,irow+j)
      push!(aex,gamma[j])
# Equation (60):
      if i > 1
#        aex[irow+j,irow-p-1+j] = complex(real(gamma[j]),-imag(gamma[j]))
        push!(iarow,irow+j)
        push!(iacol,irow-p-1+j)
        push!(aex, complex(real(gamma[j]),-imag(gamma[j])))
      end
#      aex[irow+j,irow] = complex(real(gamma[j]),-imag(gamma[j]))
      push!(iarow,irow+j)
      push!(iacol,irow)
      push!(aex, complex(real(gamma[j]),-imag(gamma[j])))
#      aex[irow+j,irow+p+j] = -1
      push!(iarow,irow+j)
      push!(iacol,irow+p+j)
      push!(aex, complex(-1.0,0.0))
# Equation for r (59):
#      aex[irow+p+j,irow+j] = -1
      push!(iarow,irow+p+j)
      push!(iacol,irow+j)
      push!(aex, complex(-1.0,0.0))
#      aex[irow+p+j,irow+2p+1] = alpha[j]
      push!(iarow,irow+p+j)
      push!(iacol,irow+2p+1)
      push!(aex, complex(alpha[j],0.0))
      if i < n-1
        phi = imag(beta[j])*(t[i+2]-t[i+1])
#        aex[irow+p+j,irow+2p+1+j] = exp(-real(beta[j])*(t[i+2]-t[i+1]))*complex(cos(phi),-sin(phi))
        push!(iarow,irow+p+j)
        push!(iacol,irow+2p+1+j)
        push!(aex, exp(-real(beta[j])*(t[i+2]-t[i+1]))*complex(cos(phi),-sin(phi)))
      end
    end
  end
end
aex = sparse(iarow,iacol,aex,nex,nex)
bex = sparsevec(ib,bex,nex)
#return log_likelihood
return aex,bex
end
