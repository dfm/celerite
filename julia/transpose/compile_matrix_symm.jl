# This is the real version of the calculation based on 8/18/16 notes.

#function compile_matrix_symm(alpha,beta_real,beta_imag,w,t,y)
function compile_matrix_symm(alpha::Vector,beta_real::Vector,beta_imag::Vector, w::Real,t::Vector,
       nex::Int64,aex::Array,al_small::Array,indx::Vector{Int64})

# The vectors are arranged as:
# [x_k {{r^R_{k+1,i},r^I_{k+1,i}},i=1..p} {{l^R_{k+1,i},l^I_{k+1,i},i=1..p} x_{k+1} ...]
# For a total of (N-1)*(4(p-p0)+2p0+1)+1 = N(4(p-p0)+2p0+1)-4p equations.
# The equations are arranged as:
# 1). Equation 61 (real only; one single equation);
# 2). Equation 60 (real & imaginary; i=1..p);
# 3). Equation 59 (real & imagingary; i=1..p).

n = length(t)
p = length(alpha)
czero = zero(eltype(alpha))
p0 = 0
for i=1:p
  if beta_imag[i] == czero
    p0 += 1
  end
end
#nex = (n-1)*(4(p-p0)+2p0+1)+1
m1 = p0+2(p-p0+1)
width = 2m1+1
@assert(size(aex)==(width,nex))
@assert(size(al_small)==(m1,nex))
@assert(length(indx)==nex)
@assert(length(beta_real)==p)
@assert(length(beta_imag)==p)

#aex = zeros(eltype(alpha),width,nex)

# Do the first row, eqn (61), which is a special case since l_1 = 0:
irow = 1
k = 1
d = sum(alpha)+w
jcol0 = 1
# Factor multiplying x_1:
jcol = jcol0-irow + m1+1
aex[jcol,irow]= d/2.0
one_type = one(eltype(alpha))
gamma_real = zeros(eltype(alpha),p)
gamma_imag = zeros(eltype(alpha),p)
ebt = czero
phi = czero
for j=1:p0
  ebt = exp(-beta_real[j]*(t[k+1]-t[k]))
  gamma_real[j] =  ebt*cos(phi)
# Factor multiplying r^R_{2,j}:
  jcol0 = 1+j
  jcol = jcol0-irow + m1+1
  aex[jcol,irow]= gamma_real[j]
end
for j=(p0+1):p
  ebt = exp(-beta_real[j]*(t[k+1]-t[k]))
  phi = beta_imag[j]*(t[k+1]-t[k])
  gamma_real[j] =  ebt*cos(phi)
  gamma_imag[j] = -ebt*sin(phi)
# Factor multiplying r^R_{2,j}:
  jcol0 = 1+p0+(j-p0-1)*2+1
  jcol = jcol0-irow + m1+1
  aex[jcol,irow]= gamma_real[j]
# Factor multiplying r^I_{2,j}:
  jcol0 = 1+p0+(j-p0-1)*2+2
  jcol = jcol0-irow + m1+1
  aex[jcol,irow]= gamma_imag[j]
end
gamma_real_km1=copy(gamma_real)
gamma_imag_km1=copy(gamma_imag)
# Now, loop over the middle part of the matrix
for k=2:n
  if k < n
    for j=1:p
      ebt = exp(-beta_real[j]*(t[k+1]-t[k]))
      phi = beta_imag[j]*(t[k+1]-t[k])
      gamma_real[j] =  ebt*cos(phi)
      gamma_imag[j] = -ebt*sin(phi)
    end
  end
  for j=1:p0
# Real part of equation (60):
    irow = (k-2)*(4(p-p0)+2p0+1) + 1 + j
# Factors multiplying (l^R_{k,j})
    if k > 2
      jcol0 = (k-3)*(4(p-p0)+2p0+1) + 1 +2p-p0 + j
      jcol = jcol0-irow + m1+1
      aex[jcol,irow] = gamma_real_km1[j]
    end
# Factor multiply x_k:
    jcol0 = (k-2)*(4(p-p0)+2p0+1) + 1
    jcol = jcol0-irow + m1+1
    aex[jcol,irow] = gamma_real_km1[j]
# Factor multipling l^R_{k+1,j}:
    jcol0 = (k-2)*(4(p-p0)+2p0+1) + 1 +2p-p0 + j
    jcol = jcol0-irow + m1+1
    aex[jcol,irow] = -one_type
# Real part of equation (59):
    irow = (k-2)*(4(p-p0)+2p0+1) + 1 + 2p-p0 + j
# Factor multiplying r^R_{k,j}:
    jcol0 = (k-2)*(4(p-p0)+2p0+1) + 1 + j
    jcol = jcol0-irow + m1+1
    aex[jcol,irow] = -one_type
# Factor multiplying x_k:
    jcol0 = (k-1)*(4(p-p0)+2p0+1) + 1
    jcol = jcol0-irow + m1+1
    aex[jcol,irow] = 0.5*alpha[j]
# Factor multiplying r^R_{k+1,j}:
    if k < n
      jcol0 = (k-1)*(4(p-p0)+2p0+1) + 1 + j
      jcol = jcol0-irow + m1+1
      aex[jcol,irow] = gamma_real[j]
    end
  end
  for j=(p0+1):p
# Real part of equation (60):
    irow = (k-2)*(4(p-p0)+2p0+1) + 1 + p0 + (j-p0-1)*2 + 1
# Factors multiplying (l^R_{k,j},l^I_{k,j})
    if k > 2
      jcol0 = (k-3)*(4(p-p0)+2p0+1) + 1 + 2p-p0 + p0 + (j-p0-1)*2 + 1
      jcol = jcol0-irow + m1+1
      aex[jcol,irow] = gamma_real_km1[j]
      jcol0 = (k-3)*(4(p-p0)+2p0+1) + 1 + 2p-p0 + p0 + (j-p0-1)*2 + 2
      jcol = jcol0-irow + m1+1
      aex[jcol,irow] = gamma_imag_km1[j]
    end
# Factor multiply x_k:
    jcol0 = (k-2)*(4(p-p0)+2p0+1) + 1
    jcol = jcol0-irow + m1+1
    aex[jcol,irow] = gamma_real_km1[j]
# Factor multipling l^R_{k+1,j}:
    jcol0 = (k-2)*(4(p-p0)+2p0+1) + 1 + 2p-p0 + p0 + (j-p0-1)*2 +1
    jcol = jcol0-irow + m1+1
    aex[jcol,irow] = -one_type
# Imaginary part of equation (60):
    irow = (k-2)*(4(p-p0)+2p0+1) + 1 + p0 + (j-p0-1)*2 + 2
# Factors multiplying (l^R_{k,j},l^I_{k,j}):
    if k > 2
      jcol0 = (k-3)*(4(p-p0)+2p0+1) + 1 + 2p-p0 + p0 + (j-p0-1)*2 + 1
      jcol = jcol0-irow + m1+1
      aex[jcol,irow] =  gamma_imag_km1[j]
      jcol0 = (k-3)*(4(p-p0)+2p0+1) + 1 + 2p-p0 + p0 +  (j-p0-1)*2 + 2
      jcol = jcol0-irow + m1+1
      aex[jcol,irow] = -gamma_real_km1[j]
    end
# Factor multiply x_k:
    jcol0 = (k-2)*(4(p-p0)+2p0+1) + 1
    jcol = jcol0-irow + m1+1
    aex[jcol,irow] = gamma_imag_km1[j]
# Factor multiplying l^I_{k+1,j}:
    jcol0 = (k-2)*(4(p-p0)+2p0+1) + 1 + 2p-p0 + p0 + (j-p0-1)*2 + 2
    jcol = jcol0-irow + m1+1
    aex[jcol,irow] =  one_type
# Real part of equation (59):
    irow = (k-2)*(4(p-p0)+2p0+1) + 1 + 2p-p0 + p0 + (j-p0-1)*2 + 1
# Factor multiplying r^R_{k,j}:
    jcol0 = (k-2)*(4(p-p0)+2p0+1) + 1 + p0 + (j-p0-1)*2 + 1
    jcol = jcol0-irow + m1+1
    aex[jcol,irow] = -one_type
# Factor multiplying x_k:
    jcol0 = (k-1)*(4(p-p0)+2p0+1) + 1
    jcol = jcol0-irow + m1+1
    aex[jcol,irow] = 0.5*alpha[j]
# Factor multiplying r^R_{k+1,j}:
    if k < n
      jcol0 = (k-1)*(4(p-p0)+2p0+1) + 1 + p0 + (j-p0-1)*2 + 1
      jcol = jcol0-irow + m1+1
      aex[jcol,irow] = gamma_real[j]
# Factor multiplying r^I_{k+1,j}:
      jcol0 = (k-1)*(4(p-p0)+2p0+1) + 1 + p0 + (j-p0-1)*2 + 2
      jcol = jcol0-irow + m1+1
      aex[jcol,irow] =  gamma_imag[j]
    end
# Imaginary part of equation (59):
    irow = (k-2)*(4(p-p0)+2p0+1) + 1 + 2p-p0 + p0 + (j-p0-1)*2 + 2
# Factor multiplying r^I_{k,j}:
    jcol0 = (k-2)*(4(p-p0)+2p0+1) + 1 + p0 + (j-p0-1)*2 + 2
    jcol = jcol0-irow + m1+1
    aex[jcol,irow] =  one_type
# Factor multiplying r^R_{k+1,j}:
    if k < n
      jcol0 = (k-1)*(4(p-p0)+2p0+1) + 1 + p0 + (j-p0-1)*2 + 1
      jcol = jcol0-irow + m1+1
      aex[jcol,irow] = gamma_imag[j]
# Factor multiplying r^I_{k+1,j}:
      jcol0 = (k-1)*(4(p-p0)+2p0+1) + 1 + p0 + (j-p0-1)*2 + 2
      jcol = jcol0-irow + m1+1
      aex[jcol,irow] = -gamma_real[j]
    end
  end
# Equation (61), only real:
  irow = (k-1)*(4(p-p0)+2p0+1) + 1
# Factor multiplying x_k:
  jcol0 = (k-1)*(4(p-p0)+2p0+1) + 1
  jcol = jcol0-irow + m1+1
  aex[jcol,irow] = d/2.0
  for j=1:p0
# Factor multiplying l^R_{k,j}:
    jcol0 = (k-2)*(4(p-p0)+2p0+1) + 1 + 2p-p0 + j
    jcol = jcol0-irow + m1+1
    aex[jcol,irow] =  alpha[j]/2.0
    if k < n
# Factor multiplying r^R_{k+1,j}:
      jcol0 = (k-1)*(4(p-p0)+2p0+1) + 1 + j
      jcol = jcol0-irow + m1+1
      aex[jcol,irow]=  gamma_real[j]
    end
  end
  for j=(p0+1):p
# Factor multiplying l^R_{k,j}:
    jcol0 = (k-2)*(4(p-p0)+2p0+1) + 1 + 2p-p0 + p0+ (j-p0-1)*2 + 1
    jcol = jcol0-irow + m1+1
    aex[jcol,irow] =  alpha[j]/2.0
    if k < n
# Factor multiplying r^R_{k+1,j}:
      jcol0 = (k-1)*(4(p-p0)+2p0+1) + 1 + p0 + (j-p0-1)*2 + 1
      jcol = jcol0-irow + m1+1
      aex[jcol,irow]=  gamma_real[j]
# Factor multiplying r^I_{k+1,j}:
      jcol0 = (k-1)*(4(p-p0)+2p0+1) + 1 + p0 + (j-p0-1)*2 + 2
      jcol = jcol0-irow + m1+1
      aex[jcol,irow]=  gamma_imag[j]
    end
  end
  for j=1:p
    gamma_real_km1[j]=gamma_real[j]
    gamma_imag_km1[j]=gamma_imag[j]
  end
end
# Specify the number of bands below & above the diagonal:
m2 = m1
# Set up matrix & vector needed for band-diagonal solver:
#al_small = zeros(eltype(alpha),m1,nex)
#indx = collect(1:nex)
# Do the band-diagonal LU decomposition (indx is a permutation vector for
# pivoting; d gives the sign of the determinant based on the number of pivots):
d=bandec_trans(aex,nex,m1,m2,al_small,indx)
# Solve the equation A^{-1} y = b using band-diagonal LU back-substitution on
# the extended equations: A_{ex}^{-1} y_{ex} = b_{ex}:
#banbks_trans(aex,nex,m1,m2,al_small,indx,bex)
# Now select solution to compute the log likelihood:
# The equation A^{-1} y = b has been solved in bex (the extended vector).
# So, I need to pick out the b portion from bex, and take the dot product
# with y (which is the residuals of the data minus model, which is correlated
# noise that we are modeling with the multi-Lorentzian covariance function):
#log_like = 0.0
#logdetofa = czero
logdetofa = n*log(2.0)
for i=1:nex
  logdetofa += log(abs(aex[1,i]))
#  println(i," ",aex[1,i])
end
#for i=1:n
#  i0 = (i-1)*(4(p-p0)+2p0+1)+1
#  logdetofa += log(abs(aex[1,i0]))
#  println(i," ",i0," ",aex[1,i0])
#end
return logdetofa
end
