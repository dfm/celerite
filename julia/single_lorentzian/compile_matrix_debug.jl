# This is the real version of the calculation based on 8/18/16 notes.

function compile_matrix(alpha,beta_real,beta_imag,w,t,y)

# The vectors are arranged as:
# [x_k {{r^R_{k+1,i},r^I_{k+1,i}},i=1..p} {{l^R_{k+1,i},l^I_{k+1,i},i=1..p} x_{k+1} ...]
# For a total of (N-1)*(4(p-p0)+2p+1)+1 = N(4(p-p0)+2p+1)-4p equations.
# The equations are arranged as:
# 1). Equation 61 (real only; one single equation);
# 2). Equation 60 (real & imaginary; i=1..p);
# 3). Equation 59 (real & imagingary; i=1..p).

tic()
n = length(t)
p = length(alpha)
p0 = 0
czero = zero(eltype(alpha))
for i=1:p
  if beta_imag[i] == czero
    p0 += 1
  end
end
nex = (n-1)*(4(p-p0)+2p0+1)+1
width = 4p+5
aex = zeros(eltype(alpha),width,nex)
#aex_factor = fill("",nex,nex)
aex_factor = fill("",width,nex)
# First compile bex:
bex = zeros(eltype(alpha),nex)
# Make a string array to hold the variables:
variables = fill("",nex)
variables[1] = " x_1/2"
bex[1] = y[1]/2.0
for k=2:n
  for j=1:p0
    variables[(k-2)*(4(p-p0)+2p+1)+1+(j-1)*2+1] = string(" r^R_{",k,",",j,"}")
  end
  for j=1:(p-p0)
    variables[(k-2)*(4(p-p0)+2p+1)+1+(j-1)*2+1] = string(" r^R_{",k,",",j,"}")
    variables[(k-2)*(4(p-p0)+2p+1)+1+(j-1)*2+2] = string(" r^I_{",k,",",j,"}")
  end
  for j=1:p
    variables[(k-2)*(4(p-p0)+2p+1)+1+2p+(j-1)*2+1] = string(" l^R_{",k,",",j,"}")
    variables[(k-2)*(4(p-p0)+2p+1)+1+2p+(j-1)*2+2] = string(" l^I_{",k,",",j,"}")
  end
  variables[(k-1)*(4(p-p0)+2p+1)+1] = string(" x_",k)
  bex[(k-1)*(4(p-p0)+2p+1)+1] = y[k]/2.0
end
#for k=1:nex
#  println(k," ",variables[k])
#end
#read(STDIN,Char)

# Do the first row, eqn (61), which is a special case since l_1 = 0:
irow = 1
k = 1
d = sum(alpha)+w
#jcol = 1
jcol = 2p+3
# Factor multiplying x_1:
aex[jcol,irow]= d/2.0
aex_factor[jcol,irow] = " d/2 "
# Compile aex_factor & variables to get equations:
equations = fill("",nex)
equations[irow] = string(equations[irow],aex_factor[jcol,irow],variables[jcol])
one_type = one(eltype(alpha))
gamma_real = zeros(eltype(alpha),p)
gamma_imag = zeros(eltype(alpha),p)
ebt = czero
phi = czero
for j=1:p
  ebt = exp(-beta_real[j]*(t[k+1]-t[k]))
  phi = beta_imag[j]*(t[k+1]-t[k])
  gamma_real[j] =  ebt*cos(phi)
  gamma_imag[j] = -ebt*sin(phi)
# Factor multiplying r^R_{2,j}:
  jcol = 1+(j-1)*2+1
  jcol = jcol - irow + 2p+3
  aex[jcol,irow]= gamma_real[j]
  aex_factor[jcol,irow] = string(" +gamma^R_{1,",j,"} ")
  equations[irow] = string(equations[irow],aex_factor[jcol,irow],variables[jcol])
# Factor multiplying r^I_{2,j}:
  jcol = 1+(j-1)*2+2
  jcol = jcol - irow + 2p+3
  aex[jcol,irow]= gamma_imag[j]
  aex_factor[jcol,irow] = string(" +gamma^I_{1,",j,"} ")
  equations[irow] = string(equations[irow],aex_factor[jcol,irow],variables[jcol])
end
equations[irow] = string(equations[irow]," = b_1/2")
#println(irow,equations[irow])
#read(STDIN,Char)
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
  for j=1:p
# Real part of equation (60):
    irow = (k-2)*(4(p-p0)+2p+1) + 1 + (j-1)*2 + 1
# Factors multiplying (l^R_{k,j},l^I_{k,j})
    if k > 2
      jcol = (k-3)*(4(p-p0)+2p+1) + 1 + 2p + (j-1)*2 + 1
  jcol = jcol - irow + 2p+3
      aex[jcol,irow] = gamma_real_km1[j]
      aex_factor[jcol,irow] = string(" +gamma^R_{",k-1,",",j,"} ")
      equations[irow] = string(equations[irow],aex_factor[jcol,irow],variables[jcol])
      jcol = (k-3)*(4(p-p0)+2p+1) + 1 + 2p + (j-1)*2 + 2
  jcol = jcol - irow + 2p+3
      aex[jcol,irow] = gamma_imag_km1[j]
      aex_factor[jcol,irow] = string(" +gamma^I_{",k-1,",",j,"} ")
      equations[irow] = string(equations[irow],aex_factor[jcol,irow],variables[jcol])
    end
# Factor multiply x_k:
    jcol = (k-2)*(4(p-p0)+2p+1) + 1
  jcol = jcol - irow + 2p+3
    aex[jcol,irow] = gamma_real_km1[j]
    aex_factor[jcol,irow] = string(" +gamma^R_{",k-1,",",j,"} ")
    equations[irow] = string(equations[irow],aex_factor[jcol,irow],variables[jcol])
# Factor multipling l^R_{k+1,j}:
    jcol = (k-2)*(4(p-p0)+2p+1) + 1 + 2p + (j-1)*2 +1
  jcol = jcol - irow + 2p+3
    aex[jcol,irow] = -one_type
    aex_factor[jcol,irow] = string(" - ")
    equations[irow] = string(equations[irow],aex_factor[jcol,irow],variables[jcol]," = 0 ")
#    println(irow,equations[irow])
# Imaginary part of equation (60):
    irow = (k-2)*(4(p-p0)+2p+1) + 1 + (j-1)*2 + 2
# Factors multiplying (l^R_{k,j},l^I_{k,j}):
    if k > 2
      jcol = (k-3)*(4(p-p0)+2p+1) + 1 + 2p + (j-1)*2 + 1
  jcol = jcol - irow + 2p+3
      aex[jcol,irow] =  gamma_imag_km1[j]
      aex_factor[jcol,irow] = string(" +gamma^I_{",k-1,",",j,"} ")
      equations[irow] = string(equations[irow],aex_factor[jcol,irow],variables[jcol])
      jcol = (k-3)*(4(p-p0)+2p+1) + 1 + 2p + (j-1)*2 + 2
  jcol = jcol - irow + 2p+3
      aex[jcol,irow] = -gamma_real_km1[j]
      aex_factor[jcol,irow] = string(" -gamma^R_{",k-1,",",j,"} ")
      equations[irow] = string(equations[irow],aex_factor[jcol,irow],variables[jcol])
    end
# Factor multiply x_k:
    jcol = (k-2)*(4(p-p0)+2p+1) + 1
  jcol = jcol - irow + 2p+3
    aex[jcol,irow] = gamma_imag_km1[j]
    aex_factor[jcol,irow] = string(" +gamma^I_{",k-1,",",j,"} ")
    equations[irow] = string(equations[irow],aex_factor[jcol,irow],variables[jcol])
# Factor multiplying l^I_{k+1,j}:
    jcol = (k-2)*(4(p-p0)+2p+1) + 1 + 2p + (j-1)*2 + 2
  jcol = jcol - irow + 2p+3
    aex[jcol,irow] =  one_type
    aex_factor[jcol,irow] = string(" + ")
    equations[irow] = string(equations[irow],aex_factor[jcol,irow],variables[jcol]," = 0 ")
#    println(irow,equations[irow])
# Real part of equation (59):
    irow = (k-2)*(4(p-p0)+2p+1) + 1 + 2p + (j-1)*2 + 1
# Factor multiplying r^R_{k,j}:
    jcol = (k-2)*(4(p-p0)+2p+1) + 1 + (j-1)*2 + 1
  jcol = jcol - irow + 2p+3
    aex[jcol,irow] = -one_type
    aex_factor[jcol,irow] = string(" - ")
    equations[irow] = string(equations[irow],aex_factor[jcol,irow],variables[jcol])
# Factor multiplying x_k:
    jcol = (k-1)*(4(p-p0)+2p+1) + 1
  jcol = jcol - irow + 2p+3
    aex[jcol,irow] = 0.5*alpha[j]
    aex_factor[jcol,irow] = string(" + alpha_{",j,"}/2 ")
    equations[irow] = string(equations[irow],aex_factor[jcol,irow],variables[jcol])
# Factor multiplying r^R_{k+1,j}:
    if k < n
      jcol = (k-1)*(4(p-p0)+2p+1) + 1 + (j-1)*2 + 1
  jcol = jcol - irow + 2p+3
      aex[jcol,irow] = gamma_real[j]
      aex_factor[jcol,irow] = string(" +gamma^R_{",k,",",j,"} ")
      equations[irow] = string(equations[irow],aex_factor[jcol,irow],variables[jcol])
# Factor multiplying r^I_{k+1,j}:
      jcol = (k-1)*(4(p-p0)+2p+1) + 1 + (j-1)*2 + 2
  jcol = jcol - irow + 2p+3
      aex[jcol,irow] =  gamma_imag[j]
      aex_factor[jcol,irow] = string(" +gamma^I_{",k,",",j,"} ")
      equations[irow] = string(equations[irow],aex_factor[jcol,irow],variables[jcol])
    end
    equations[irow] = string(equations[irow]," = 0 ")
#    println(irow,equations[irow])
# Imaginary part of equation (59):
    irow = (k-2)*(4(p-p0)+2p+1) + 1 + 2p + (j-1)*2 + 2
# Factor multiplying r^I_{k,j}:
    jcol = (k-2)*(4(p-p0)+2p+1) + 1 + (j-1)*2 + 2
  jcol = jcol - irow + 2p+3
    aex[jcol,irow] =  one_type
    aex_factor[jcol,irow] = string(" + ")
    equations[irow] = string(equations[irow],aex_factor[jcol,irow],variables[jcol])
# Factor multiplying r^R_{k+1,j}:
    if k < n
      jcol = (k-1)*(4(p-p0)+2p+1) + 1 + (j-1)*2 + 1
  jcol = jcol - irow + 2p+3
      aex[jcol,irow] = gamma_imag[j]
      aex_factor[jcol,irow] = string(" +gamma^I_{",k,",",j,"} ")
      equations[irow] = string(equations[irow],aex_factor[jcol,irow],variables[jcol])
# Factor multiplying r^I_{k+1,j}:
      jcol = (k-1)*(4(p-p0)+2p+1) + 1 + (j-1)*2 + 2
  jcol = jcol - irow + 2p+3
      aex[jcol,irow] = -gamma_real[j]
      aex_factor[jcol,irow] = string(" -gamma^R_{",k,",",j,"} ")
      equations[irow] = string(equations[irow],aex_factor[jcol,irow],variables[jcol])
    end
    equations[irow] = string(equations[irow]," = 0 ")
  end
#  println(irow,equations[irow])
# Equation (61), only real:
  irow = (k-1)*(4(p-p0)+2p+1) + 1
# Factor multiplying x_k:
  jcol = (k-1)*(4(p-p0)+2p+1) + 1
  jcol = jcol - irow + 2p+3
  aex[jcol,irow] = d/2.0
  aex_factor[jcol,irow] = string(" +d/2 ")
  equations[irow] = string(equations[irow],aex_factor[jcol,irow],variables[jcol])
  for j=1:p
# Factor multiplying l^R_{k,j}:
    jcol = (k-2)*(4(p-p0)+2p+1) + 1 + 2p + (j-1)*2 + 1
  jcol = jcol - irow + 2p+3
    aex[jcol,irow] =  alpha[j]/2.0
    aex_factor[jcol,irow] = string(" +alpha_{",j,"} ")
    equations[irow] = string(equations[irow],aex_factor[jcol,irow],variables[jcol])
    if k < n
# Factor multiplying r^R_{k+1,j}:
      jcol = (k-1)*(4(p-p0)+2p+1) + 1 + (j-1)*2 + 1
  jcol = jcol - irow + 2p+3
      aex[jcol,irow]=  gamma_real[j]
      aex_factor[jcol,irow] = string(" +gamma^R_{",k,",",j,"} ")
      equations[irow] = string(equations[irow],aex_factor[jcol,irow],variables[jcol])
# Factor multiplying r^I_{k+1,j}:
      jcol = (k-1)*(4(p-p0)+2p+1) + 1 + (j-1)*2 + 2
  jcol = jcol - irow + 2p+3
      aex[jcol,irow]=  gamma_imag[j]
      aex_factor[jcol,irow] = string(" +gamma^I_{",k,",",j,"} ")
      equations[irow] = string(equations[irow],aex_factor[jcol,irow],variables[jcol])
    end
  end
  equations[irow] = string(equations[irow]," = b_",k,"/2")
#  println(irow,equations[irow])
  for j=1:p
    gamma_real_km1[j]=gamma_real[j]
    gamma_imag_km1[j]=gamma_imag[j]
  end
end

# Specify the number of bands below & above the diagonal:
m1 = 2p+2
m2 = 2p+2
# Set up matrix & vector needed for band-diagonal solver:
al_small = zeros(eltype(alpha),m1,nex)
indx = collect(1:nex)
# Do the band-diagonal LU decomposition (indx is a permutation vector for
# pivoting; d gives the sign of the determinant based on the number of pivots):
d=bandec_trans(aex,nex,m1,m2,al_small,indx)
# Solve the equation A^{-1} y = b using band-diagonal LU back-substitution on
# the extended equations: A_{ex}^{-1} y_{ex} = b_{ex}:
banbks_trans(aex,nex,m1,m2,al_small,indx,bex)
# Now select solution to compute the log likelihood:
# The equation A^{-1} y = b has been solved in bex (the extended vector).
# So, I need to pick out the b portion from bex, and take the dot product
# with y (which is the residuals of the data minus model, which is correlated
# noise that we are modeling with the multi-Lorentzian covariance function):
log_like = czero
logdetofa = czero
for i=1:n
  i0 = (i-1)*(4(p-p0)+2p+1)+1
  log_like += real(bex[i0])*y[i]
  logdetofa += log(2*abs(aex[1,i0]))
end
# Convert this to log likelihood:
log_like = -0.5*log_like
# Next compute the determinant of A_{ex}:
#for i=1:nex
#  println(i," ",2*aex[1,i])
#end
println("Log determinant of A_{ex}: ",logdetofa)
# Add determinant to the likelihood function:
log_like += -0.5*logdetofa
toc()
# Return the log likelihood:
#return log_like
return aex,bex,log_like,logdetofa

#return aex,bex,equations,variables
end
