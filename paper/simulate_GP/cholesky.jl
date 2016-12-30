alpha = rand()+im*rand()
beta = rand()+im*rand()
#beta = 0.0+im*rand()

# Compute covariance matrix:
nt = 5
t = linspace(0,nt-1,nt)

cov = zeros(Complex{Float64},nt,nt)
#cov = zeros(Float64,nt,nt)
for i=1:nt
  for j=i:nt
    cov[i,j]=alpha*exp(-beta*(t[j]-t[i]))
    cov[j,i]=conj(cov[i,j])
  end
  cov[i,i]=real(alpha)
end

ccov=chol(cov)

#covinv=inv(cov)
#
#for i=1:nt
#  for j=1:nt
#    if abs(covinv[i,j]) < eps()
#       covinv[i,j]=0.0+im*0.0
##       covinv[i,j]=0.0
#    end
#  end
#end
#
#covinvchol=chol(covinv)

# Check that cholesky decomposition is correct:
*(ctranspose(ccov),ccov)-cov

# Now try constructing:
a=zeros(Complex64,nt)
b=zeros(Complex64,nt)
v=zeros(Complex64,nt)
matrix = zeros(Complex64,nt,nt)
for i=1:nt
  a[i]=sqrt(alpha)*exp(beta*t[i])
  b[i]=sqrt(alpha)*exp(-beta*t[i])
  v[i]=b[i]/a[i]
end
d = Diagonal(a)
l = zeros(Complex64,nt,nt)
lam=zeros(Complex64,nt,nt)
for i=1:nt
  for j=1:i
    l[i,j]=1.0+im*0.0
  end
  if i < nt
    lam[i,i] = sqrt(v[i]-v[i+1])
  else
    lam[i,i] = sqrt(v[i])
  end
  for j=1:nt
    if j >= i
      matrix[i,j] = a[i]*b[j]
    else
      matrix[i,j] = a[j]*b[i]
    end
  end
end
cholm = *(d,*(l,lam))

