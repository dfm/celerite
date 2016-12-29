alpha = rand()+im*rand()
beta = rand()+im*rand()

# Compute covariance matrix:
nt = 11
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
