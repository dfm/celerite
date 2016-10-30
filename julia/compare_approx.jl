using PyPlot

# Herein we compare the approximate kernel with the exact kernel.

ntime = 1000
time = collect(linspace(0,10,ntime))

kex = zeros(ntime,ntime)
kap = zeros(ntime,ntime)
beta = 3.0
w = 1.e-9
#nc = 4
#amp = [2.953823582534888+1.5e-3,-0.04961614737314587,-1.6900719772140012,-0.21413545794774125]
#b = [1.589510614489265,3.4729345651560872,1.8698492189074047,6.363451205766626]
#omega = [0.3329119766576696,4.5999414581901386,1.7115890219170917,2.205939216106191]
nc = 3
amp = [2.7518074963750734,-0.3546256365636662,-1.3968344706386124]
b = [1.5312499407621147,4.753504292338284,1.7967431129812566]
omega = [0.34148888178746056,2.042571639590954,1.7641684073510928]
# Matern 3/2 kernel:
#nc = 2
#amp = [-99.0,100.0]
#b = [1.0,0.99]
#omega = [0.0,0.0]
for i=1:ntime
  for j=1:ntime
    z= abs(time[i]-time[j])*beta
#    kex[i,j]=(1.0+z)*exp(-z)
    kex[i,j]=exp(-z^2/2.0)
#    kap[i,j]=2.670143*exp(-1.500258*z)*cos(0.346731*z)-
#             0.458705*exp(-4.125588*z)*cos(1.390912*z)-
#             1.211438*exp(-1.744602*z)*cos(1.798436*z)
     for k=1:nc
       kap[i,j] +=amp[k]*exp(-b[k]*z)*cos(omega[k]*z)
     end
  end
  kex[i,i]+=w
  kap[i,i]+=w
end
println("Determinant: ",det(kex))
eigex,eigvec = eig(kex)
println("Minimum eigenvalue: ",minimum(eigex))
eigap,eigvec = eig(kap)
println("Minimum eigenvalue: ",minimum(eigap))
# Cholesky decomposition (square root) of kernels:
sqrt_kex = chol(kex)
sqrt_kap = chol(kap)

# Random normal deviates:
y=randn(ntime)
# Compute correlated noise component:
corrnoise_ex=*(transpose(sqrt_kex),y);
corrnoise_ap=*(transpose(sqrt_kap),y);


plot(time,corrnoise_ex)
plot(time,corrnoise_ap,"r.")
