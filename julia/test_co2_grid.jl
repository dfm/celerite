using Optim
using PyPlot
using StatsBase
using ForwardDiff
include("compile_matrix_symm.jl")
include("compute_likelihood.jl")
include("bandec_trans.jl")
include("banbks_trans.jl")
include("regress.jl")

function test_co2_grid()

function log_like_derivative_wrapper(p0,p,x,t,y)
  n = length(t)
  nex = (4(p-p0)+2p0+1)*(n-1)+1
  m1 = 2(p-p0)+p0+2
  if p0 == p
    m1 = p0+ 1
  end
  width = 2m1+1
  indx::Vector{Int64} = collect(1:nex)
#  aex = zeros(Number,width,nex)
#  al_small= zeros(Number,m1,nex)
#  bex = zeros(Number,nex)
  nx = length(x)
  x0 = copy(x)
  xvary=x[1:nx-1]
  function log_like(xvary::Vector)
    x = [xvary;x0[nx]]
    w0 = exp(x[1])
    alpha = exp(x[2:p+1])
    beta_real = exp(x[p+2:2p+1])
    beta_imag = zeros(eltype(x),p)
    if p > p0
      beta_imag[p0+1:p] = exp(x[2p+2:3p-p0+1])
    end
    aex = zeros(eltype(x),width,nex)
    al_small= zeros(eltype(x),m1,nex)
    bex = zeros(eltype(x),nex)
    logdeta = compile_matrix_symm(alpha,beta_real,beta_imag,w0,t,nex,aex,al_small,indx)
    log_like = compute_likelihood(p,p0,y,aex,al_small,indx,logdeta,bex)
#    println("alpha: ",alpha," beta_real: ",beta_real," white: ",w0," log_like: ",log_like)
    return -log_like
  end

## Optimize the likelihood:
  xdiff = copy(xvary)
## Now perturb x:
#  dlogdx = zeros(eltype(x),length(x))
#  log_like0=log_like(x)
#  println(log_like0,x)
#  log_like0=log_like(x)
#  x0 = copy(x)
#  for i=1:length(x)
#    x    = copy(x0)
#    if (x[i] == 0.) then
#      h = 1e-6
#    else
#      h = 1e-6*x[i]
#    end
#    x[i] = x[i]+h
#    println(i," ",x,size(x))
#    log_like_h = log_like(x)
#    dlogdx[i]=(log_like_h-log_like0)/h
#  end
#  result=ForwardDiff.gradient(log_like,xdiff)
#  result=ForwardDiff.gradient(log_like,xdiff)
#  println("Gradient AutoDiff:    ",result)
#  println("Gradient Finite diff: ",dlogdx)

  result = optimize(log_like, xdiff, BFGS(), OptimizationOptions(autodiff = true))
#  result = optimize(log_like, xdiff, BFGS())
#  result = optimize(log_like, x0, BFGS())
#  result = optimize(log_like, xdiff)
  println(result)
  xopt = x0
  xopt[1:nx-1]=exp(Optim.minimizer(result))
  return xopt,Optim.minimum(result)
#  return result
end

# Read in CO2 data:

data = readdlm("CO2_data.csv.txt",',')
co2 = vec(data[:,2])
time = vec(data[:,1] - data[1,1])
# Carry out cubic regression:
nt = length(time)
fn = zeros(4,nt)
tnorm = (time-time[1])/(time[nt]-time[1])
fn[1,:]=1.0
fn[2,:]= tnorm
fn[3,:]= tnorm.*tnorm
fn[4,:]= tnorm.*tnorm.*tnorm
sig = ones(nt)
coeff,cov = regress(fn,co2,sig)
println("Coeff: ",coeff)
co2_trend = zeros(nt)
co2_sub = copy(co2)
for i=1:nt
  for j=1:4
    co2_trend[i] += coeff[j]*fn[j,i]
  end
  co2_sub[i] -= co2_trend[i]
end

plot(time,co2)
plot(time,co2_trend)
plot(time,co2_sub)
#read(STDIN,Char)

# Compute ACF:
nt = length(time)
dt = median(time[2:nt]-time[1:nt-1])
lags = collect(0:1:200)
println(typeof(lags))
acf = autocov(co2_sub,lags)
lags *=dt
clf()
# Now, model covariance:
acf_model = 3.5.*exp(-lags./50.).*cos(2.*pi.*lags)+0.5*cos(2.*pi*lags/0.5)
clf()
plot(lags,acf-acf_model)
#read(STDIN,Char)

# Initialize model with guessed parameters:

nkermax = 5
nperiod = 200
alpha   = [1.0]
beta_real = [0.01]
beta_imag = Float64[]
period = logspace(-1,1,nperiod)
log_like_period = zeros(nperiod)
xbest = Float64[]
ibest = -1
for nkernel=2:nkermax
  alpha = [alpha;1.0]
  beta_real = [beta_real;0.01]
  log_like_best = Inf
  for iperiod=1:nperiod
# Now, insert into the parameter vector:
    x0 = [log(1.0);log(alpha);log(beta_real);log(beta_imag);log(2pi/period[iperiod])]
    nx = length(x0)
    p = nkernel-1
    p0 = 1
# Vary all parameters except the last one (frequency):
# x,best_log_like = log_like_derivative_wrapper(p0,p,x0,time,co2_sub)
    xopt,log_like_opt = log_like_derivative_wrapper(p0,p,x0,time,co2_sub)
    log_like_period[iperiod]=log_like_opt
    if log_like_opt < log_like_best
       log_like_best = copy(log_like_opt)
       xbest = copy(xopt)
       ibest = iperiod
    end
  end
  clf()
  semilogx(period,log_like_period)
  print(ibest," ",xbest," ",log_like_best)
  beta_imag = [beta_imag;2pi/period[ibest]]
  read(STDIN,Char)
end
#println(x0,x,best_log_like)
println(xbest)
# Plot the optimized ACF:
acf_model = xbest[2].*exp(-lags.*xbest[nkernel+2])
for j=2:nkernel
  acf_model += xbest[j+1]*exp(-lags.*xbest[nkernel+j+1]).*cos(lags.*xbest[2*nkernel+j]) 
end
#+xbest[4]*exp(-lags.*xbest[8]).*cos(lags.*xbest[11])+xbest[5]*exp(-lags.*xbest[9]).*cos(lags.*xbest[12])
acf_model[1] += xbest[1]
clf()
plot(lags,acf_model)
plot(lags,acf)
plot(lags,acf-acf_model)
read(STDIN,Char)
# Plot the power spectrum:
nfreq = 1000
omega = collect(logspace(log10(0.001),log10(1000.),nfreq))
psd = zeros(nfreq)
for i=1:nfreq
  psd[i] = 2.0*xbest[2]*xbest[6]/(omega[i]^2+xbest[6]^2)
  for j=2:nkernel
    psd[i] += 2.0*xbest[j+1]*xbest[nkernel+j+1]/((omega[i]-xbest[2*nkernel+j])^2+xbest[nkernel+j+1]^2)
  end
end
clf()
loglog(omega,psd)
read(STDIN,Char)

return
end
