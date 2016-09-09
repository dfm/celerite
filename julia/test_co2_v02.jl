using Optim
using PyPlot
using StatsBase
using ForwardDiff
include("compile_matrix_symm.jl")
include("compute_likelihood.jl")
include("bandec_trans.jl")
include("banbks_trans.jl")
include("regress.jl")

function test_co2_v02()

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

  function log_like(x::Vector)
    w0 = exp(x[1])
    alpha = x[2:p+1]
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
  xdiff = copy(x)
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

  lower = [log(1e-15), 1e-8,1e-8,1e-8,1e-8,-3e0, log(1e-8),log(1e-8),log(1e-8),log(1e-8),log(1e-8), log(1e-3),log(1e-3),log(1e-3),log(1e-3)]
  upper = [log(1.0), 1e8,1e8,1e8,1e8,1e8, log(1e4),log(1e4),log(1e4),log(1e4),log(1e4), log(1e3),log(1e3),log(1e3),log(1e3)]
#  result = optimize(log_like, xdiff, BFGS(), OptimizationOptions(autodiff = true))
#  result = optimize(DifferentiableFunction(log_like), xdiff, lower, upper, Fminbox(), optimizer=GradientDescent)
  result = optimize(DifferentiableFunction(log_like), xdiff, lower, upper, Fminbox(), optimizer=BFGS)
#  result = optimize(log_like, xdiff, BFGS())
#  result = optimize(log_like, x0, BFGS())
#  result = optimize(log_like, xdiff)
  println(result)
  return Optim.minimizer(result),Optim.minimum(result)
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

nperiod = 10
#period = logspace(log10(5.0),log10(5.0),nperiod)
period = logspace(-0.5,0.5,nperiod)
#period = logspace(log10(1.1),log10(1.1),nperiod)
# Best fit from initial run with 3 kernels:
#[7.50113753448555e-14,0.052634997384683246,3.814476305265215,0.2990524739656785,0.43730650302365087,19.982885984516415,0.00012381766899862475,0.0001249074549875943,0.2638033900492329,6.285942400966896,12.5689911272956,0.2470381285466161]

nkernel=5
xopt = Float64[]
log_like_opt = Inf

# Best fit with 4 kernels:
# Results of Optimization Algorithm
#  * Algorithm: Fminbox with BFGS
#  * Starting Point: [-30.221136621610558,-2.944374030966639, ...]
#  * Minimizer: [-3.300945139395281,-0.8237995678563096, ...]
#  * Minimum: -4.816914e+02
#  * Iterations: 4
#  * Convergence: true
#    * |x - x'| < 1.0e-32: false
#    * |f(x) - f(x')| / |f(x)| < 1.0e-32: true
#    * |g(x)| < 1.0e-08: false
#    * Reached Maximum Number of Iterations: false
#  * Objective Function Calls: 636
#  * Gradient Calls: 636
# New best fit- initial period 3.3729337950720426 best params: [0.03684832413517181,0.4387613797192722,3.809611698603267,0.2993294592999687,0.0033183947487366045,0.36055491419151137,0.00013119968869186474,0.00015241021114397978,0.00010000000928728985,6.285932185183963,12.569069334961931,18.8434942480683] log like: -481.69138538434845

#xsave = log([7.50113753448555e-14,0.052634997384683246,3.814476305265215,0.2990524739656785,0.43730650302365087,19.982885984516415,0.00012381766899862475,0.0001249074549875943,0.2638033900492329,6.285942400966896,12.5689911272956,0.2470381285466161])
#xsave = log([7.50113753448555e-14,0.052634997384683246,3.814476305265215,0.2990524739656785,0.1,19.982885984516415,0.00012381766899862475,0.0001249074549875943,0.2638033900492329,6.285942400966896,12.5689911272956,0.2470381285466161])
xsave = log([0.03684832413517181,exp(0.4387613797192722),exp(3.809611698603267),exp(0.2993294592999687),exp(0.0033183947487366045),exp(0.01),0.36055491419151137,0.00013119968869186474,0.00015241021114397978,0.00010000000928728985,1e-3,6.285932185183963,12.569069334961931,18.8434942480683,1.0])
nx = length(xsave)
for iperiod=1:nperiod
#  alpha   = [0.0526,3.814,0.3,0.5]
#  beta_real = [20.,1e-4,1e-4,0.26]
#  alpha   = [0.1,0.1,0.1,0.1]
#  beta_real = [1e-2,1e-2,1e-2,1e-2]
#  beta_imag = 2pi./[1.0,0.5,period[iperiod]]
  x0 = copy(xsave)
  x0[nx]=log(2pi/period[iperiod])
  p = nkernel
  p0 = 1
  xbest,log_like_best = log_like_derivative_wrapper(p0,p,x0,time,co2_sub)
  if log_like_best < log_like_opt
    println("New best fit- initial period ",x0[nx]," best params: ",xbest," log like: ",log_like_best)
    log_like_opt = log_like_best
    xopt = xbest
# Plot the optimized ACF:
    acf_model = xbest[2].*exp(-lags.*xbest[nkernel+2])
    for j=2:nkernel
      acf_model += xbest[j+1]*exp(-lags.*xbest[nkernel+j+1]).*cos(lags.*xbest[2*nkernel+j]) 
    end
    acf_model[1] += xbest[1]
    clf()
    fig, axes = subplots(2,1)
    ax = axes[1,1]
    ax[:plot](lags,acf_model,"b-")
    ax[:plot](lags,acf,"r-")
    ax[:plot](lags,acf-acf_model,"g-")
    ax[:set_xlabel]("Time [yr]")
    ax[:set_ylabel]("Covariance [(ppm CO2)^2]")
#    read(STDIN,Char)
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
#    clf()
    ax = axes[2,1]
    ax[:loglog](omega,psd,"k-")
    ax[:set_xlabel]("Frequency/omega [rad/yr]")
    ax[:set_ylabel]("Power")
    read(STDIN,Char)
  end
end
return
end
