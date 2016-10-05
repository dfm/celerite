using Optim
using PyPlot
using StatsBase
using ForwardDiff
include("compile_matrix_symm.jl")
include("compute_likelihood.jl")
include("bandec_trans.jl")
include("banbks_trans.jl")
include("regress.jl")

function test_sp500()

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

  lower = log([1e-15, 1e-8,1e-8,1e-8,1e-8,1e-8, 1e-8,1e-8,1e-8,1e-8,1e-8, 1e-3,1e-3,1e-3,1e-3,1e-3])
  upper = log([1e4, 1e8,1e8,1e8,1e8,1e8, 1e4,1e4,1e4,1e4,1e4, 1e3,1e3,1e3,1e3,1e3])
#  result = optimize(log_like, xdiff, BFGS(), OptimizationOptions(autodiff = true))
#  result = optimize(DifferentiableFunction(log_like), xdiff, lower, upper, Fminbox(), optimizer=GradientDescent)
#  result = optimize(DifferentiableFunction(log_like), xdiff, lower, upper, Fminbox(), optimizer=NelderMead)
  result = optimize(DifferentiableFunction(log_like), xdiff, lower, upper, Fminbox(), optimizer=BFGS, optimizer_o= OptimizationOptions(autodiff = true))
#  result = optimize(DifferentiableFunction(log_like), xdiff, lower, upper, Fminbox(), optimizer=ConjugateGradient)
#  result = optimize(log_like, xdiff, BFGS())
#  result = optimize(log_like, x0, BFGS())
#  result = optimize(log_like, xdiff)
  println(result)
  return exp(Optim.minimizer(result)),Optim.minimum(result)
#  return result
end

# Read in S&P 500 data:

data = readdlm("table.csv",',')
#Date,Open,High,Low,Close,Volume,Adj Close
sp500 = Array{Float64}(vec(data[:,5]))
sp500 = log(sp500)
nt = length(sp500)
println("Length of data: ",length(sp500))
time = zeros(nt)
for i=1:nt
  time[i] = Date(data[i,1])-Date(data[nt,1]) 
end
plot(time,sp500)
read(STDIN,Char)

# Carry out linear regression:
ord = 1
fn = zeros(ord+1,nt)
tnorm = (time-time[1])/(time[nt]-time[1])
for i=1:ord+1
  fn[i,:]=tnorm.^(i-1)
end
sig = ones(nt)
coeff,cov = regress(fn,sp500,sig)
println("Coeff: ",coeff)
sp500_trend = zeros(nt)
sp500_sub = copy(sp500)
for i=1:nt
  for j=1:ord+1
    sp500_trend[i] += coeff[j]*fn[j,i]
  end
  sp500_sub[i] -= sp500_trend[i]
end

plot(time,sp500)
plot(time,sp500_trend)
plot(time,sp500_sub)
read(STDIN,Char)

# Compute ACF:
nt = length(time)
dt = median(time[2:nt]-time[1:nt-1])
lags = collect(0:1:200)
println(typeof(lags))
acf = autocov(sp500_sub,lags)
lags *=dt
clf()
# Now, model covariance:
acf_model = 3.5.*exp(-lags./100.).*cos(2.*pi.*lags)+0.5*cos(2.*pi*lags/0.5)
clf()
plot(lags,acf)
plot(lags,acf_model)
plot(lags,acf-acf_model)
#read(STDIN,Char)

# Initialize model with guessed parameters:

nperiod = 1
#period = logspace(log10(5.0),log10(5.0),nperiod)
#period = logspace(-1.0,2.0,nperiod)
period = logspace(0.0,0.0,nperiod)
#period = logspace(log10(1.1),log10(1.1),nperiod)
# Best fit from initial run with 3 kernels:
#[7.50113753448555e-14,0.052634997384683246,3.814476305265215,0.2990524739656785,0.43730650302365087,19.982885984516415,0.00012381766899862475,0.0001249074549875943,0.2638033900492329,6.285942400966896,12.5689911272956,0.2470381285466161]

# Best fit from initial run with 5 kernels.  ACF match looks terrible....  sp500_psd_v01.png
#New best fit- initial params: [-0.5986951906468994,-0.47390425458548036,0.9609731796508907,0.30004136703676076,-0.7217327410001237,0.5288721138262757,-0.16943295213003973,-0.012994821145197122,0.14289041773272704,-0.8402580004438254,-0.8430719194021692,0.4293014742447645,0.5908973910877733,-0.9694619555140389,-0.3467403161436997] best params: [0.7170661453521265,0.47776707889546643,0.7454845071892328,0.6907093846757348,0.40009162284219685,0.8568580106404585,0.0009290130419164398,1.234040420311936e-7,1.0270102480925339e-8,0.0004132060175645361,0.0003972692197700031,1.2044944615613637,1.572711606998557,0.3807616972409821,0.673615655353782] log like: -6.656152972205763e8

nkernel=2
xopt = Float64[]
log_like_opt = Inf

#xsave = -1.0+2.0*rand(15)
xsave = -1.0+2.0*rand(6)
nx = length(xsave)
for iperiod=1:nperiod
#  alpha   = [0.0526,3.814,0.3,0.5]
#  beta_real = [20.,1e-4,1e-4,0.26]
#  alpha   = [0.1,0.1,0.1,0.1]
#  beta_real = [1e-2,1e-2,1e-2,1e-2]
#  beta_imag = 2pi./[1.0,0.5,period[iperiod]]
  x0 = copy(xsave)
#  x0[nx]=log(2pi/period[iperiod])
  p = nkernel
  p0 = 1
  xbest,log_like_best = log_like_derivative_wrapper(p0,p,x0,time,sp500_sub)
  if log_like_best < log_like_opt
    println("New best fit- initial params: ",x0," best params: ",xbest," log like: ",log_like_best)
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
      psd[i] = 2.0*xbest[2]*xbest[nkernel+2]/(omega[i]^2+xbest[nkernel+2]^2)
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
