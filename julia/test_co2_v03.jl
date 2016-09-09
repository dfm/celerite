using Optim
using PyPlot
using StatsBase
using ForwardDiff
include("compile_matrix_symm.jl")
include("compute_likelihood.jl")
include("bandec_trans.jl")
include("banbks_trans.jl")
include("regress.jl")

function test_co2_v03()

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

# Read in CO2 data:

data = readdlm("CO2_data.csv.txt",',')
co2 = vec(data[:,2])
println("Length of data: ",length(co2))
time = vec(data[:,1] - data[1,1])
# Carry out quadratic regression:
nt = length(time)
ord = 2
fn = zeros(ord+1,nt)
tnorm = (time-time[1])/(time[nt]-time[1])
for i=1:ord+1
  fn[i,:]=tnorm.^(i-1)
end
sig = ones(nt)
coeff,cov = regress(fn,co2,sig)
println("Coeff: ",coeff)
co2_trend = zeros(nt)
co2_sub = copy(co2)
for i=1:nt
  for j=1:ord+1
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
#xsave = log([0.03684832413517181,0.4387613797192722,3.809611698603267,0.2993294592999687,0.0033183947487366045,0.5,0.36055491419151137,0.00013119968869186474,0.00015241021114397978,0.00010000000928728985,1e-3,6.285932185183963,12.569069334961931,18.8434942480683,2pi])
#xsave = log([0.03684832413517181, 0.4387613797192722,3.809611698603267, 0.36055491419151137,0.00013119968869186474, 6.285932185183963])
#New best fit- initial params: [0.24525060661563458,-0.6706280702467335,0.6135393449157531,0.5158110704105079,-0.19071204291502974,-0.8683608405275405,-0.4454421228919916,0.19497110540072837,-0.10251527601637767,-0.9098131451360563,-0.011106363961333798,-0.9510402952632626] best params: [0.04029466597429114,0.2526358743611703,0.2986196855835003,3.8223408614702126,0.19617575450045882,0.6614637313826826,0.0001429291061867777,0.00012651638000879924,1.2683737146520694e-6,12.569019942371668,6.285907554790972,0.21506018780384212] log like: 160.69228348034142
#New best fit- initial params: [-0.7235566627564012,0.6194689646767091,0.029787409423436273,0.06264056996187195,0.7721099787691372,-0.031346813435856546,0.9944816013289715,-0.9923234906574838,0.6680014200732902,0.05233606985509143,0.9914124992763558,0.8947403433841048,0.7972837525544665,-0.10271485834584615] best params: [2.6711800255620015e-6,0.43519427651205367,0.050770915734427255,0.29775630072792336,3.8047644172968242,0.2930869808770129,20.768233217196343,0.0001266517006615938,0.00012184198184990059,0.010015841478464087,12.568981432423138,6.2859299962988935,0.9999971834821783,1.0000011746976747] log like: 160.73525576118857
# New best fit- initial params: [0.061136562954420715,-0.05913570088861064,-0.6077859327932353,0.8006118980060903,0.05891929293106468,0.34715208605827863,0.17937202290847631,0.09639794885504882,-0.34528679613275415,0.5769921986616793,0.337034498127855,-0.5627620564531655,0.016106597035654868,-0.7390684897388655] best params: [3.163633940791601e-6,0.44345590215659636,3.8160116462380707,0.29897108013090246,0.04226615839014087,0.32474949169199874,0.0001351318599722061,0.00012927971290721362,10.842253788156388,6.28592852289604,12.56897388114301,17.48063508199799,0.9999990035091292,0.999998751695153] log like: 158.09364527030715  CO2_spectrum_v04.png
#New best fit- initial params: [0.19456431327663282,0.17777008451335163,0.8403990148156821,0.6839614249344197,-0.07929762858610623,0.7793808090360561,0.7073480572067927,0.14315807344378895,0.33877630747091825,0.4072966264675575,-0.6676937819100126,-0.21481808817726478] best params: [2.7957389459161393e-6,0.5766912572663334,3.8223871455915246,0.04749499292515523,0.2993418814033055,0.20558714343221735,0.00016680586922843604,22.476650723282553,0.0001476106178401725,6.2869384732932,0.010009637397587567,87.96822286229002] log like: 104.73308577578865  (Fewer data points to match Foreman-Mackey's model CO2_spectrum_v05.png)
#New best fit- initial params: [-0.8685768119401973,-0.2132623980155186,-0.07868415414696539,0.7476202152461235,-0.342931132412406,0.6347252166656303,0.8488001935291196,-0.2975105273303984,-0.6143468418867672,0.09299778736926534,-0.06292736152275547,-0.5110956767023276] best params: [3.654198299839067e-7,0.5814441799891235,0.30029024967528783,3.830771738177122,0.04055130349945212,0.22506337451478275,0.00015594402269633877,0.00017916252815285494,11.521068977115174,12.570037522503661,6.286957191746393,17.736327369565533] log like: 103.22857603764857 (Ditto, v06.png)




xsave = -1.0+2.0*rand(15)
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
  xbest,log_like_best = log_like_derivative_wrapper(p0,p,x0,time,co2_sub)
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
