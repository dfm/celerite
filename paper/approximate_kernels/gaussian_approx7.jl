# Try to approximate squared-exponential kernel:
using PyPlot
using Optim
using Roots
using Polynomials
include("sturms_theorem.jl")

#function gaussian_approx6()

ntime = 1000
time = collect(linspace(0,10,ntime))
gaussian=exp(-time.^2/2.)

function gauss_model(x)
model = zeros(ntime)
for k=1:nexp
  if k > 1
    x[2+(k-1)*4]=0.0
    x[4+(k-1)*4]=0.0
  end
  model += x[1+(k-1)*4].*exp(-x[3+(k-1)*4].*time).*cos(x[4+(k-1)*4].*time)
  model += x[2+(k-1)*4].*exp(-x[3+(k-1)*4].*time).*sin(x[4+(k-1)*4].*time)
end
return model
end

function compute_chi(x)
model = gauss_model(x)
sum_coeff = zeros(x)
# Compute Taylor-series coefficients:
for k=1:nexp
  sum_coeff[1] += x[1+(k-1)*4]
  sum_coeff[2] +=-x[1+(k-1)*4]*x[3+(k-1)*4]+x[2+(k-1)*4]*x[4+(k-1)*4]
  sum_coeff[3] += x[1+(k-1)*4]*x[3+(k-1)*4]^2-2*x[2+(k-1)*4]*x[3+(k-1)*4]*x[4+(k-1)*4]-x[1+(k-1)*4]*x[4+(k-1)*4]^2
  sum_coeff[4] +=-x[1+(k-1)*4]*x[3+(k-1)*4]^3+3*x[2+(k-1)*4]*x[3+(k-1)*4]^2*x[4+(k-1)*4]+3*x[1+(k-1)*4]*x[3+(k-1)*4]*x[4+(k-1)*4]^2-x[2+(k-1)*4]*x[4+(k-1)*4]^3
end

return sum((gaussian-model).^2) +100.*((sum_coeff[1]-1.0)^2+sum_coeff[2]^2+(sum_coeff[3]+1.0)^2+sum_coeff[4]^2)+(sturms_theorem(x) > 0)*1e10
end  

nexp = 4
npar = 4*nexp

#atrial = [-0.5185744409672999,0.2854238724264553,1.3848025999389613,1.8339730386273703,63.11883856322811,-0.002585613714221086,1.479593361171615,-0.001569459359505545,-32.174013349165506,-0.00439276758185438,1.593669981824972,0.00035151389750538633,-29.42595216593894,-0.0013496416338785663,1.3889097102817822,0.002608575323797371]

#atrial = [-0.5185618444470842,0.28542350796810384,1.3848024403617407,1.8339737219232786,63.118858757909,-0.002595606253169336,1.4795927335503265,-0.001568656597234968,-32.17399733592503,-0.004866618683786094,1.5936706619228656,0.0015491916728168549,-29.425929677155498,-0.0013496810594145385,1.38891017342687,0.002608476149940659,0.0006171992813012984,0.001000720300704954,0.10003813140421611,0.099854716608336]
atrial = [-0.5185618444470842,0.28542350796810384,1.3848024403617407,1.8339737219232786,63.118858757909,-0.002595606253169336,1.4795927335503265,-0.001568656597234968,-32.17399733592503,-0.004866618683786094,1.5936706619228656,0.0015491916728168549,-29.425929677155498,-0.0013496810594145385,1.38891017342687,0.002608476149940659]

#chibest=sum((gaussian-model).^2)
chibest=compute_chi(atrial)

println("Initial chi-square: ",chibest)
read(STDIN,Char)

#result = optimize(compute_chi, atrial, BFGS(), OptimizationOptions(autodiff = true))
result = optimize(compute_chi, atrial)

println(Optim.minimizer(result),Optim.minimum(result))
abest= Optim.minimizer(result)
clf()
model = gauss_model(abest)
plot(time,gaussian)
plot(time,model)
plot(time,(gaussian-model))
chi=sum((gaussian-model).^2)
println("New minimum: ",chi," ",abest," ",std(gaussian-model)," ",maximum(abs(gaussian-model)))
read(STDIN,Char)
# Now, plot the power spectrum:
nfreq = 100000
#omega = logspace(-1,1,nfreq)
omega = logspace(-3,1,nfreq)
psd_grid = zeros(nfreq)
clf()
#plot(omega,psd_grid)
# Compute the asymptotic value:
for k=1:nexp
  a1 = abest[1+(k-1)*4]
  a2 = abest[2+(k-1)*4]
  b  = abest[3+(k-1)*4]
  om = abest[4+(k-1)*4]
  psdt = (a2*om.*(b^2 + om^2 - omega.^2) + a1*b.*(b^2 + om^2 + omega.^2))./
         (b^4 + (om^2 - omega.^2).^2 + 2*b^2.*(om^2 + omega.^2))
  psd_grid += psdt
end
clf()
plot(omega,psd_grid)
plot(omega,exp(-omega.^2./2)*sqrt(pi/2))
plot(omega,psd_grid-exp(-omega.^2./2)*sqrt(pi/2))
#return
#end
