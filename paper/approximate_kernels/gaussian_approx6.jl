# Try to approximate squared-exponential kernel:
using PyPlot
using Optim
using Roots

function psd(abest)
nfreq = 1000
#omega = logspace(-1,1,nfreq)
omega = logspace(0,1,nfreq)
psd = zeros(nfreq)
# Compute the asymptotic value:
coeff_inf = 0.0
for k=1:nexp
  a1 = abest[1+(k-1)*4]
  a2 = abest[2+(k-1)*4]
  b  = abest[3+(k-1)*4]
  om = abest[4+(k-1)*4]
  psdt = (a2*om.*(b^2 + om^2 - omega.^2) + a1*b.*(b^2 + om^2 + omega.^2))./
         (b^4 + (om^2 - omega.^2).^2 + 2*b^2.*(om^2 + omega.^2))
  psd += psdt
end
return psd
end

#function gaussian_approx6()

ntime = 100
time = collect(linspace(0,10,ntime))
gaussian=exp(-time.^2/2.)

function gauss_model(x)
model = zeros(ntime)
for k=1:nexp
  model += x[1+(k-1)*4].*exp(-x[3+(k-1)*4].*time).*cos(x[4+(k-1)*4].*time)
  model += x[2+(k-1)*4].*exp(-x[3+(k-1)*4].*time).*sin(x[4+(k-1)*4].*time)
end
return model
end

function compute_chi(x)
  function dpsd_domega(omega)
# Let the kernel components be:
# a1*Exp[-b*t]*Cos[omega_0*t] + a2*Exp[-b*t]*Sin[omega_0*t]
# x = [{a1,a2,b,omega_0}_1; {a1,a2,b,omega_0}_2; {a1,a2,b,omega_0}_3; ... ]
#
  dpsd = zero(Real)
  for i=1:nexp
    a1 = x[(i-1)*4+1]
    a2 = x[(i-1)*4+2]
    b  = x[(i-1)*4+3]
    om0 = x[(i-1)*4+4]
    dpsd += (-2*sqrt(2/pi)*omega*(a1*b*(b^4 - 3*om0^4 + 2*om0^2*omega^2 + omega^4 - 2*b^2*(om0^2 - omega^2)) -
       a2*om0*(-3*b^4 + (om0^2 - omega^2)^2 - 2*b^2*(om0^2 + omega^2))))/
       ((b^2 + (om0 - omega)^2)^2*(b^2 + (om0 + omega)^2)^2)
  end
  return dpsd
  end

  function psd_func(omega)
  psd = 0.
  for k=1:nexp
    a1 = x[1+(k-1)*4]
    a2 = x[2+(k-1)*4]
    b  = x[3+(k-1)*4]
    om = x[4+(k-1)*4]
    psdt = (a2*om*(b^2 + om^2 - omega^2) + a1*b*(b^2 + om^2 + omega^2))/
         (b^4 + (om^2 - omega^2)^2 + 2*b^2*(om^2 + omega^2))
    psd += psdt
  end
  return psd
  end

model = gauss_model(x)
sum_coeff = zeros(x)
# Compute Taylor-series coefficients:
for k=1:nexp
  sum_coeff[1] += x[1+(k-1)*4]
  sum_coeff[2] +=-x[1+(k-1)*4]*x[3+(k-1)*4]+x[2+(k-1)*4]*x[4+(k-1)*4]
  sum_coeff[3] += x[1+(k-1)*4]*x[3+(k-1)*4]^2-2*x[2+(k-1)*4]*x[3+(k-1)*4]*x[4+(k-1)*4]-x[1+(k-1)*4]*x[4+(k-1)*4]^2
  sum_coeff[4] +=-x[1+(k-1)*4]*x[3+(k-1)*4]^3+3*x[2+(k-1)*4]*x[3+(k-1)*4]^2*x[4+(k-1)*4]+3*x[1+(k-1)*4]*x[3+(k-1)*4]*x[4+(k-1)*4]^2-x[2+(k-1)*4]*x[4+(k-1)*4]^3
end
  
#minpsd = minimum(psd(x))
#omega_min = fzero(dpsd_domega,[1e-3,1e3])
#minpsd = psd_func(omega_min)
result = optimize(psd_func,1.0,10.0,Brent())
minpsd = psd_func(result.minimum)
println("PSD minimum: ",minpsd," ",result.minimum)

return sum((gaussian-model).^2) +100.*((sum_coeff[1]-1.0)^2+sum_coeff[2]^2+(sum_coeff[3]+1.0)^2+sum_coeff[4]^2)+(minpsd < 0)*abs(minpsd)*1e5
end  

# The number of exponentials is actually +1:
nexp = 4
npar = 4*nexp
#abest=zeros(npar)

#atrial = zeros(npar)
#for k=1:npar
#  atrial[k] = abest[k]
#end
#atrial[1:8]=0.5
#atrial = [-0.2967663878261003,0.25057478717710174,1.3827297351466428,1.702023384832311,-0.2967663878261003,0.25057478717710174,1.3827297351466428,1.702023384832311,3.0791800790217154,0.0,1.2400386725361678,0.0,-0.3561416933055925,0.0,2.801219489380058,0.0,-1.1290392201099397,0.0,1.0162289199237395,0.0]
#atrial = [-0.2967663878261003*2,0.25057478717710174*2,1.3827297351466428,1.702023384832311,3.0791800790217154,0.0,1.2400386725361678,0.0,-0.3561416933055925,0.0,2.801219489380058,0.0,-1.1290392201099397,0.0,1.0162289199237395,0.0]
#atrial = [-0.5489307813321151,0.3318064860491032,1.3836581492240345,1.7999142202401186,26.778627442628697,0.0,1.2786729767481086,0.0,-1.8044641240364967,0.0,2.018989606638419,0.0,-23.42487315195119,0.0,1.248306537682799,0.0] #0.0003011873652880382
#atrial = [-0.5336021091255497,0.2999401472397189,1.3833756425260124,1.8206931811172136,16.30785328812252,0.0,1.3870095791283663,0.0,-4.448314627596262,0.0,1.7985112282392832,0.0,-10.3255967541902,0.0,1.29143086997431,0.0]
#atrial = [-0.5297041429112943,0.2933240470063606,1.3830031311925879,1.8253942281094937,14.986091075429751,0.0,1.464640372400348,0.0,-7.931800351569044,0.0,1.717799667957279,0.0,-5.524251375777077,0.0,1.2773197102081726,0.0]
#atrial = [-0.5243881693053298,0.2836999647812343,1.3832697630116844,1.8321298873666765,49.904780921019984,0.0,1.4654597590489415,0.0,-22.280787592453734,0.0,1.6135419037436551,0.0,-26.099271853203692,0.0,1.3769569862282118,0.0]
#atrial = [-0.5251847078236657,0.2835380989822935,1.3837836249167428,1.8318749094338078,53.123080563224285,0.0,1.4718814839416798,0.0,-25.331145427490217,0.0,1.6066377692092113,0.0,-26.26641632953569,0.0,1.3799755188402958,0.0]
#atrial = [-0.5249565598434786,0.28333358471586606,1.383715102743298,1.8320654762189466,55.52393985773224,0.0,1.4749129722610685,0.0,-27.470786754029175,0.0,1.601868924110799,0.0,-26.523862614593764,0.0,1.3812751618641164,0.0]
#New minimum: 0.00023883842674643924 [-0.5247762751816804,0.2828915570634826,1.3837656415003445,1.8323467099030555,63.120917132667905,0.0,1.478130410600365,0.0,-32.17183668366479,0.0,1.5929285243364661,0.0,-29.423970266940803,0.0,1.38693874263686,0.0] 0.00047359417598702186 0.0012219351046597327
#atrial = [-0.5247762751816804,0.2828915570634826,1.3837656415003445,1.8323467099030555,63.120917132667905,0.0,1.478130410600365,0.0,-32.17183668366479,0.0,1.5929285243364661,0.0,-29.423970266940803,0.0,1.38693874263686,0.0] # 0.00047359417598702186 0.0012219351046597327
#New minimum: 0.00023883842674643924 [-0.5247762751816804,0.2828915570634826,1.3837656415003445,1.8323467099030555,63.120917132667905,0.0,1.478130410600365,0.0,-32.17183668366479,0.0,1.5929285243364661,0.0,-29.423970266940803,0.0,1.38693874263686,0.0] 0.00047359417598702186 0.0012219351046597327
#atrial = [-0.5237762751816804,0.2828915570634826,1.3837656415003445,1.8323467099030555,63.120917132667905,0.0,1.478130410600365,0.0,-32.17183668366479,0.0,1.5929285243364661,0.0,-29.423970266940803,0.0,1.38693874263686,0.0]

atrial = [-0.5185744409672999,0.2854238724264553,1.3848025999389613,1.8339730386273703,63.11883856322811,-0.002585613714221086,1.479593361171615,-0.001569459359505545,-32.174013349165506,-0.00439276758185438,1.593669981824972,0.00035151389750538633,-29.42595216593894,-0.0013496416338785663,1.3889097102817822,0.002608575323797371]

model = gauss_model(atrial)
clf()
plot(time,gaussian)
plot(time,model)
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
coeff_inf = 0.0
for k=1:nexp
  a1 = abest[1+(k-1)*4]
  a2 = abest[2+(k-1)*4]
  b  = abest[3+(k-1)*4]
  om = abest[4+(k-1)*4]
  psdt = (a2*om.*(b^2 + om^2 - omega.^2) + a1*b.*(b^2 + om^2 + omega.^2))./
         (b^4 + (om^2 - omega.^2).^2 + 2*b^2.*(om^2 + omega.^2))
  psd_grid += psdt
  coeff_inf += a1*b-a2*om
#  plot(omega,psdt)
#  plot(omega,(a1*b-a2*om)./omega.^2,"r.")
#  read(STDIN,Char)
end
clf()
plot(omega,psd_grid)
plot(omega,exp(-omega.^2./2)*sqrt(pi/2))
#plot(omega,coeff_inf./omega.^2)
#return
#end
