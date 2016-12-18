using PyPlot
using Roots

ntime = 100
time = collect(linspace(0,10,ntime))
gaussian=exp(-time.^2/2.)

function psd_func(x,omega)
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

nexp = 4
x = [-0.5247762751816804,0.2828915570634826,1.3837656415003445,1.8323467099030555, 63.120917132667905,0.0,1.478130410600365,0.0, -32.17183668366479,0.0,1.5929285243364661,0.0, -29.423970266940803,0.0,1.38693874263686,0.0]

nfreq = 1000
omega = logspace(-1,1,nfreq)
psd = zeros(nfreq)
for i=1:nfreq
  psd[i]=psd_func(x,omega[i])
end

# model = gauss_model(atrial)
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


# Now, solve for roots of PSD:

#omega_min = newton(dpsd_domega,5.0)
omega_min = fzero(dpsd_domega,[1e-4,1e4])
fmin = psd_func(x,omega_min)

plot(omega,psd)
scatter(omega_min,fmin)
