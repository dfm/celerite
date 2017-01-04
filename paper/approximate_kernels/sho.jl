# Plot PSD of simple harmonic oscillator with different values of Q:

using PyPlot


sho_psd(x,Q) = 1./((x.^2-1.).^2+x.^2./Q^2)

x = logspace(-1.5,1.5,100000)

lw = 3

fig,ax= subplots(1,1)
Q = 0.5
ax[:loglog](x,sho_psd(x,Q),label=L"$Q=\frac{1}{2}$; Matern 3/2",linewidth=lw)
Q = 1./sqrt(2.)
ax[:loglog](x,sho_psd(x,Q),label=L"$Q=\frac{1}{\sqrt{2}}$",linewidth=lw)
#Q = 1.
#ax[:loglog](x,sho_psd(x,Q),label=L"$Q=1$",linewidth=lw)
Q = 2.0
ax[:loglog](x,sho_psd(x,Q),label=L"$Q=2$",linewidth=lw)
#Q = 4.0
#ax[:loglog](x,sho_psd(x,Q),label="Q=4",linewidth=lw)
Q = 10.0
ax[:loglog](x,sho_psd(x,Q),label=L"$Q=10$",linewidth=lw)
#lorentz_psd(x,Q) = Q^2./ ((x-sqrt(1.-1/2/Q^2)).^2.*(1-1/2/Q^2) * (2*Q)^2 + 1-1/4/Q^2)
#lorentz_psd(x,Q) = Q^2.* (1./ ((x-sqrt(1.-1/2/Q^2)).^2 * (2*Q)^2 + 1-1/4/Q^2) +
lorentz_psd(x,Q) = Q^2.*(1./ ((x-sqrt(1.-1/2/Q^2)).^2.*(1-1/2/Q^2) * (2*Q)^2 + 1-1/4/Q^2)
 +1./ ((x+sqrt(1.-1/2/Q^2)).^2.*(1-1/2/Q^2) * (2*Q)^2 + 1-1/4/Q^2))

#lorentzian = 1./((4-2./Q^2).*(x-sqrt(1-.5/Q^2)).^2+(1./Q^2-.25/Q^4))
ax[:loglog](x,lorentz_psd(x,Q),label="Lorentzian",ls="dashed",linewidth=lw)

qgrid = linspace(1./sqrt(2.)+1e-3,10,1000)
xmax = sqrt(1.-.5./qgrid.^2)
ax[:loglog](xmax,4.*qgrid.^4./(4.*qgrid.^2-1.),linewidth=2,ls="dotted",color="black")

#matern= 1./(1+x.^2).^2
#ax[:loglog](x,matern,label="Matern",ls="dashed",linewidth=lw)

fontsize=20
ax[:legend](loc="upper right",fontsize=fontsize)
ax[:set_xlabel](L"$\omega/\omega_0$",fontsize=fontsize*2)
ax[:set_ylabel](L"$P(\omega)/P(0)$",fontsize=fontsize*2)

ax[:axis]([1./10.^1.5,10^1.5,1e-4,1e2])

function sho_acf(omtau,Q)
# omtau is \omega_0*\tau
# ACF is normalized by P_0*\omega_0
# This is only valid for Q >= 1/2
@assert(Q >= 0.5)
lam = sqrt(1.-.25/Q^2).*omtau
phi = atan(1/sqrt(4.*Q^2-1.))
acf = exp(-.5.*omtau./Q).*cos(lam-phi)
return acf/acf[1]
end

omtau = collect(linspace(0,6*pi,1000))

fig,ax= subplots(1,1)
Q = 0.50001
ax[:plot](omtau,sho_acf(omtau,Q),label=L"$Q=\frac{1}{2}$",linewidth=lw,color="blue")
Q = 1/sqrt(2)
ax[:plot](omtau,sho_acf(omtau,Q),label=L"$Q=\frac{1}{\sqrt{2}}$",linewidth=lw,color="orange")
Q = 2.0
ax[:plot](omtau,sho_acf(omtau,Q),label=L"$Q=2$",linewidth=lw,color="green")
Q = 10.0
ax[:plot](omtau,sho_acf(omtau,Q),label=L"$Q=10$",linewidth=lw,color="red")
ax[:plot](omtau,exp(-omtau./2/Q).*cos(omtau),label="Lorentzian",linewidth=lw,color="black",linestyle="dashed")
#ax[:plot](omtau,exp(-omtau/2).*cos(omtau))

#ax[:legend](loc="lower right",fontsize=fontsize)
ax[:set_xlabel](L"$\omega_0 \tau$",fontsize=fontsize*2)
ax[:set_ylabel](L"$k(\tau)/k(0)$",fontsize=fontsize*2)

#ax[:axis]([1./10.^1.5,10^1.5,1e-4,1e2])

savefig("acf_sho.pdf", bbox_inches="tight")

