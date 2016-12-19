# Plot PSD of simple harmonic oscillator with different values of Q:

using PyPlot

fix,ax = subplots(1,1)

x = logspace(-1.5,1.5,100000)

lw = 3

Q = 0.5
psd = 1./((x.^2-1.).^2+x.^2./Q^2)
ax[:loglog](x,psd,label=L"$Q=\frac{1}{2}$; Matern 3/2",linewidth=lw)
Q = 1./sqrt(2.)
psd = 1./((x.^2-1.).^2+x.^2./Q^2)
ax[:loglog](x,psd,label=L"$Q=\frac{1}{\sqrt{2}}$",linewidth=lw)
#Q = 1.
#psd = 1./((x.^2-1.).^2+x.^2./Q^2)
#ax[:loglog](x,psd,label=L"$Q=1$",linewidth=lw)
Q = 2.0
psd = 1./((x.^2-1.).^2+x.^2./Q^2)
ax[:loglog](x,psd,label=L"$Q=2$",linewidth=lw)
#Q = 4.0
#psd = 1./((x.^2-1.).^2+x.^2./Q^2)
#ax[:loglog](x,psd,label="Q=4",linewidth=lw)
Q = 10.0
psd = 1./((x.^2-1.).^2+x.^2./Q^2)
ax[:loglog](x,psd,label=L"$Q=10$",linewidth=lw)
#psd = 0.25./((x-1.).^2+1./4./Q^2)
psd = 1./((4-2./Q^2).*(x-sqrt(1-.5/Q^2)).^2+(1./Q^2-.25/Q^4))
ax[:loglog](x,psd,label="Lorentzian",ls="dashed",linewidth=lw)

qgrid = linspace(1./sqrt(2.)+1e-3,10,1000)
xmax = sqrt(1.-.5./qgrid.^2)
ax[:loglog](xmax,4.*qgrid.^4./(4.*qgrid.^2-1.),linewidth=2,ls="dotted",color="black")

#psd = 1./(1+x.^2).^2
#ax[:loglog](x,psd,label="Matern",ls="dashed",linewidth=lw)

fontsize=20
ax[:legend](loc="upper right",fontsize=fontsize)
ax[:set_xlabel](L"$\omega/\omega_0$",fontsize=fontsize*2)
ax[:set_ylabel](L"$P(\omega)/P(0)$",fontsize=fontsize*2)

ax[:axis]([1./10.^1.5,10^1.5,1e-4,1e2])

