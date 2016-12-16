# Plot PSD of simple harmonic oscillator with different values of Q:

using PyPlot

fix,ax = subplots(1,1)

x = logspace(-1.5,1.5,100000)

lw = 3

Q = 0.5
psd = 1./((x.^2-1.).^2+x.^2./Q^2)
ax[:loglog](x,psd,label="Q=0.5",linewidth=lw)
Q = 2.0
psd = 1./((x.^2-1.).^2+x.^2./Q^2)
ax[:loglog](x,psd,label="Q=2",linewidth=lw)
Q = 10.0
psd = 1./((x.^2-1.).^2+x.^2./Q^2)
ax[:loglog](x,psd,label="Q=10",linewidth=lw)
psd = 0.25./((x-1.).^2+1./4./Q^2)
ax[:loglog](x,psd,label="Lorentzian",ls="dashed",linewidth=lw)


psd = 1./(1+x.^2).^2
ax[:loglog](x,psd,label="Matern",ls="dashed",linewidth=lw)

fontsize=10
ax[:legend](loc="upper right",fontsize=fontsize)

ax[:axis]([1./10.^1.5,10^1.5,1e-4,1e2])

