using PyPlot

phi = linspace(-pi,pi,1000)
dphi = collect(linspace(-2pi,2pi,100))

spot = cos(phi)
spot = 0.5.*(spot+abs(spot))


amp = zeros(100)
for i=1:100
  sspot = cos(phi+dphi[i])
  sspot = 0.5.*(sspot+abs(sspot))
  plot(phi,spot)
  plot(phi,sspot)
  plot(phi,spot.*sspot)
  amp[i] = sum(spot.*sspot)
end

clf()
plot(dphi,amp)
gamma = 1.7
plot(dphi,(exp(-gamma.*sin(.5.*dphi).^2)-exp(-gamma))./(1.-exp(-gamma)).*maximum(amp))
