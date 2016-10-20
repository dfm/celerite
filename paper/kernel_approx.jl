# Try to approximate squared-exponential kernel:
using PyPlot

time = collect(linspace(0,10,1000))

plot(time,exp(-time^2/2))
#plot(time,exp(-time)
fa = exp(-time).*cos(time)
plot(time,fa)

# If we want to approximate Gaussian, then we just get a cosine... that doesn't work well!

# How about sum of two?  This I solved numerically with Mathematically, and unfortunately I find
# a growing solution in each case... darn.
