using PyPlot

include("simulate_exp_gp.jl")
nt = 100000
t = collect(linspace(0,10,nt))
beta = 0.3
alpha = 1.0
ndev = randn(nt)
data = simulate_exp_gp(t,alpha,beta,ndev)
plot(t,data)
