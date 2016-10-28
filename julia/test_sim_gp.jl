using PyPlot

include("simulate_exp_gp.jl")
nt = 100
t = collect(linspace(0,100,nt))
beta = complex(0.,1.0)
#beta = 1.0
alpha = 1.0
ndev = randn(nt)
data = simulate_exp_gp(t,alpha,beta,ndev)
#datac = simulate_exp_gp(t,alpha,conj(beta),ndev)
plot(t,real(data))
