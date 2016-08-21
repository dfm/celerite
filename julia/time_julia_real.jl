using PyPlot
using ForwardDiff

include("lorentz_likelihood_hermitian_band_init.jl")
include("lorentz_likelihood_hermitian_band_save.jl")
include("lorentz_likelihood_real_band_init.jl")
include("lorentz_likelihood_real_band_save.jl")
include("bandec.jl")
include("banbks.jl")

# Create a closure 'wrapper' to compute the derivative of the log likelihood function:
function log_like_derivative_wrapper(p,x,t,y)
  n = length(t)
  function log_like(x)
    w0::eltype(x) = x[1]
    alpha = x[2:p+1]
    beta_real = x[p+2:2p+1]
    beta_imag = x[2p+2:3p+1]
    nex::Int64 = 2*((2p+1)*n-2p)
    aex_real = zeros(eltype(x),nex,4*p+7)
    m1::Int64 = 2*p+3
    al_small_real = zeros(eltype(x),nex,m1)
    indx_real::Vector{Int64} = collect(1:nex)
    logdeta_real = lorentz_likelihood_real_band_init(alpha,beta_real,beta_imag,w0,t,nex,aex_real,al_small_real,indx_real)
    log_like_real = lorentz_likelihood_real_band_save(p,y,aex_real,al_small_real,indx_real,logdeta_real)
    return log_like_real
  end
  
# Now, try to run AutoDiff:
#  result = HessianResult(x)
#  ForwardDiff.hessian!(result,log_like,x)
  result=ForwardDiff.gradient(log_like,x,Chunk{10}())
#  println(result)
#  println("Determinant:  ",ForwardDiff.value(result))
#  println("Gradient:     ",ForwardDiff.gradient(result))
#  println("Hessian:      ",ForwardDiff.hessian(result))
  return result
end    

# Compute time for Generalized Rybicki-Press with three Lorentzian components
# for benchmarking:

function time_julia_real()
omega = 2pi/12.203317
#alpha = [1.0428542, -0.38361831, 0.30345984/2, 0.30345984/2]*1.6467e-7
alpha = [1.0428542, 0.30345984/2, 0.30345984/2]*1.6467e-7
alpha_real = [1.0428542, 0.30345984/2, 0.30345984/2]*1.6467e-7
#alpha_real = [1.0428542, 0.30345984]*1.6467e-7
#alpha = [1.0428542, 0.30345984]*1.6467e-7
p_real = length(alpha_real)
p = length(alpha)
#beta = [complex(0.1229159,0.0),complex(0.48922908,0.0),complex(0.09086397,omega),complex(0.09086397,-omega)]
beta = [complex(0.1229159,0.0),complex(0.09086397,omega),complex(0.09086397,-omega)]
beta_real = [0.1229159,0.09086397,0.09086397]
#beta_real = [0.1229159,0.09086397]
beta_imag = [0.0,omega,-omega]
#beta_imag = [0.0,omega]
w0 =  0.03027 * 1.6467e-7
#nt = [16,32,64,128,256,512,1024,2048,4096,8192,16384,32768,65536,131072,262144,524288]
nt = [16,32,64,128,256,512,1024,2048,4096,8192,16384,32768,65536,131072] #,262144,524288]
#nt = [16,32,64,128,256,512,1024,2048,4096,8192] #,16384,32768,65536,131072,262144,524288]
#nt = [16,32,64,128,256,512]
nnt = length(nt)
time_compute = zeros(nnt)
time_compute_real = zeros(nnt)
time_likelihood = zeros(nnt)
time_likelihood_real = zeros(nnt)
time_derivative = zeros(nnt)
for it=1:nnt
  n = nt[it]
  t = collect(linspace(0,n-1,n))
# White noise component:

# Use white noise for doing the timing test:
  y=randn(n)*sqrt(w0)
# Now use Ambikarasan O(N) method:
  tic()
  nex_real = 2*((2p_real+1)*n-2p_real)
  width_real = 4p_real+7
  m1_real = 2p_real+3
  aex_real = zeros(Float64,nex_real,width_real)
  al_small_real = zeros(Float64,nex_real,m1_real)
  indx_real = collect(1:nex_real)
  logdeta_real = lorentz_likelihood_real_band_init(alpha_real,beta_real,beta_imag,w0,t,nex_real,aex_real,al_small_real,indx_real)
  time_compute_real[it] = toq();
  tic()
#  @code_warntype log_like_real = lorentz_likelihood_real_band_save(p_real,y,aex_real,al_small_real,indx_real,logdeta_real)
  log_like_real = lorentz_likelihood_real_band_save(p_real,y,aex_real,al_small_real,indx_real,logdeta_real)
  time_likelihood_real[it] = toq();
# Now, try to compute the derivative:
  tic()
  x = [w0;alpha;beta_real;beta_imag]
#  log_like_hessian = HessianResult(x)
  log_like_hessian = log_like_derivative_wrapper(p_real,x,t,y)
  time_derivative[it] = toq();
  println(n," ",time_compute_real[it]," ",time_likelihood_real[it]," ",time_derivative[it])
  tic()
  nex = (2p+1)*n-2p
  width = 2p+3
  m1 = p+1
  aex::Array{Complex{Float64},2} = zeros(Complex{Float64},nex,width)
  al_small::Array{Complex{Float64},2} = zeros(Complex{Float64},nex,m1)
  indx::Vector{Int64} = collect(1:nex)
  tic()
# See if I can look into types of this routine.
#  @code_warntype logdeta = lorentz_likelihood_hermitian_band_init(alpha,beta,w0,t,nex,aex,al_small,indx)
  logdeta_hermitian = lorentz_likelihood_hermitian_band_init(alpha,beta,w0,t,nex,aex,al_small,indx)
  time_compute[it] = toq();
  tic()
  log_like_hermitian = lorentz_likelihood_hermitian_band_save(p,y,aex,al_small,indx,logdeta_hermitian)
  time_likelihood[it] = toq();
  println(n," ",time_compute[it]," ",time_likelihood[it])
#  println("Log Det, real: ",logdeta_real," hermitian: ",logdeta_hermitian)
  println("Log like, real: ",log_like_real," hermitian: ",log_like_hermitian)
end
loglog(nt,time_compute_real)
loglog(nt,time_compute)
loglog(nt,time_derivative)
return
end
