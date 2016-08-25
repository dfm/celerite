using PyPlot
using ForwardDiff

include("lorentz_likelihood_hermitian_band_init.jl")
include("lorentz_likelihood_hermitian_band_save.jl")
include("lorentz_likelihood_real2_band_init.jl")
include("lorentz_likelihood_real2_band_save.jl")
include("compile_matrix_symm.jl")
include("compute_likelihood.jl")
include("bandec_real2_trans.jl")
include("banbks_real2_trans.jl")
include("bandec_trans.jl")
include("banbks_trans.jl")

# Create a closure 'wrapper' to compute the derivative of the log likelihood function:
function log_like_derivative_wrapper(p,x,t,y)
  n = length(t)
  function log_like(x)
    w0::eltype(x) = x[1]
    alpha = x[2:p+1]
    beta_real = x[p+2:2p+1]
    beta_imag = x[2p+2:3p+1]
    nex::Int64 = (2p+1)*n-2p
    aex_real1 = zeros(eltype(x),2*p+3,nex)
    aex_real2 = zeros(eltype(x),2*p+3,nex)
    m1::Int64 = p+1
    al_small_real1 = zeros(eltype(x),m1,nex)
    al_small_real2 = zeros(eltype(x),m1,nex)
    indx_real::Vector{Int64} = collect(1:nex)
#    @code_warntype lorentz_likelihood_real2_band_init(alpha,beta_real,beta_imag,w0,t,nex,aex_real1,aex_real2,al_small_real1,al_small_real2,indx_real)
    logdeta_real = lorentz_likelihood_real2_band_init(alpha,beta_real,beta_imag,w0,t,nex,aex_real1,aex_real2,al_small_real1,al_small_real2,indx_real)
#    @code_warntype lorentz_likelihood_real2_band_save(p,y,aex_real1,aex_real2,al_small_real1,al_small_real2,indx_real,logdeta_real)
    log_like_real = lorentz_likelihood_real2_band_save(p,y,aex_real1,aex_real2,al_small_real1,al_small_real2,indx_real,logdeta_real)
    return log_like_real
  end
  
# Now, try to run AutoDiff:
#  result = HessianResult(x)
#  ForwardDiff.hessian!(result,log_like,x)
  result=ForwardDiff.gradient(log_like,x,Chunk{10}())
#  result=ForwardDiff.gradient(log_like,x)
#  println(result)
#  println("Determinant:  ",ForwardDiff.value(result))
#  println("Gradient:     ",ForwardDiff.gradient(result))
#  println("Hessian:      ",ForwardDiff.hessian(result))
# Now perturb x:
#  dlogdx = zeros(eltype(x),length(x))
#  log_like0=log_like(x)
#  x0 = copy(x)
#  for i=1:length(x)
#    x    = copy(x0)
#    if (x[i] == 0.) then
#      h = 1e-6
#    else
#      h = 1e-6*x[i]
#    end
#    x[i] = x[i]+h
#    dlogdx[i]=(log_like(x)-log_like0)/h
#  end
#  println("Gradient AutoDiff:    ",result)
#  println("Gradient Finite diff: ",dlogdx)
  return result
end    

# Compute time for Generalized Rybicki-Press with three Lorentzian components
# for benchmarking:
#
function time_julia_real2()
omega1 = 2pi/12.203317
omega2 = 2pi/25.0
omega3 = 2pi/50.0
omega4 = 2pi/100.0
#alpha = [1.0428542, -0.38361831, 0.30345984/2, 0.30345984/2]*1.6467e-7
#alpha = [1.0428542, 0.30345984/2, 0.30345984/2, 0.4/2., 0.4/2., 0.1, 0.1, 0.05, 0.05]*1.6467e-7
alpha = [1.0428542, 0.30345984/2, 0.30345984/2]
#alpha = [1.]
alpha_real = alpha
#alpha_real = [1.0428542]*1.6467e-7
#alpha_real = [1.0428542, 0.30345984]*1.6467e-7
#alpha = [1.0428542, 0.30345984]*1.6467e-7
p_real = length(alpha_real)
p = length(alpha)
#beta = [complex(0.1229159,0.0),complex(0.48922908,0.0),complex(0.09086397,omega1),complex(0.09086397,-omega1)]
#beta = [complex(0.1229159,0.0)]
#beta_real = [0.1229159,0.09086397,0.09086397, 0.01, 0.01, 0.005, 0.005, 0.0025, 0.0025]
beta_real = [0.1229159,0.09086397,0.09086397]
#beta_real = [0.4]
#beta_real = [0.1229159,0.09086397]
#beta_imag = [0.1,-0.1,omega1,-omega1, omega2, -omega2, omega3, -omega3, omega4, -omega4]
beta_imag = [0.0,omega1,-omega1]
#beta_imag = [0.0]
#beta = [complex(0.1229159,0.0),complex(0.09086397,omega1),complex(0.09086397,-omega1),complex(0.01,omega2),complex(0.01,-omega2)]
beta = complex(beta_real, beta_imag)
#beta_imag = [0.0,omega1]
alpha_final = [1.0428542, 0.30345984]
beta_real_final = [0.1229159,0.09086397]
beta_imag_final = [0.0,omega1]
#alpha_final = [1.]
#beta_real_final = [0.4]
#beta_imag_final = [0.0]
p_final = length(alpha_final)
p0_final = 0
for i=1:p_final
  if beta_imag_final[i] == 0.0
    p0_final +=1
  end
end
w0 =  0.
#nt = [16,32,64,128,256,512,1024,2048,4096,8192,16384,32768,65536,131072,262144,524288]
#nt = [16,32,64,128,256,512,1024,2048,4096,8192,16384,32768,65536]
#nt = [16,32,64,128,256,512,1024,2048,4096,8192,16384,32768,65536,131072] #,262144,524288]
#nt = [16,32,64,128,256,512,1024,2048,4096,8192] #,16384,32768,65536,131072,262144,524288]
#nt = [16,32,64,128,256,512]
nt = [16384]
#nt = [4]
nnt = length(nt)
time_compute = zeros(nnt)
time_compute_real = zeros(nnt)
time_compute_final = zeros(nnt)
time_likelihood = zeros(nnt)
time_likelihood_real = zeros(nnt)
time_likelihood_final = zeros(nnt)
time_derivative = zeros(nnt)
for it=1:nnt
  n = nt[it]
  t = collect(linspace(0,n-1,n))
# White noise component:

# Use white noise for doing the timing test:
  y=randn(n)
# Now use Ambikarasan O(N) method:
  nex_real = (2p_real+1)*n-2p_real
  width_real = 2p_real+3
  m1_real = p_real+1
  aex_real1 = zeros(Float64,width_real,nex_real)
  aex_real2 = zeros(Float64,width_real,nex_real)
  al_small_real1 = zeros(Float64,m1_real,nex_real)
  al_small_real2 = zeros(Float64,m1_real,nex_real)
  indx_real = collect(1:nex_real)
  tic()
  logdeta_real = lorentz_likelihood_real2_band_init(alpha_real,beta_real,beta_imag,w0,t,nex_real,aex_real1,aex_real2,al_small_real1,al_small_real2,indx_real)
  time_compute_real[it] = toq();
#  @code_warntype log_like_real = lorentz_likelihood_real_band_save(p_real,y,aex_real,al_small_real,indx_real,logdeta_real)
  tic()
  log_like_real = lorentz_likelihood_real2_band_save(p_real,y,aex_real1,aex_real2,al_small_real1,al_small_real2,indx_real,logdeta_real)
  time_likelihood_real[it] = toq();
# Now use the new slimmed-down real version:
  nex_final = (4(p_final-p0_final)+2p0_final+1)*(n-1)+1
  m1_final = 2(p_final-p0_final)+p0_final+2
  width_final = 2m1_final+1
  aex_final = zeros(Float64,width_final,nex_final)
  al_small_final = zeros(Float64,m1_final,nex_final)
  indx_final= collect(1:nex_final)
  tic()
#  @code_warntype   compile_matrix_symm(alpha_final,beta_real_final,beta_imag_final,w0,t,nex_final,aex_final,al_small_final,indx_final)
  logdeta_final= compile_matrix_symm(alpha_final,beta_real_final,beta_imag_final,w0,t,nex_final,aex_final,al_small_final,indx_final)
  time_compute_final[it] = toq();
  tic()
#  @code_warntype   compute_likelihood(p_final,y,aex_final,al_small_final,indx_final,logdeta_final)
  log_like_final= compute_likelihood(p_final,p0_final,y,aex_final,al_small_final,indx_final,logdeta_final)
  time_likelihood_final[it] = toq();
  println(n," final:   ",time_compute_final[it]," ",time_likelihood_final[it])

# Now, try to compute the derivative:
  x = [w0;alpha;beta_real;beta_imag]
#  log_like_hessian = HessianResult(x)
#  @code_warntype log_like_derivative_wrapper(p_real,x,t,y)
#  @time log_like_hessian = log_like_derivative_wrapper(p_real,x,t,y)
  tic()
#  log_like_hessian = log_like_derivative_wrapper(p_real,x,t,y)
  time_derivative[it] = toq();
  println(n," real2:   ",time_compute_real[it]," ",time_likelihood_real[it]," ",time_derivative[it])
#  println(n," ",time_compute_real[it]," ",time_likelihood_real[it])
  nex = (2p+1)*n-2p
  width = 2p+3
  m1 = p+1
  aex::Array{Complex{Float64},2} = zeros(Complex{Float64},width,nex)
  al_small::Array{Complex{Float64},2} = zeros(Complex{Float64},m1,nex)
  indx::Vector{Int64} = collect(1:nex)
# See if I can look into types of this routine.
#  @code_warntype logdeta = lorentz_likelihood_hermitian_band_init(alpha,beta,w0,t,nex,aex,al_small,indx)
  tic()
  logdeta_hermitian = lorentz_likelihood_hermitian_band_init(alpha,beta,w0,t,nex,aex,al_small,indx)
  time_compute[it] = toq();
  tic()
  log_like_hermitian = lorentz_likelihood_hermitian_band_save(p,y,aex,al_small,indx,logdeta_hermitian)
  time_likelihood[it] = toq();
  println(n," complex: ",time_compute[it]," ",time_likelihood[it])

  nlor = p
  logdetC_ana = zeros(Complex{Float64},nlor)

  for k=1:nlor
    logdetC_ana[k] += n*log(alpha[k])
    for i=1:n-1
      logdetC_ana[k] += log(1.0-exp(-2.0*beta[k]*(t[i+1]-t[i])))
    end
  end

  println("Log Det, real: ",logdeta_real," hermitian: ",logdeta_hermitian," final: ",logdeta_final," analytic: ",sum(logdetC_ana))
  println("Log like, real: ",log_like_real," hermitian: ",log_like_hermitian," final: ",log_like_final)
end
#loglog(nt,time_compute_real)
#loglog(nt,time_compute)
#loglog(nt,time_derivative)

fig,ax = subplots()

ax[:loglog](nt,time_compute_real,"g-",label = "real",alpha=0.6,linewidth=2)
ax[:loglog](nt,time_compute,"b-",label = "complex",alpha=0.6,linewidth=2)
ax[:loglog](nt,time_compute_final,"k-",label = "final",alpha=0.6,linewidth=2)
ax[:loglog](nt,time_derivative,"r-",label = "autodiff",alpha=0.6,linewidth=2)
#ax[:semilogx]([1,1],[miny,maxy],"m-",label="actual",linewidth=2,alpha=0.6)
#ax[:semilogx]([sig_best[igen],sig_best[igen]],[miny,maxy],"c-",label="inferred",linewidth=2,alpha=0.6)
ax[:legend](loc="upper left")

return
end
