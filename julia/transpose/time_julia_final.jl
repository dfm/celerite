using PyPlot
include("compile_matrix_symm.jl")
include("compute_likelihood.jl")
#include("lorentz_likelihood_hermitian.jl")
include("bandec_trans.jl")
include("banbks_trans.jl")


# Compute time for Generalized Rybicki-Press with three Lorentzian components
# for benchmarking:

function time_julia_final()
omega = 2pi/12.203317
#alpha = [1.0428542, -0.38361831, 0.30345984/2, 0.30345984/2]*1.6467e-7
alpha = [1.0428542, 0.30345984/2, 0.30345984/2]*1.6467e-7
p = length(alpha)
#beta = [complex(0.1229159,0.0),complex(0.48922908,0.0),complex(0.09086397,omega),complex(0.09086397,-omega)]
beta = [complex(0.1229159,0.0),complex(0.09086397,omega),complex(0.09086397,-omega)]
w0 =  0.03027 * 1.6467e-7
nt = [16,32,64,128,256,512,1024,2048,4096,8192,16384,32768,65536,131072,262144,524288]
#nt = [16]
nnt = length(nt)
time_compute = zeros(nnt)
time_likelihood = zeros(nnt)
for it=1:nnt
  n = nt[it]
  t = collect(linspace(0,n-1,n))
# White noise component:

# Use white noise for doing the timing test:
  y=randn(n)*sqrt(w0)
# Now use Ambikarasan O(N) method:
  nex = (2p+1)*n-2p
  width = 2p+3
  m1 = p+1
  aex::Array{Complex{Float64},2} = zeros(Complex{Float64},width,nex)
  al_small::Array{Complex{Float64},2} = zeros(Complex{Float64},m1,nex)
  indx::Vector{Int64} = collect(1:nex)
  tic()
# See if I can look into types of this routine.
#  @code_warntype logdeta = lorentz_likelihood_hermitian_band_init(alpha,beta,w0,t,nex,aex,al_small,indx)
  logdeta = lorentz_likelihood_hermitian_band_init(alpha,beta,w0,t,nex,aex,al_small,indx)
#  aex_full,bex_full  = lorentz_likelihood_hermitian(alpha,beta,w0,t,y)
#  eigval_aex = eigvals(aex_full)
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

  
  time_compute[it] = toq();
  tic()
  log_like = lorentz_likelihood_hermitian_band_save(p,y,aex,al_small,indx,logdeta)
  time_likelihood[it] = toq();
  println(n," ",time_compute[it]," ",time_likelihood[it])
#  println("Eigenvalues: ",eigval_aex)
#  read(STDIN,Char)
end
return
end
