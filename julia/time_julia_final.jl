using PyPlot
include("compile_matrix_symm.jl")
include("compute_likelihood.jl")
include("bandec_trans.jl")
include("banbks_trans.jl")


# Compute time for Generalized Rybicki-Press with three Lorentzian components
# for benchmarking:

function time_julia_final()
omega = 2pi/12.203317
alpha_final = [1.0428542, 0.30345984]
beta_real_final = [0.1229159,0.09086397]
beta_imag_final = [0.0,omega]
#beta_imag_final = [0.0,0.0]
p_final = length(alpha_final)
p0_final = 0
for i=1:p_final
  if beta_imag_final[i] == 0.0
    p0_final +=1
  end
end

w0 =  0.03027
nt = [16,16,32,64,128,256,512,1024,2048,4096,8192,16384,32768,65536,131072,262144,524288]
#nt = [16]
nnt = length(nt)
time_compute_final = zeros(nnt)
time_likelihood_final = zeros(nnt)
for it=1:nnt
  n = nt[it]
  t = collect(linspace(0,n-1,n))
# White noise component:

# Use white noise for doing the timing test:
  y=randn(n)*sqrt(w0)
# Now use Ambikarasan O(N) method:
#  nex = (2p+1)*n-2p
#  width = 2p+3
#  m1 = p+1
#  aex::Array{Complex{Float64},2} = zeros(Complex{Float64},width,nex)
#  al_small::Array{Complex{Float64},2} = zeros(Complex{Float64},m1,nex)
#  indx::Vector{Int64} = collect(1:nex)
#  tic()
# See if I can look into types of this routine.
#  @code_warntype logdeta = lorentz_likelihood_hermitian_band_init(alpha,beta,w0,t,nex,aex,al_small,indx)
#  logdeta = lorentz_likelihood_hermitian_band_init(alpha,beta,w0,t,nex,aex,al_small,indx)
#  aex_full,bex_full  = lorentz_likelihood_hermitian(alpha,beta,w0,t,y)
#  eigval_aex = eigvals(aex_full)
# Now use the new slimmed-down real version:
  nex_final = (4(p_final-p0_final)+2p0_final+1)*(n-1)+1
  m1_final = 2(p_final-p0_final)+p0_final+2
  if p0_final == p_final
    m1_final = p0_final + 1
  end
  width_final = 2m1_final+1
  aex_final = zeros(Float64,width_final,nex_final)
  al_small_final = zeros(Float64,m1_final,nex_final)
  indx_final= collect(1:nex_final)
  tic()
#  @code_warntype   compile_matrix_symm(alpha_final,beta_real_final,beta_imag_final,w0,t,nex_final,aex_final,al_small_final,indx_final)
  logdeta_final= compile_matrix_symm(alpha_final,beta_real_final,beta_imag_final,w0,t,nex_final,aex_final,al_small_final,indx_final)
  time_compute_final[it] = toq();
  tic()
  bex = zeros(Float64,nex_final)
#  @code_warntype compute_likelihood(p_final,y,aex_final,al_small_final,indx_final,logdeta_final,bex)
  log_like_final= compute_likelihood(p_final,p0_final,y,aex_final,al_small_final,indx_final,logdeta_final,bex)
  time_likelihood_final[it] = toq();
  println(n," final:   ",time_compute_final[it]," ",time_likelihood_final[it])

  
#  time_compute[it] = toq();
#  tic()
#  log_like = lorentz_likelihood_hermitian_band_save(p,y,aex,al_small,indx,logdeta)
#  time_likelihood[it] = toq();
#  println(n," ",time_compute[it]," ",time_likelihood[it])
#  println("Eigenvalues: ",eigval_aex)
#  read(STDIN,Char)
end
return
end
