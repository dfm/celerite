function doolittle(a::Array)
tmp = size(a)
n = tmp[1]
l = zeros(eltype(a),n,n)
d = zeros(eltype(a),n)
for k=1:n
  l[k,k]=1.0
  d[k]=a[k,k]
  for j=1:k-1
    d[k] -= d[j]*l[k,j]*l[k,j]
  end
  println(k," ",d[k])
  for j=k+1:n
    l[k,j]=0.0
    l[j,k] = a[j,k]
    for nu=1:k-1
      l[j,k] -= l[j,nu]*d[nu]*l[k,nu]
    end
    l[j,k] /= d[k]
  end
end
return l,d
end

include("lorentz_likelihood_hermitian_band_trans.jl")
include("bandec_trans.jl")
include("banbks_trans.jl")
include("compile_matrix_full.jl")
include("compile_matrix_symm.jl")
include("compile_matrix_symm_full.jl")
include("compile_matrix_symm01.jl")
include("compile_matrix.jl")

#function test_lorentzian()
# For now, just a single Lorentzian.  This requires two complex components.
omega1 = 2pi/12.
omega2 = 2pi/25.
omega3 = 2pi/50.
omega4 = 2pi/100.
alpha = [0.75/2, 0.75/2., 0.5, 0.5, 0.25, 0.25]
beta = [complex(0.05,0.0),complex(0.1,0.0),complex(0.4,omega1),complex(0.4,-omega1),complex(0.2,omega2),complex(0.2,-omega2)]
#alpha = [0.5, 0.5]
#beta = [complex(0.4,omega1),complex(0.4,-omega1)]
#alpha = [1.0]
#beta = [complex(0.4,omega1)]

nt = 10
w0 = 0.0
# For now, no white noise; just two Lorentzians:
w = ones(nt)*w0
y = randn(nt)
t=collect(linspace(0,nt-1,nt))
aex_complex,bex_complex,log_like_complex= lorentz_likelihood_hermitian_band_trans(alpha,beta,w,t,y)

## Now, just use one component:
#alpha = [1.0]
#beta = [complex(0.4,omega1)]
#aex,bex,log_like = lorentz_likelihood_hermitian_band_trans(alpha,beta,w,t,y)

# Now, try making real version:
alpha = [1.0,0.5] #,0.25,.125]
beta_real = [0.4,0.2] #,0.1,0.05,.025]
#beta_real = real(beta)
#beta_imag = [0.0,0.0,omega1,omega2] #,omega3,omega4]
beta_imag = [omega1,omega2] #,omega3,omega4]
#beta_imag = imag(beta)
#alpha = [1.0]
#beta_real = [0.4]
#beta_imag = [omega1]
#beta = [complex(0.4,omega1)]
#aex_full,bex_full,equations_full,variables_full = compile_matrix_full(alpha,beta_real,beta_imag,w0,t,y)
#aex_symm01,bex_symm01,equations_symm01,variables_symm01 = compile_matrix_symm01(alpha,beta_real,beta_imag,w0,t,y)
aex_symm_full,bex_symm_full,equations_symm_full,variables_symm_full = compile_matrix_symm_full(alpha,beta_real,beta_imag,w0,t,y)
aex_symm,bex_symm,equations_symm,variables_symm = compile_matrix_symm(alpha,beta_real,beta_imag,w0,t,y)
#aex_real,bex_real,al_small_real,indx_real,log_like,logdetofa = compile_matrix(alpha,beta_real,beta_imag,w0,t,y)

p = length(alpha)
nex = (nt-1)*(4p+1)+1

p0 = 0
for i=1:p
  if beta_imag[i] ==  0.0
    p0 += 1
  end
end
nex0 = (nt-1)*(4(p-p0)+2p0+1)+1
m1 = p0+2(p-p0+1)
width = 2m1+1

#for i=1:(4p+1):nex
#  aex_real[:,i]=aex_real[:,i]/2.0
#  bex_real[i]=bex_real[i]/2.0
#end

# Change the sign of the complex components:
#for k=2:nt
#  for j=1:p
#    jcol = (k-2)*(4p+1)+1+(j-1)*2+2
#    aex_real[jcol,:]=-aex_real[jcol,:]
#    jcol = (k-2)*(4p+1)+1+2p+(j-1)*2+2
#    aex_real[jcol,:]=-aex_real[jcol,:]
#  end
#end
# Now we have a symmetric, real matrix!  Unfortunately it's not
# positive-definite... Duoh.

# What is the bandwidth?
band = 0
for k=1:nex0
  for j=1:nex0
    if aex_symm_full[k,j] != 0.0
      band = maximum([band,abs(k-j)])
    end
  end
end

println("Bandwidth: ",band)
println("Expected : ",p0+2(p-p0+1))

# Compute the Doolittle decomposition:
# (this fails with NaN's!)
#lower,diag = doolittle(aex_real)

# Now solve:
#invkernel_y = \(transpose(aex_real),bex_real)
#invkernel_y = \(transpose(aex_symm),bex_symm)

# Multiply kernel times bex_real to see if we get back y:


#kinv_y = zeros(nex)

#for i=1:nt
#  i0 = 1+(4p+1)*(i-1)
#  kinv_y[i0]=real(bex_complex[i0])
#  if i < nt
#    for j=1:p
#      kinv_y[i0 + (j-1)*2 + 1] = real(bex_complex[i0+(j-1)*2 + 1])
#      kinv_y[i0 + (j-1)*2 + 2] = imag(bex_complex[i0+(j-1)*2 + 1])
#      kinv_y[i0 + 2p + (j-1)*2 + 1] = real(bex_complex[i0+ 2p + (j-1)*2 + 1])
#      kinv_y[i0 + 2p + (j-1)*2 + 2] = -imag(bex_complex[i0+ 2p + (j-1)*2 + 1])
#    end
#  end
#end

#y_recover = *(transpose(aex_real),kinv_y)

#agreement = 0.0
#for i=1:nt
#  i0 = 1+(4p+1)*(i-1)
#  diff = real(bex_complex[i0])-invkernel_y[i0]
#  println(real(bex_complex[i0])," ",invkernel_y[i0]," ",diff)
#  if diff > agreement
#    agreement = diff
#  end
##  println(y[i]," ",y_recover[i0]*2," ",y[i]-y_recover[i0]*2)
#end
#
#println("Agreement: ",agreement)
#return
#end
