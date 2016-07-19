using PyPlot
include("lorentz_likelihood_hermitian_band.jl")
include("lorentz_likelihood_hermitian.jl")
include("bandec.jl")
include("banbks.jl")


#alpha = [2./3.,8./7.,0.1,0.1]
#alpha = [0.,0.,0.1,0.1]
#alpha = [2./3.,8./7.]
#beta  = [complex(0.1,0.0),complex(1./23.,0.0),complex(0.03,0.2),complex(0.03,-0.2)]
#beta  = [complex(0.1,0.0),complex(1./23.,0.0)]
#alpha = [1.019, -0.36, 0.298/2, 0.298/2]
#alpha = [1.019, 0.36, 0.298/2, 0.298/2]
#beta = [0.122 + 0im, 0.532+0im, 0.0845+0.517im, 0.0845-0.517im]
omega = 2pi/12.203317
alpha = [1.0428542, -0.38361831, 0.30345984/2, 0.30345984/2]*1.6467e-7
#beta = [complex(0.1229159,0.0),complex(0.48922908,0.0),complex(0.09086397,omega),complex(0.09086397,-omega)]
beta = [complex(0.,0.0),complex(0.,0.0),complex(0.,0.),complex(0.,0.)]
# Approximate Matern kernel:
#b = 100.0
#tau = 20.0
#alpha = [1.0-b, b]
#beta = [ 1/tau+0im, 1/tau*(b-1)/b+0im]
#nt = 65268
nt = 500
t = collect(linspace(0,nt-1,nt))
acf = zeros(nt)
p = length(alpha)

# Plot the auto-correlation function:
for i=1:nt
  for j=1:p 
    acf[i] += alpha[j]*exp(-real(beta[j])*abs(t[i]-t[1]))*cos(imag(beta[j])*(t[i]-t[1]))
  end
end
clf()
plot(t,acf)
#read(STDIN,Char)

# First solve in traditional manner:
w = 0.03027 * ones(nt) * 1.6467e-7
A = zeros(Float64,nt,nt)
for i=1:nt
  A[i,i] += w[i]
  for j=1:nt
    for k=1:p
      A[i,j] += alpha[k] * exp(-real(beta[k])*abs(t[j]-t[i]))*cos(imag(beta[k])*(t[j]-t[i]))
    end
  end
end

# Compute a realization of correlated noise with this covariance matrix.
y=randn(nt)
# First, do Cholesky decomposition:
sqrta = chol(A)
# Now make a realization of the correlated noise:
#corrnoise = *(sqrta,y)
corrnoise=*(transpose(sqrta),y);
plot(corrnoise)

# Now, solve for A*y = x to get inverse of Kernel (A) times the correlated noise:
x2 = \(A,corrnoise)

# Take dot product with correlated noise, and compare with original noise realization:
println("Dot product of white noise:                          ",dot(y,y))
println("Dot product of correlated noise with inverse kernel: ",dot(corrnoise,x2))


plot(x2)

# Now use Ambikarasan O(N) method:

y = corrnoise

n = nt
nex = (2p+1)*n-2p
aex,bex = lorentz_likelihood_hermitian(alpha,beta,w,t,y);
aex_save = zeros(Complex{Float64},nex,nex)
for i=1:nex
  for j=1:nex
    aex_save[i,j]=aex[i,j]
  end
end
#eig(aex_save)
#chol_aex = cholesky_hermitian(aex)
#arec = *(chol_aex,ctranspose(chol_aex))
#println(maximum(abs(aex_save-arec)))
#read(STDIN, Char)

#aex_band,bex_band = lorentz_likelihood_hermitian_band(alpha,beta,w,t,y);
log_like = lorentz_likelihood_hermitian_band(alpha,beta,w,t,y);
bex_save = zeros(nex)
for i=1:nex
  bex_save[i]=bex[i]
end
bex2 = \(aex,bex_save)
#for i=1:nex
#  println(vec(real(aex[i,maximum([i-2p-1,1]):minimum([i+2p+1,nex])])))
#end

clf()
#PyPlot.imshow(real(aex), interpolation="nearest")
#PyPlot.imshow(imag(aex), interpolation="nearest")
#PyPlot.imshow(real(aex-transpose(aex)), interpolation="nearest")

# Now, compress this to a smaller matrix:
#tic()
#m1 = p+1
#m2 = p+1
#mm = m1 + m2 + 1
#aex_small = zeros(Complex{Float64},nex,mm)
#aex_small_save = zeros(Complex{Float64},nex,mm)
#al_small = zeros(Complex{Float64},nex,m1)
#for i=1:nex
#  for j=1:mm
##  aex_small[i,maximum([1,i-p-1])-i+p+2:minimum([nex,i+p+1])-i+p+2] = aex[i,maximum([1,i-p-1]):minimum([nex,i+p+1])]
#    joff = i-p-2+j
#    if joff >= 1 && joff <= nex
#      aex_small[i,j] = aex[i,joff]
#      aex_small_save[i,j] = aex[i,joff]
#    end
#  end
##  aex_small_save[i,maximum([1,i-p-1])-i+p+2:minimum([nex,i+p+1])-i+p+2] = aex[i,maximum([1,i-p-1]):minimum([nex,i+p+1])]
#end
#println(maximum(abs(aex_small - aex_band)))
#read(STDIN,Char)

#indx = collect(1:nex)
#d=bandec(aex_band,nex,m1,m2,al_small,indx)
#banbks(aex_band,nex,m1,m2,al_small,indx,bex)
#toc()
# Now select solution:
#x1 = zeros(n)
x3 = zeros(n)
log_like3 = 0.0
for i=1:n
#  x1[i]=real(bex[(i-1)*(2p+1)+1])
  x3[i]=real(bex2[(i-1)*(2p+1)+1])
  log_like3 += x3[i]*y[i]
end
#plot(x1)
log_like3 = -0.5 * log_like3

plot(x3)
# Compute the determinant:
#logdetofa = 0.0
#for i=1:nex
#  logdetofa += log(abs(aex_band[i,1]))
#end
logdetofa = real(logdet(aex))
log_like3 += -0.5*logdet(A)
println("Log Determinant: ",logdetofa," ",logdet(A))
println("Log Likelihood:  ",log_like," ",log_like3)
