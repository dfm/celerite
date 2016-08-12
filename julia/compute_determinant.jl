# I may have found a determinant formula for the single Lorentzian case!
# But that's not good enough for the general case...

nt = 1000
C = zeros(Complex{Float64},nt,nt)
nlor = 3
beta = [0.3+im*0.5, 0.3-im*0.5, 0.1+im*0.0]
alpha = [1.0, 1.0, 2.5]
t = linspace(0,10,nt)
for i=1:nt
  for j=1:nt
    for k=1:nlor
      C[i,j] += alpha[k]*exp(-abs(t[i]-t[j])*beta[k])
    end
  end
end

Cinv = inv(C)
lower = 0.0
upper = 0.0
for i=1:nt
  lower += Cinv[i,i]
  upper += C[i,i]
end
lower = nt/lower
upper = upper/nt

logdetC_num = logdet(C)

# Now use analytic formula:

logdetC_ana = zeros(Complex{Float64},nlor)

for k=1:nlor
  logdetC_ana[k] += nt*log(alpha[k])
  for i=1:nt-1
    logdetC_ana[k] += log(1.0-exp(-2.0*beta[k]*(t[i+1]-t[i])))
  end  
end

#println(logdetC_num," ",lower," ",upper)
#println(logdetC_num," ",nt*log(lower)," ",nt*log(upper))
println(logdetC_num/nt," ",log(exp(logdetC_ana[1]/nt)+exp(logdetC_ana[2]/nt)+exp(logdetC_ana[3])))

# But, the general case is more complicated since there is not a simple formula for det(A+B).
#  See: https://en.wikipedia.org/wiki/Matrix_determinant_lemma
# Look here: http://mathoverflow.net/questions/65424/determinant-of-sum-of-positive-definite-matrices

# The Minkowski determinant theorem may be used to place a bound on the determinant that becomes
# tigher as the it gets larger!  I'm not sure if this applies to derivative as well, but I suspect
# it does:
# http://mathoverflow.net/questions/65424/determinant-of-sum-of-positive-definite-matrices?rq=1

# Also see: http://mathoverflow.net/questions/42594/concavity-of-det1-n-over-hpd-n

# Now try to come up with an analytic expression:

#using SymPy
#x = symbols("x")
#a = [x 1; 1 x]
#a[:det]()
