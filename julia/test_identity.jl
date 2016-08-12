# Tests the identity from Tegmark et al. (1997), just before
# equation 10:

nt = 10
C = zeros(nt,nt)
tau = 3.0
alpha = 10.0
for i=1:nt
  for j=1:nt
    C[i,j] = alpha*exp(-abs(i-j)/tau)
  end
end

logdetC = logdet(C)
test = trace(log(C))
println(logdetC, " ",test)


