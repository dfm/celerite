function simulate_exp_gp(t,alpha,beta,ndev)
#
# Function for simulating a GP based upon an exponential correlation
# matrix using an analytic form for the Cholesky decomposition and
# a computation in O(N) operations.  The autocorrelation function is:
#
#    K_ij = alpha * exp(-beta*(|t_i-t_j|))
#
# Requirements:
#  - alpha should be a real positive number.
#  - The times t *must* be sorted in order from least to greatest.
#  - beta may be complex
#  - ndev: normal deviates drawn from N(0,1) with the same length as time vector t.
# Output:
#  - data is a GP drawn from this correlation function with length nt
#
nt = length(t)
data = zeros(eltype(beta),nt)
data[nt] = sqrt(alpha)*ndev[nt]
gamma = zero(eltype(beta))
for i=nt-1:-1:1
  gamma = exp(-beta*(t[i+1]-t[i]))
  println(i," ",abs(sqrt(1.0-gamma^2)))
  data[i] = sqrt(1.0-gamma^2)*sqrt(alpha)*ndev[i]+gamma*data[i+1]
end
return data
end
