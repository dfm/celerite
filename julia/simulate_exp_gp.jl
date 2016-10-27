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
gamma = zeros(eltype(beta),nt)
for i=1:nt-1
  gamma[i] = exp(-beta*(t[i+1]-t[i]))
  data[i] = sqrt(1.0-gamma[i]^2)*sqrt(alpha)*ndev[i]
end
data[nt] = sqrt(alpha)*ndev[nt]
for i=nt-1:-1:1
  data[i] += gamma[i]*data[i+1]
end
return data
end
