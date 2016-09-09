# Compute the derivative of the determinant with AutoDiff.

using ForwardDiff
using ReverseDiffSource

function derivative_determinant(x,n)
# Closure for computing the determinant:
  
  function compute_determinant(x::Vector)
  matrix = zeros(Real,n,n)
# The x vector contains 3 parameters for each Lorentzian:
#   1). log_amplitude:  alpha = exp(log_amplitude)
#   2). log_qafactor: beta.real = exp(-log_qfactor)
#   3). log_frequency: beta.imag = 2*pi*exp(log_frequency)
# So, assert that the length of x is a multiple of 3:
#    @assert(mod(length(x),3) == 0)
# For now I'm going to try this with a single Lorentzian:
# The number of Lorentzians to model is a multiple of 3:
#    nlor = length(x)/3 
  nlor = 1
# Note: for every non-zero value of the imaginary part of the lorentzian,
# there should be a complex conjugate.
  for i=1:n
    matrix[i,i] = x[1]
    for j=1:i-1
      matrix[i,j] = x[1]*exp(-x[2]*(i-j))
    end
    for j=i+1:n
      matrix[i,j] = x[1]*exp(-x[2]*(j-i))
    end
  end
  lumat = lufact!(matrix)
#  println(typeof(lumat))
#  println(typeof(lumat))
  return logdet(lumat)
  end
 
  println(compute_determinant(x)) 
  result = HessianResult(x)
  ForwardDiff.hessian!(result,compute_determinant,x)
  println(result)
  println("Determinant:  ",ForwardDiff.value(result))
  println("Gradient:     ",ForwardDiff.gradient(result))
  println("Hessian:      ",ForwardDiff.hessian(result))
# Now compute these analytically:
  logdet_ana = n*log(x[1]) + (n-1)*log(1.0-exp(-2.0*x[2]))
  logdet_grad = zeros(2)
  logdet_grad[1] = n/x[1]
  logdet_grad[2] = (n-1)/(1.0-exp(-2.0*x[2]))*(2.0*exp(-2.0*x[2]))
  println("Analytic det:  ",logdet_ana)
  println("Analytic grad: ",logdet_grad)
# Now see if I can use ReverseDiff:
#  diff_code = rdiff(compute_determinant)

#  println(typeof(diff_code))
  return result
end 
