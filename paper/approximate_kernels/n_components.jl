using PyPlot
using Polynomials
include("sturms_theorem.jl")

# Compute power-spectrum & number of positive zeros:

# Choose an arbitrary number of Lorentzians from 2-10:
n_lor = rand(2:10)

# Set up random coefficients in a vector:
x = zeros(4*n_lor)
# Choose the maximum time & frequency for plotting purposes:
max_time = 0
max_freq = 0

# Randomly choose coefficients.
# These are in the order:  x = {(ar_j,ai_j,br_j,bi_j); j=1,...,n_lor}
for i=1:n_lor
# Linearly distribute ar & ai from -1 to 1:
  x[(i-1)*4+1]=(rand()-.5)*2
  x[(i-1)*4+2]=(rand()-.5)*2
# Logarithmically distribute br & bi:
  x[(i-1)*4+3]=10.^(rand()*2-1)
  x[(i-1)*4+4]=10.^(rand()*2-1)
# Set range of plot to capture behavior of these:
  if 1/x[(i-1)*4+3] > max_time
    max_time = 1/x[(i-1)*4+3]
  end
  if x[(i-1)*4+3] > max_freq
    max_freq = x[(i-1)*4+3]
  end
  if 1/x[(i-1)*4+4] > max_time
    max_time = 1/x[(i-1)*4+4]
  end
  if x[(i-1)*4+4] > max_freq
    max_freq = x[(i-1)*4+4]
  end
end


# The order of the numerator polynomial in z = \omega^2 that we need to
# see if it has any zeros:
pord = 2*(n_lor-1) + 1


# Compute coefficients in the numerator & denominator of PSD:
# The PSD is given by:
#
# \sum_i (a_i z +b_i)/(z^2 + c_i*z + d_i)
#
# where z = \omega^2
#
# We are computing the number of positive real zeros of this equation.
#
a = zeros(n_lor)
b = zeros(n_lor)
c = zeros(n_lor)
d = zeros(n_lor)

for i=1:n_lor
  ar_j = x[(i-1)*4+1]
  ai_j = x[(i-1)*4+2]
  br_j = x[(i-1)*4+3]
  bi_j = x[(i-1)*4+4]
  a[i] = ar_j*br_j-ai_j*bi_j
  b[i] = (bi_j^2+br_j^2)*(ai_j*bi_j+ar_j*br_j)
  c[i] = 2*(br_j^2-bi_j^2)
  d[i] = (bi_j^2+br_j^2)^2
end

# Now, loop over coefficients:

# Initialize a polynomial:
p0 = Poly(zeros(pord+1))
for i=1:n_lor
# The polynomial for the current Lorentzian term in the common-denominator expression:
  pcur = Poly([b[i],a[i]])
  for j=1:n_lor
# Only multiply by the denominators from the other Lorentzians:
    if j != i
      pcur *= Poly([d[j],c[j],1])
    end
  end
  p0 += pcur
end

# Compute the roots of this polynomial:
poly_root = roots(p0)

# Now that we've computed coefficients of the polynomial, we just need
# to apply Sturm's theorem!

# Set up an array to hold signs of polynomial at zero & infinity:
f_of_0 = zeros(pord+1)
f_of_inf = zeros(pord+1)

# Take the derivative of the polynomial:
p1 = polyder(p0)
#println(p0)
#println(p1)
# Insert the coefficients of the z^0 term:
f_of_0[1] = p0(0)
f_of_0[2] = p1(0)
# Insert the coefficient of the z^(p_ord-i) term:
f_of_inf[1] = p0[pord]
f_of_inf[2] = p1[pord-1]

# Now, loop over the Sturm polynomial series:
for i=3:pord+1
  p2 = -rem(p0,p1)
# Check that round-off error hasn't left us with a polynomial
# that is the same order as p0 or p1, but with a small coefficient:
  if length(p2) >= length(p1)
    coeff = zeros(pord-i+2)
    for j=0:pord-i+1
      coeff[j+1]=p2[j]
    end
    p2 = Poly(coeff)
  end
# Insert the z^0 term:
  f_of_0[i]=p2(0)
# Insert the z -> \infty term:
  f_of_inf[i]=p2[pord-i+1]
# Now move promote these polynomials, readying them for recursion in the next step:
  p0=copy(p1)
  p1=copy(p2)
end

# Now we'll compute the number of sign changes at z=0:
sig_0 = 0
for i=1:pord
#  if f_of_0[i+1]*f_of_0[i] < 0 && abs(f_of_0[i+1]) > eps() && abs(f_of_0[i]) > eps()
  if sign(f_of_0[i+1]) != sign(f_of_0[i])
    sig_0 +=1
  end
end
# Next, compute the number of sign changes at z=\infty:
sig_inf = 0
for i=1:pord
#  if f_of_inf[i+1]*f_of_inf[i] < 0 && abs(f_of_inf[i+1]) > eps() && abs(f_of_inf[i]) > eps()
  if sign(f_of_inf[i+1]) != sign(f_of_inf[i])
    sig_inf +=1
  end
end

# The difference between z=0 & z=\infty gives the number of positive, real roots:
println("Number of positive, real roots: ",sig_0-sig_inf)
println("Result  of sturms_theorem.jl:   ",sturms_theorem(x))

# Print out the actual roots for inspection:
println("Roots: ",poly_root)

# Now, determine which roots are positive & real from the numerial solver:
n_pos_real = 0
real_root = Float64[]
for i=1:pord
  if real(poly_root[i]) > 0 && imag(poly_root[i]) == 0
    n_pos_real +=1
    poly_root0 = sqrt(real(poly_root[i]))
    real_root = [real_root;poly_root0]
    if poly_root0 >  max_freq
      max_freq = poly_root0
    end
  end
end
println("Count of positive, real roots: ",n_pos_real)

# Finally, plot the power spectrum & auto-correlation functions:
nom = 100000
omega = linspace(0,2max_freq,nom)

psd = zeros(nom)
ntau = 100000
tau = linspace(0,max_time*4*pi,ntau)
acf = zeros(ntau)

# Compute PSD & ACF:
for i=1:n_lor
  ar_j = x[(i-1)*4+1]
  ai_j = x[(i-1)*4+2]
  br_j = x[(i-1)*4+3]
  bi_j = x[(i-1)*4+4]
  psd += sqrt(2/pi)*((omega.^2.*(ar_j*br_j-ai_j*bi_j)+(ai_j*bi_j+ar_j*br_j)*(bi_j^2+br_j^2))./(omega.^4+2(br_j^2-bi_j^2).*omega.^2 + (bi_j^2+br_j^2)^2))
  acf += (exp(-br_j.*tau).*(ar_j.*cos(bi_j.*tau)+ai_j.*sin(bi_j.*tau)))
end

# Make the plots:
fig,axes = subplots(2,1)
ax = axes[1]
ax[:plot](omega,psd)
# Overplot the numerically computed roots:
if n_pos_real > 0
  for i=1:n_pos_real
    ax[:scatter](real_root[i],0)
  end
end
# Now plot ACF:
ax = axes[2]
ax[:plot](tau,acf)

#println(minimum(psd))
