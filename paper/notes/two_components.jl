using PyPlot
using Polynomials

# Compute PSD for a two-component Lorentzian with
# both complex coefficients.  Check that PSD is positive
# and plot it.

#psd_check(x)
x = rand(8)
# x=[ar1,ai1,br1,bi1,ar2,ai2,br2,bi2]

ar1=x[1]
ai1=x[2]
br1=x[3]
bi1=x[4]
ar2=x[5]
ai2=x[6]
br2=x[7]
bi2=x[8]
# Compute d=0 case 10% of time:
if rand() < 0.1
  bi2 = (ar1*br1+ar2*br2-ai1*bi1)/ai2
end

d = ar1*br1 - ai1*bi1 + ar2*br2 - ai2*bi2

a = (2(br2^2-bi2^2)*(ar1*br1-ai1*bi1)+2(br1^2-bi1^2)*(ar2*br2-ai2*bi2)
 +(bi1^2+br1^2)*(ai1*bi1+ar1*br1)+(bi2^2+br2^2)*(ai2*bi2+ar2*br2))

b = (2(br2^2-bi2^2)*(bi1^2+br1^2)*(ai1*bi1+ar1*br1)
+2(br1^2-bi1^2)*(bi2^2+br2^2)*(ai2*bi2+ar2*br2)
+(bi2^2+br2^2)^2*(ar1*br1-ai1*bi1)
+(bi1^2+br1^2)^2*(ar2*br2-ai2*bi2))

c = ((bi2^2+br2^2)^2*(bi1^2+br1^2)*(ai1*bi1+ar1*br1)
+(bi1^2+br1^2)^2*(bi2^2+br2^2)*(ai2*bi2+ar2*br2))

if abs(d) < eps()
  d = 0.0
  println("Reached quadratic test")
  disc = b^2-4a*c
  if disc < 0
# there is no z_max, so set it to a negative value:
    zmax = -1.0
  else
    if a > 0
      zmax = (-b+sqrt(disc))/(2*a)
    else
      zmax = (-b-sqrt(disc))/(2*a)
    end
  end
else

  a /= d
  b /= d
  c /= d

  delta = 18a*b*c-4a^3*c+a^2*b^2-4b^3-27c^2

  if delta >= 0
    disc = sqrt(a^2-3b)
    theta = acos(0.5*(2a^3-9a*b+27c)/disc^3)
    zmax = -2/3*disc*cos((mod((theta+pi),2pi)+pi)/3)-a/3
  else
    r = (2a^3-9a*b+27c)/54
    q = (a^2-3b)/9
    aa = -(abs(r)+sqrt(r^2-q^3))^(1/3)
    if r < 0
      aa = -aa
    end
    if aa == 0
      bb = 0
    else
      bb = q/aa
    end
    zmax = aa+bb-a/3
  end
end

if zmax < 0
  println("Positive definite spectrum.  Maximum omega^2: ",zmax)
else
  println("Negative spectrum.  Maximum omega^2: ",zmax)
end

omega = linspace(0,20,100000)

psd = sqrt(2/pi)*((omega.^2.*(ar1*br1-ai1*bi1)+(ai1*bi1+ar1*br1)*(bi1^2+br1^2))./(omega.^4+2(br1^2-bi1^2).*omega.^2 + (bi1^2+br1^2)^2)
+(omega.^2.*(ar2*br2-ai2*bi2)+(ai2*bi2+ar2*br2)*(bi2^2+br2^2))./(omega.^4+2(br2^2-bi2^2).*omega.^2 + (bi2^2+br2^2)^2))


fig,axes = subplots(2,1)
ax = axes[1]
ax[:plot](omega,psd)
if zmax > 0
  ax[:scatter](sqrt(zmax),0)
end
# Now plot ACF:

tau = linspace(0,10,100000)
acf = (exp(-br1.*tau).*(ar1.*cos(bi1.*tau)+ai1.*sin(bi1.*tau))
+ exp(-br2.*tau).*(ar2.*cos(bi2.*tau)+ai2.*sin(bi2.*tau)))

ax = axes[2]
ax[:plot](tau,acf)

println(minimum(psd))

# Now, try Sturm's approach:

f0 = Poly([c,b,a,1])
f1 = polyder(f0)
f2 = -rem(f0,f1)
f3 = -rem(f1,f2)
#f4 = -rem(f2,f3)

f_of_0 = [f0(0),f1(0),f2(0),f3(0)]
f_of_inf = [f0[3],f1[2],f2[1],f3[0]]
sig_0 = 0
for i=1:3
  if f_of_0[i+1]*f_of_0[i] < 0
    sig_0 +=1
  end
end
sig_inf = 0
for i=1:3
  if f_of_inf[i+1]*f_of_inf[i] < 0
    sig_inf +=1
  end
end
println("Number of positive roots: ",sig_0-sig_inf)
