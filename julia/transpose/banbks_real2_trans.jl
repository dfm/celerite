function banbks_real2_trans(a1::Array,a2::Array,n::Int64,m1::Int64,m2::Int64,al1::Array,al2::Array,indx::Vector{Int64},b1::Vector)
# This does complex back substitution, but with real arrays to be compatible with ForwardDiff
mm=m1+m2+1
#@assert(size(a1)  == (mm,n))
#@assert(size(a2)  == (mm,n))
#@assert(size(al1) == (m1,n))
#@assert(size(al2) == (m1,n))
#@assert(length(indx) == n)
#@assert(length(b1) == n)
l=m1
# Define 'dummy' for number swaps:
czero = zero(eltype(a1))
dum1 = czero
dum2 = czero
# Denominator for complex division:
den = czero
rat = czero
ac = czero
bd = czero
# b1 is real, so we need to create the complex vector
b2 = zeros(eltype(a1),n)
for k=1:n
  i=indx[k]
  if (i != k)
    dum1  = b1[k]
    b1[k] = b1[i]
    b1[i] = dum1
    dum2  = b2[k]
    b2[k] = b2[i]
    b2[i] = dum2
  end
  if (l < n)
    l += 1
  end
  for i=k+1:l
    b1[i] -= al1[i-k,k]*b1[k] - al2[i-k,k]*b2[k]
    b2[i] -= al1[i-k,k]*b2[k] + al2[i-k,k]*b1[k]
  end
end
l=1
for i in n:-1:1
  dum1=b1[i]
  dum2=b2[i]
  for k=2:l
    ac = a1[k,i]*b1[k+i-1]
    bd = a2[k,i]*b2[k+i-1]
    dum1 -= ac - bd
    dum2 -= (a1[k,i]+a2[k,i])*(b1[k+i-1] + b2[k+i-1]) - ac - bd
  end
  if (abs(a1[1,i]) >= abs(a2[1,i]))
    rat = a2[1,i]/a1[1,i]
    den = a1[1,i]+a2[1,i]*rat
    b1[i]=(dum1+dum2*rat)/den
    b2[i]=(dum2-dum1*rat)/den
  else
    rat = a1[1,i]/a2[1,i]
    den = a1[1,i]*rat+a2[1,i]
    b1[i]=(dum1*rat+dum2)/den
    b2[i]=(dum2*rat-dum1)/den
  end
  if (l < mm)
    l += 1
  end
end
return
end
