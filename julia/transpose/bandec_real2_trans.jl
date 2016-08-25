function bandec_real2_trans(a1::Array,a2::Array,n::Int64,m1::Int64,m2::Int64,al1::Array,al2::Array,indx::Vector{Int64})
# This version does complex arithmetic, but with real arrays to be compatible with ForwardDiff.
# a1 is real part of a; a2 is imaginary part of a
# al1 is real part al; al2 is imaginary part of al
#      INTEGER m1,m2,mp,mpl,n,np,indx(n)
#      REAL d,a(np,mp),al(np,mpl),TINY
mm=m1+m2+1
#@assert(size(a1)  == (n,mm))
#@assert(size(a2)  == (n,mm))
#@assert(size(al1) == (n,m1))
#@assert(size(al2) == (n,m1))
#@assert(length(indx) == n)
TINY = one(eltype(a1))*1e-20
#      INTEGER i,j,k,l,mm
#      REAL dum
l=m1
czero = zero(eltype(a1))
# dum = dum1 + im*dum2
dum1 = czero
dum2 = czero
den = czero
rat = czero
ac = czero
bd = czero
for i=1:m1
  for j=m1+2-i:mm
    a1[j-l,i]=a1[j,i]
    a2[j-l,i]=a2[j,i]
  end
  l -= 1
  for j=mm-l:mm
    a1[j,i]=czero
    a2[j,i]=czero
  end
end
d = one(eltype(a1))
l=m1
for k=1:n
  dum1=a1[1,k]
  dum2=a2[1,k]
  i=k
  if (l < n)
    l += 1
  end
  for j=k+1:l
    if ((a1[1,j]*a1[1,j]+a2[1,j]*a2[1,j]) > (dum1*dum1+dum2*dum2))
      dum1=a1[1,j]
      dum2=a2[1,j]
      i=j
    end
  end
  indx[k]=i
  if (dum1 == czero && dum2 == czero)
    a1[1,k]=TINY
    a2[1,k]=TINY
  end
  if (i != k)
    d = -d
    for j=1:mm
      dum1=a1[j,k]
      dum2=a2[j,k]
      a1[j,k]=a1[j,i]
      a2[j,k]=a2[j,i]
      a1[j,i]=dum1
      a2[j,i]=dum2
    end
  end
  for i=k+1:l
# Denominator for complex division:
# Use complex division from Numerical Recipes (5.4.5):
    if (abs(a1[1,k]) >= abs(a2[1,k]))
      rat = a2[1,k]/a1[1,k]
      den = a1[1,k]+a2[1,k]*rat
      dum1=(a1[1,i]+a2[1,i]*rat)/den
      dum2=(a2[1,i]-a1[1,i]*rat)/den
    else
      rat = a1[1,k]/a2[1,k]
      den = a1[1,k]*rat + a2[1,k]
      dum1=(a1[1,i]*rat+a2[1,i])/den
      dum2=(a2[1,i]*rat-a1[1,i])/den
    end
    al1[i-k,k]=dum1
    al2[i-k,k]=dum2
    for j=2:mm
      ac = dum1*a1[j,k]
      bd = dum2*a2[j,k]
      a1[j-1,i] = a1[j,i]
      a1[j-1,i] -= ac
      a1[j-1,i] += bd
      a2[j-1,i] = a2[j,i]
      a2[j-1,i] -= (dum1+dum2)*(a1[j,k]+a2[j,k])
      a2[j-1,i] += ac
      a2[j-1,i] += bd
    end
    a1[mm,i]=czero
    a2[mm,i]=czero
  end
end
return d
end
