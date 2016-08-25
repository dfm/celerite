# LU decomposition of a band-diagonal matrix (ala Numerical Recipes)
#
function bandec(a::Array{Float64,2},n::Int64,m1::Int64,m2::Int64,al::Array{Float64,2},indx::Vector{Int64})
#      INTEGER m1,m2,mp,mpl,n,np,indx(n)
#      REAL d,a(np,mp),al(np,mpl),TINY
mm=m1+m2+1
@assert(size(a)  == (n,mm)) 
@assert(size(al) == (n,m1)) 
@assert(length(indx) == n) 
TINY=1.e-20
#      INTEGER i,j,k,l,mm
#      REAL dum
l=m1
dum = zero(eltype(a))
for i=1:m1
  for j=m1+2-i:mm
    a[i,j-l]=a[i,j]
  end
  l -= 1
  for j=mm-l:mm
    a[i,j]=zero(eltype(a))
  end
end
d=one(eltype(a))
l=m1
for k=1:n
  dum=a[k,1]
  i=k
  if (l < n) 
    l += 1
  end
  for j=k+1:l
    if (abs(a[j,1]) > abs(dum))
      dum=a[j,1]
      i=j
    end
  end
  indx[k]=i
  if (dum == zero(eltype(a))) 
    a[k,1]=TINY
  end
  if (i != k)
    d = -d
    for j=1:mm
      dum=a[k,j]
      a[k,j]=a[i,j]
      a[i,j]=dum
    end
  end
  for i=k+1:l
    dum=a[i,1]/a[k,1]
    al[k,i-k]=dum
    for j=2:mm
#      a[i,j-1] = a[i,j] - dum*a[k,j]
      a[i,j-1] = a[i,j]
      a[i,j-1] -= dum*a[k,j]
    end
    a[i,mm]=zero(eltype(a))
  end
end
return d
end

#  (C) Copr. 1986-92 Numerical Recipes Software

function bandec(a::Array{Complex{Float64},2},n::Int64,m1::Int64,m2::Int64,al::Array{Complex{Float64},2},indx::Vector{Int64})
#      INTEGER m1,m2,mp,mpl,n,np,indx(n)
#      REAL d,a(np,mp),al(np,mpl),TINY
mm=m1+m2+1
@assert(size(a)  == (n,mm)) 
@assert(size(al) == (n,m1)) 
@assert(length(indx) == n) 
TINY=complex(1.e-20,0.0)
#      INTEGER i,j,k,l,mm
#      REAL dum
l=m1
czero::Complex{Float64} = complex(0.0,0.0)
dum::Complex{Float64} = czero
for i=1:m1
  for j=m1+2-i:mm
    a[i,j-l]=a[i,j]
  end
  l -= 1
  for j=mm-l:mm
    a[i,j]=czero
  end
end
d::Float64 =1.0
l=m1
for k=1:n
  dum=a[k,1]
  i=k
  if (l < n) 
    l += 1
  end
  for j=k+1:l
    if (abs(a[j,1]) > abs(dum))
      dum=a[j,1]
      i=j
    end
  end
  indx[k]=i
  if (dum == czero) 
    a[k,1]=TINY
  end
  if (i != k)
    d = -d
    for j=1:mm
      dum=a[k,j]
      a[k,j]=a[i,j]
      a[i,j]=dum
    end
  end
  for i=k+1:l
    dum=a[i,1]/a[k,1]
    al[k,i-k]=dum
    for j=2:mm
      a[i,j-1] = a[i,j] - dum*a[k,j]
    end
    a[i,mm]=czero
  end
end
return d
end

#  (C) Copr. 1986-92 Numerical Recipes Software

function bandec(a::Array,n::Int64,m1::Int64,m2::Int64,al::Array,indx::Vector{Int64})
#      INTEGER m1,m2,mp,mpl,n,np,indx(n)
#      REAL d,a(np,mp),al(np,mpl),TINY
mm=m1+m2+1
@assert(size(a)  == (n,mm)) 
@assert(size(al) == (n,m1)) 
@assert(length(indx) == n) 
TINY=1.e-20
#      INTEGER i,j,k,l,mm
#      REAL dum
l=m1
dum = zero(eltype(a))
for i=1:m1
  for j=m1+2-i:mm
    a[i,j-l]=a[i,j]
  end
  l -= 1
  for j=mm-l:mm
    a[i,j]=zero(eltype(a))
  end
end
d=one(eltype(a))
l=m1
for k=1:n
  dum=a[k,1]
  i=k
  if (l < n) 
    l += 1
  end
  for j=k+1:l
    if (abs(a[j,1]) > abs(dum))
      dum=a[j,1]
      i=j
    end
  end
  indx[k]=i
  if (dum == zero(eltype(a))) 
    a[k,1]=TINY
  end
  if (i != k)
    d = -d
    for j=1:mm
      dum=a[k,j]
      a[k,j]=a[i,j]
      a[i,j]=dum
    end
  end
  for i=k+1:l
    dum=a[i,1]/a[k,1]
    al[k,i-k]=dum
    for j=2:mm
#      a[i,j-1] = a[i,j] - dum*a[k,j]
      a[i,j-1]  = a[i,j]
      a[i,j-1] -= dum*a[k,j]
    end
    a[i,mm]=0.
  end
end
return d
end

#  (C) Copr. 1986-92 Numerical Recipes Software

function bandec(a1::Array,a2::Array,n::Int64,m1::Int64,m2::Int64,al1::Array,al2::Array,indx::Vector{Int64})
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
    a1[i,j-l]=a1[i,j]
    a2[i,j-l]=a2[i,j]
  end
  l -= 1
  for j=mm-l:mm
    a1[i,j]=czero
    a2[i,j]=czero
  end
end
d = one(eltype(a1))
l=m1
for k=1:n
  dum1=a1[k,1]
  dum2=a2[k,1]
  i=k
  if (l < n)
    l += 1
  end
  for j=k+1:l
    if ((a1[j,1]*a1[j,1]+a2[j,1]*a2[j,1]) > (dum1*dum1+dum2*dum2))
      dum1=a1[j,1]
      dum2=a2[j,1]
      i=j
    end
  end
  indx[k]=i
  if (dum1 == czero && dum2 == czero)
    a1[k,1]=TINY
    a2[k,1]=TINY
  end
  if (i != k)
    d = -d
    for j=1:mm
      dum1=a1[k,j]
      dum2=a2[k,j]
      a1[k,j]=a1[i,j]
      a2[k,j]=a2[i,j]
      a1[i,j]=dum1
      a2[i,j]=dum2
    end
  end
  for i=k+1:l
# Denominator for complex division:
# Use complex division from Numerical Recipes (5.4.5):
    if (abs(a1[k,1]) >= abs(a2[k,1]))
      rat = a2[k,1]/a1[k,1]
      den = a1[k,1]+a2[k,1]*rat
      dum1=(a1[i,1]+a2[i,1]*rat)/den
      dum2=(a2[i,1]-a1[i,1]*rat)/den
    else
      rat = a1[k,1]/a2[k,1]
      den = a1[k,1]*rat + a2[k,1]
      dum1=(a1[i,1]*rat+a2[i,1])/den
      dum2=(a2[i,1]*rat-a1[i,1])/den
    end
    al1[k,i-k]=dum1
    al2[k,i-k]=dum2
    for j=2:mm
      ac = dum1*a1[k,j]
      bd = dum2*a2[k,j]
      a1[i,j-1] = a1[i,j]
      a1[i,j-1] -= ac
      a1[i,j-1] += bd
      a2[i,j-1] = a2[i,j]
      a2[i,j-1] -= (dum1+dum2)*(a1[k,j]+a2[k,j])
      a2[i,j-1] += ac
      a2[i,j-1] += bd
    end
    a1[i,mm]=czero
    a2[i,mm]=czero
  end
end
return d
end
