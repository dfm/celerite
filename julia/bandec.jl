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
