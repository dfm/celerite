function bandec_trans(a::Array,n::Int64,m1::Int64,m2::Int64,al::Array,indx::Vector{Int64})
#      INTEGER m1,m2,mp,mpl,n,np,indx(n)
#      REAL d,a(np,mp),al(np,mpl),TINY
mm=m1+m2+1
@assert(size(a)  == (mm,n)) 
@assert(size(al) == (m1,n)) 
@assert(length(indx) == n) 
TINY = one(eltype(a))*1.e-20
#      INTEGER i,j,k,l,mm
#      REAL dum
l=m1
czero = zero(eltype(a))
dum = czero
for i=1:m1
  for j=m1+2-i:mm
    a[j-l,i]=a[j,i]
  end
  l -= 1
  for j=mm-l:mm
    a[j,i]=czero
  end
end
d = one(eltype(a))
l=m1
for k=1:n
  dum=a[1,k]
  i=k
  if (l < n) 
    l += 1
  end
  for j=k+1:l
    if (abs(a[1,j]) > abs(dum))
      dum=a[1,j]
      i=j
    end
  end
  indx[k]=i
  if (dum == czero) 
    a[1,k]=TINY
  end
  if (i != k)
    d = -d
    for j=1:mm
      dum=a[j,k]
      a[j,k]=a[j,i]
      a[j,i]=dum
    end
  end
  for i=k+1:l
    dum=a[1,i]/a[1,k]
    al[i-k,k]=dum
    for j=2:mm
      a[j-1,i] = a[j,i] - dum*a[j,k]
    end
    a[mm,i]=czero
  end
end
return d
end

function bandec_trans(a::Array{Complex{Float64},2},n::Int64,m1::Int64,m2::Int64,al::Array{Complex{Float64},2},indx::Vector{Int64})
#      INTEGER m1,m2,mp,mpl,n,np,indx(n)
#      REAL d,a(np,mp),al(np,mpl),TINY
mm=m1+m2+1
@assert(size(a)  == (mm,n)) 
@assert(size(al) == (m1,n)) 
@assert(length(indx) == n) 
TINY=complex(1.e-20,0.0)
#      INTEGER i,j,k,l,mm
#      REAL dum
l=m1
czero::Complex{Float64} = complex(0.0,0.0)
dum::Complex{Float64} = czero
for i=1:m1
  for j=m1+2-i:mm
    a[j-l,i]=a[j,i]
  end
  l -= 1
  for j=mm-l:mm
    a[j,i]=czero
  end
end
d::Float64 =1.0
l=m1
for k=1:n
  dum=a[1,k]
  i=k
  if (l < n) 
    l += 1
  end
  for j=k+1:l
    if (abs(a[1,j]) > abs(dum))
      dum=a[1,j]
      i=j
    end
  end
  indx[k]=i
  if (dum == czero) 
    a[1,k]=TINY
  end
  if (i != k)
    d = -d
    for j=1:mm
      dum=a[j,k]
      a[j,k]=a[j,i]
      a[j,i]=dum
    end
  end
  for i=k+1:l
    dum=a[1,i]/a[1,k]
    al[i-k,k]=dum
    for j=2:mm
      a[j-1,i] = a[j,i] - dum*a[j,k]
    end
    a[mm,i]=czero
  end
end
return d
end
