# LU back-substitution in Julia (ala Numerical Recipes)
function banbks(a::Array{Float64,2},n::Int64,m1::Int64,m2::Int64,al::Array{Float64,2},indx::Vector{Int64},b::Vector{Float64})
mm=m1+m2+1
@assert(size(a)  == (n,mm))
@assert(size(al) == (n,m1))
@assert(length(indx) == n)
@assert(length(b) == n)
l=m1
for k=1:n
  i=indx[k]
  if (i != k)
    dum  = b[k]
    b[k] = b[i]
    b[i] = dum
  end
  if (l < n)
    l += 1
  end
  for i=k+1:l
    b[i] -= al[k,i-k]*b[k]
  end
end
l=1
for i in n:-1:1
  dum=b[i]
  for k=2:l
    dum -= a[i,k]*b[k+i-1]
  end
  b[i]=dum/a[i,1]
  if (l < mm) 
    l += 1
  end
end
return
end

# (C) Copr. 1986-92 Numerical Recipes Software

function banbks(a::Array{Complex{Float64},2},n::Int64,m1::Int64,m2::Int64,al::Array{Complex{Float64},2},indx::Vector{Int64},b::Vector{Complex{Float64}})
mm=m1+m2+1
@assert(size(a)  == (n,mm))
@assert(size(al) == (n,m1))
@assert(length(indx) == n)
@assert(length(b) == n)
l=m1
for k=1:n
  i=indx[k]
  if (i != k)
    dum  = b[k]
    b[k] = b[i]
    b[i] = dum
  end
  if (l < n)
    l += 1
  end
  for i=k+1:l
    b[i] -= al[k,i-k]*b[k]
  end
end
l=1
for i in n:-1:1
  dum=b[i]
  for k=2:l
    dum -= a[i,k]*b[k+i-1]
  end
  b[i]=dum/a[i,1]
  if (l < mm) 
    l += 1
  end
end
return
end

# (C) Copr. 1986-92 Numerical Recipes Software

function banbks(a::Array,n::Int64,m1::Int64,m2::Int64,al::Array,indx::Vector{Int64},b::Vector)
mm=m1+m2+1
@assert(size(a)  == (n,mm))
@assert(size(al) == (n,m1))
@assert(length(indx) == n)
@assert(length(b) == n)
l=m1
dum = zero(eltype(a))
for k=1:n
  i=indx[k]
  if (i != k)
    dum  = b[k]
    b[k] = b[i]
    b[i] = dum
  end
  if (l < n)
    l += 1
  end
  for i=k+1:l
    b[i] -= al[k,i-k]*b[k]
  end
end
l=1
for i in n:-1:1
  dum=b[i]
  for k=2:l
    dum -= a[i,k]*b[k+i-1]
  end
  b[i]=dum/a[i,1]
  if (l < mm) 
    l += 1
  end
end
return
end

# (C) Copr. 1986-92 Numerical Recipes Software

function banbks(a1::Array,a2::Array,n::Int64,m1::Int64,m2::Int64,al1::Array,al2::Array,indx::Vector{Int64},b1::Vector)
# This does complex back substitution, but with real arrays to be compatible with ForwardDiff
mm=m1+m2+1
#@assert(size(a1)  == (n,mm))
#@assert(size(a2)  == (n,mm))
#@assert(size(al1) == (n,m1))
#@assert(size(al2) == (n,m1))
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
    b1[i] -= al1[k,i-k]*b1[k] - al2[k,i-k]*b2[k]
    b2[i] -= al1[k,i-k]*b2[k] + al2[k,i-k]*b1[k]
  end
end
l=1
for i in n:-1:1
  dum1=b1[i]
  dum2=b2[i]
  for k=2:l
    ac = a1[i,k]*b1[k+i-1]
    bd = a2[i,k]*b2[k+i-1]
    dum1 -= ac - bd
    dum2 -= (a1[i,k]+a2[i,k])*(b1[k+i-1] + b2[k+i-1]) - ac - bd
  end
  if (abs(a1[i,1]) >= abs(a2[i,1]))
    rat = a2[i,1]/a1[i,1]
    den = a1[i,1]+a2[i,1]*rat
    b1[i]=(dum1+dum2*rat)/den
    b2[i]=(dum2-dum1*rat)/den
  else
    rat = a1[i,1]/a2[i,1]
    den = a1[i,1]*rat+a2[i,1]
    b1[i]=(dum1*rat+dum2)/den
    b2[i]=(dum2*rat-dum1)/den
  end
  if (l < mm)
    l += 1
  end
end
return
end
