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
