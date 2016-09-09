# Implements linear fitting (analagous to IDL's regress function)

# Dependent in the vector Y, which should have length N_data
# N_var independent variables are contained in the array X.
# X should have the size [N_var,N_data].

function regress(X,Y,sigY)
# Check that X has two dimensions, and that
# X and Y have are compatible (second dimension of X
# should equal the dimension of Y):
nx = ndims(X)
sx = size(X)
sy = size(Y)
if (nx == 2) & (sx[2] == sy[1]) then
# Set up array for computing coefficients:
  A = zeros(sx[1],sx[1])
  B = zeros(sx[1])
# Compute A = \sum_i \sum_j \sum_k X_{i,k} X_{j,k}/\sigY_k^2
  for i = 1:sx[1]
    for j = 1:sx[1]
      for k = 1:sx[2]
        A[i,j] += X[i,k]*X[j,k]/sigY[k]^2
      end
    end
# Compute B = \sum_i \sum_k X_{i,k} Y_k/\sigY_k^2
    for k = 1:sx[2]
      B[i] += X[i,k]*Y[k]/sigY[k]^2
    end
  end
# Covariance:
  covC = inv(A)
# Now, solve for the coefficients, A*C = B:
  C = \(A,B)
else
  println("Error")
end
return C,covC
end
