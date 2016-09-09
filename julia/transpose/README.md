8/17/2016
It turns out for the real version of the code, the indexing of the array *does*
make a difference in speed.  In this directory I've reversed the indices (hence 'transpose'),
and the speed between real2 & complex is neck-and-neck again.

This also causes a speed up of the complex version of the code.

8/25/2016

Okay, I've implemented the new real solver, and it seems to work well!

The speed-up is a factor of ~2, and it is now completely real, so hopefully
we shouldn't have a problem using autodiff.  

test_matrix.jl:  Runs a test of the generalized R-P solver against a standard
linear-algebra inversion of the covariance kernel.

time_julia_final.jl:  Tests scaling of code with size of dataset.
