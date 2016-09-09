6/17/2016

Herein I implement the Extended SemiSeparable Solver in Julia:

https://github.com/sivaramambikasaran/ESS

A couple of modifications:

1). I'm going to try complex \beta values (requires
complex conjugate as well).

2). I'm going to try heterscedastic noise.

3). I'm going to make the extended matrix Hermitian.

6/18/2016

Okay, so I've implemented bandec.jl in Complex{Float64} & Float64 types,
and I've implemented banbks.jl in both as well.

I've profiled, and the complex version takes ~2-3 times longer, as it should.

So, next steps are:
1). Building banded matrix.  Should I try to generalize it for a second
(small) dimension (i.e. wavelength) from the start? [ ]
2). Computing the determinant (which is just the product of diagonals of L/U components). [x]
3). Implement a GP likelihood computation. [ ]

6/24/2016
1). How does symmetry of the matrix affect the LU decomposition (which I think
should be Cholesky)? [ ]
2). How do I generate simulated GPs from this decomposition? [x]
 -  Use transpose of Cholesky decomposition of original matrix.  The results
    look indistinguishable from the Kepler data for TYC 3559!  It would be
    nice to have an o(N) method for doing this (as well as for computing
    \Sigma^{-1/2} so we can check that inverse is iid). [ ]
3). How do I compute the determinant from the decomposition? [x]

6/27/2016
I figured out how to make matrix Hermitian: interchange rows of equation (60)
and interchange columns multiplying the l vector.  This gives the same answer.

7/6/2016

The sum of two exponential kernels may be used to approximate the Matern
kernel to arbitrary accuracy:

lim_{b->\Infty} (1-b)*[Exp[-x]+b/(1-b)*Exp[-x*(b-1)/b]] = (1+x)*Exp(-x)

which is the Matern 3/2 kernel.  The maximum error on this function relative
to the Matern kernel for large b occurs when x ~ 2(1-1/b) which yields
an error of ~2/b/e^2 ~ 0.27/b.  So, for b ~ 30, the error is only ~1%
of the maximum value of the Matern kernel.

The cool thing about this is that
the PSD of the Matern kernel is given by:

2/(1+f^2)^2

which is a fairly good approximation to the low-frequency component of the
Solar power spectrum!!!  This may be why the Matern 3/2 kernel works so well 
when modeling stars...

May be a way to fit general Lorentzians to power spectra:

http://arxiv.org/abs/0811.3345

Here is a fit to the Solar power spectrum (!):

http://adsabs.harvard.edu/full/2006ESASP.624E..94L

It is basically the sum of a bunch of Lorentzians!  And one ~Matern 3/2
kernel!

The Asteroseismology book indicates that lines have a Lorentzian
shape when viewed for an infinite amount of time:

https://books.google.com/books?id=N8pswDrdSyUC&lpg=PR3&pg=PR3#v=onepage&q&f=false

8/9/2016

I'm stuck on computing the derivative of the determinant:
1). There is no analytic expression for determinants of sums (although Minkowski determinant
  gives a tight lower limit).
2). ForwardDiff doesn't work with complex numbers.
3). ReverseDiffSource doesn't work with functions that have loops.

8/11/2016
I implemented autodiff with ForwardDiff.jl.  This only works with Real numbers,
so I had to rewrite the extended kernel computation in Real numbers.  This is
done in lorentz_likelihood_real_band_init.jl and lorentz_likelihood_real_band_save.jl.
The extended matrices are twice the size, and so take more operations to compute.
However, they use real numbers, so I expected a speedup in not having to use
complex arithmetic.  Unfortunately, the net result is that the code is ~5-10 times
slower than the Complex version (*hermitian*.jl).

Nevertheless, the ForwardDiff seems to be working, and the gradients take ~20-25
times longer to compute than the Complex version of the calculation;  which may
not be too bad since finite diff should take ~10x longer, while autodiff should
be more accurate.

To Do:

1). Implement a test for this ForwardDiff code to see that it is giving the correct
derivatives. [x]
2). See if there are any other speed gains to be had. [x]

9/6/2016

Speed gains were found in:
1). Using transpose of matrices (in subdirectory transpose/)
2). Using a real implementation.

Subsequently, the code in this directory has been deprecated, while the code in
the directory above is the latest (final?) version.
