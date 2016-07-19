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

lim_{b->\Infty} (1-b)*[Exp[-x]-b/(1-b)*Exp[-x*(b-1)/b]] = (1+x)*Exp(-x)

which is the Matern 3/2 kernel.  The cool thing about this is that
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
