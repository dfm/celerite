8/18/2016

I'm trying to get a real version of the equations since I think there is duplication in the
computation for the Lorentzians.  I'm going to examine a single Lorentzian case in detail
numerically to see where the simplification might occur.

8/24/2016
This works!  I can now write the equations as only real equations with a symmetric
extended matrix.  The speed up in solving is only modest, though, a factor of ~1.5-2.

8/25/2016
Okay, I've gotten a symmetric, real version working in which I eliminated the
equations for any case with Im(beta) = 0.

The bandwidth appears to be:  p0+2(p-p0+1) (above & below), for a total matrix
width of: 2p0+4(p-p0+1)+1 = 2p0+4(p-p0)+5

Let's see if this gives the right likelihood.
