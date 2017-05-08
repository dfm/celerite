.. :changelog:

0.2.0 (upcoming)
++++++++++++++++

- Switched to faster Cholesky factorization.
- Added O(N) simulation method using Cholesky factor.
- Implemented gradients using automatic differentiation.
- Changed implementation of white noise to ``JitterTerm`` instead of
  ``log_white_noise`` parameter.

0.1.3 (2017-03-27)
++++++++++++++++++

- Prepared manuscript and docs for submission.
- Added ``SingleSolver`` to implement the Rybicki & Press method.

0.1.2 (2017-02-23)
++++++++++++++++++

- Fixed bug when pickling ``Model`` objects.
- Added sparse solver using ``Eigen/SparseLU``.

0.1.1 (2017-02-12)
++++++++++++++++++

- Windows build support.
- Faster solver for wide problems by linking to LAPACK.

0.1.0 (2017-02-10)
++++++++++++++++++

- Initial stable release.
