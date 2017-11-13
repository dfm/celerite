.. :changelog:

0.3.0 (2017-11-13)
++++++++++++++++++

- Added support for fully general semiseparable kernels.
- Added ``quiet`` argument to likelihood functions.

0.2.1 (2017-06-12)
++++++++++++++++++

- Small bug fixes.
- New ``celerite.solver.LinAlgError`` exception thrown for non-positive
  definite matrices instead of ``RuntimeError``.

0.2.0 (2017-05-09)
++++++++++++++++++

- Switched to (~20x) faster Cholesky factorization.
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
