.. _cpp-install:

Installation
============

You can get the celerite source code by running

.. code-block:: bash

    git clone https://github.com/dfm/celerite.git

celerite is a header-only library and the header files can be found in the
``cpp/include`` directory in the source tree.


Dependencies
------------

The only hard dependency is a recent version of `Eigen
<http://eigen.tuxfamily.org/>`_. celerite has been tested with Eigen 3.2.9 but
somewhat older versions should also work.


Testing
-------

The unit tests can be run using `CMake <https://cmake.org/>`_. Navigate to the
``cpp`` directory and execute:

.. code-block:: bash

    cmake .
    make
    make test


LAPACK Support
--------------

If you compile with the ``-DWITH_LAPACK`` macro and link to an appropriate
BLAS/LAPACK implementation, celerite will be able to use LAPACK to solve the
extended band system. This can yield performance gains for "wide" problems. To
compile the examples with this support, run:

.. code-block:: bash

    cmake . -DLAPACK=ON
