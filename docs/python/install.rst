.. _python-install:

Installation
============

The core implementation of celerite is written in C++ so this will need to be
compiled to be called from Python. The easiest way for a new user to do this
will be by following the directions in the :ref:`using-conda` section below.
Power users might be able to eke out a bit more performance by tuning the
linear algebra library and installing :ref:`source` and following the
discussion in :ref:`lapack`.

.. _using-conda:

Using conda
-----------

The easiest way to install celerite is using `conda
<http://continuum.io/downloads>`_ (via `conda-forge
<https://conda-forge.github.io/>`_) with the following command:

.. code-block:: bash

    conda install -c conda-forge celerite

This version of celerite will be linked to the OpenBLAS implementation
available on conda-forge. It's possible that power users might be able to get
some extra performance by linking to an implementation that is more tuned for
your system (e.g. MKL) by following the instructions in :ref:`lapack` below.

.. note:: On Windows, celerite is not linked to a LAPACK implementation
    because OpenBLAS is not available for Windows on conda-forge so users with
    wide models will need to install from source.


Using pip
---------

celerite can also be installed using `pip <https://pip.pypa.io>`_ after
installing `Eigen <http://eigen.tuxfamily.org/>`_:

.. code-block:: bash

    pip install celerite

If the Eigen headers can't be found, you can hint the include directory as
follows:

.. code-block:: bash

    pip install celerite \
        --global-option=build_ext \
        --global-option=-I/path/to/eigen3


.. _source:

From Source
-----------

The source code for celerite can be downloaded `from GitHub
<https://github.com/dfm/celerite>`_ by running

.. code-block:: bash

    git clone https://github.com/dfm/celerite.git

.. _python-deps:

Dependencies
++++++++++++

For the Python interface, you'll (obviously) need a Python installation and I
recommend `conda <http://continuum.io/downloads>`_ if you don't already have
your own opinions.

After installing Python, the following dependencies are required to build
celerite:

1. `Eigen <http://eigen.tuxfamily.org/>`_ is required for matrix computations,
2. `NumPy <http://www.numpy.org/>`_ for math and linear algebra in Python, and
3. `pybind11 <https://pybind11.readthedocs.io>`_ for the Python–C++ interface.

If you're using conda, you can install all of the dependencies with the
following command:

.. code-block:: bash

    conda install -c conda-forge eigen numpy pybind11

Building
++++++++

After installing the dependencies, you can build the celerite module by
running:

.. code-block:: bash

    python setup.py install

in the root directory of the source tree.
If the Eigen headers can't be found, you can hint the include directory as
follows:

.. code-block:: bash

    python setup.py build_ext -I/path/to/eigen3 install


.. _lapack:

A word about LAPACK support
---------------------------

You can get a substantial speed up of the algorithm for models with a large
number of terms if you link to a LAPACK library tuned for your system.
The conda package described above will link to the linear algebra used by
NumPy on macOS and Linux but, if you're installing from source, you'll need to
request LAPACK support explicitly using:

.. code-block:: bash

    LAPACK_VARIANT='blas_opt' python setup.py install

This will again link to the LAPACK implementation used by your NumPy
installation.
The ``LAPACK_VARIANT`` can be any of the options provided by the
:func:`numpy.distutils.system_info.get_info` function.
The ``blas_opt`` variant used above is NumPy's suggested choice but if you
have an Intel MKL library installed then you can often gain some performance
by using ``LAPACK_VARIANT=mkl``.

If you want to link to a custom implementation, you can set the
``WITH_LAPACK`` macro and provide the compiler and linker flags yourself.
For example, to link to Apple's Accelerate framework on macOS, you could use
the following:

.. code-block:: bash

    CFLAGS="-DWITH_LAPACK -msse3" LDFLAGS="-Wl,-framework -Wl,Accelerate" python setup.py install


Testing
-------

To run the unit tests, install `pytest <http://doc.pytest.org/>`_ and then
execute:

.. code-block:: bash

    py.test -v

All of the tests should (of course) pass.
If any of the tests don't pass and if you can't sort out why, `open an issue
on GitHub <https://github.com/dfm/celerite/issues>`_.
