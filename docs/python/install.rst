.. _python-install:

Installation
============

.. note:: Since celerite is actively under development right now, the best way
    to install it is following :ref:`source` below.

Using conda
-----------

.. note:: This method doesn't work yet but when it does, it'll be the best!

The recommended method of installing celerite is using `conda
<http://continuum.io/downloads>`_ with the following command:

.. code-block:: bash

    conda install -c conda-forge celerite


Using pip
---------

celerite can also be install using `pip <https://pip.pypa.io>`_ after
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


Testing
-------

To run the unit tests, install `pytest <http://doc.pytest.org/>`_ and then
execute:

.. code-block:: bash

    py.test -v

All of the tests should (of course) pass.
If any of the tests don't pass and if you can't sort out why, `open an issue
on GitHub <https://github.com/dfm/celerite/issues>`_.