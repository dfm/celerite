.. _install:

Installation
============

.. note:: Since celerite is actively under development right now, the only way
    to install it is following the `from source instructions <#source>`_ below.

.. _source:

From source
-----------

First you'll need to make sure that you first have `Eigen
<http://eigen.tuxfamily.org/>`_ installed. For example, on Debian-based Linux
distributions:

.. code-block:: bash

    sudo apt-get install libeigen3-dev

or on a Mac:

.. code-block:: bash

    brew install eigen

You'll also need `NumPy <http://www.numpy.org/>`_ and `pybind11
<https://pybind11.readthedocs.io>`_ and I recommend the `Anaconda distribution
<http://continuum.io/downloads>`_ if you don't already have your own opinions.

Testing
-------

To run the unit tests, install `pytest <http://doc.pytest.org/>`_ and then
execute:

.. code-block:: bash

    py.test -v

in the ``/python`` directory. All of the tests should (of course) pass.
If any of the tests don't pass and if you can't sort out why, `open an issue
on GitHub <https://github.com/dfm/celerite/issues>`_.
