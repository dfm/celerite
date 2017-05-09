.. _start:

Getting Started
===============

The following code snippet shows a simple example of how to use the C++
interface to celerite.
In this example, the kernel has the form:

.. math::

    k(\tau) = a_1\,e^{-c_1\,\tau} + a_2\,e^{-c_2\,\tau} +
        \frac{1}{2}\left[
            (a_3 + i\,b_3)\,e^{-(c_3+i\,d_3)\,\tau} +
            (a_3 - i\,b_3)\,e^{-(c_3-i\,d_3)\,\tau}
        \right]

For more details of the kernel structure supported by celerite, check out
:ref:`kernel`.

.. code-block:: cpp

    #include <cmath>
    #include <iostream>
    #include <Eigen/Core>
    #include "celerite/celerite.h"

    using Eigen::VectorXd;

    int main () {
        // Choose some demo parameters for the solver
        int j_real = 2, j_complex = 1;
        double jitter = 0.0;
        VectorXd a_real(j_real),
                c_real(j_real),
                a_comp(j_complex),
                b_comp(j_complex),
                c_comp(j_complex),
                d_comp(j_complex);
        a_real << 1.0, 0.3;
        c_real << 0.5, 3.5;
        a_comp << 1.0;
        b_comp << 0.1;
        c_comp << 3.0;
        d_comp << 1.0;

        // Generate some fake data
        int N = 500;
        srand(42);
        VectorXd x = VectorXd::Random(N),
                yvar = VectorXd::Random(N),
                y;
        yvar.array() *= 0.1;
        yvar.array() += 1.0;
        std::sort(x.data(), x.data() + x.size()); // The independent coordinates must be sorted
        y = sin(x.array());

        // Set up the solver
        celerite::solver::CholeskySolver<double> solver;
        solver.compute(
            jitter,
            a_real, c_real,
            a_comp, b_comp, c_comp, d_comp,
            x, yvar  // Note: this is the measurement _variance_
        );

        std::cout << solver.log_determinant() << std::endl;
        std::cout << solver.dot_solve(y) << std::endl;

        return 0;
    }

Compiling:

.. code-block:: bash

    g++ -Icpp/include -Icpp/lib/eigen_3.3.3 -o celerite_demo celerite_demo.cc

and executing this example should output::

    86.405
    0.82574
