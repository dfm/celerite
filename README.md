[![Build Status](http://img.shields.io/travis/dfm/GenRP/master.svg?style=flat)](https://travis-ci.org/dfm/GenRP)
[![MIT License](http://img.shields.io/badge/license-MIT-blue.svg?style=flat)](https://github.com/dfm/GenRP/blob/master/LICENSE
)
[![Latest PDF](https://img.shields.io/badge/PDF-latest-orange.svg)](https://github.com/dfm/GenRP/blob/master-pdf/paper/ms.pdf)

Implementations of the Generalized Rybicki Press method in C++, Python, and Julia.
This method is an `O(N)` algorithm for solving matrices of the form

```
K_ij = sigma_i^2 delta_ij + sum_p a_p exp(-b_p |t_i - t_j|)
```

and computing their determinants.

See documentation in the `cpp`, `python`, and `julia` subdirectories for usage instructions.

Attribution
-----------

The method was developed by [Sivaram Ambikasaran](https://github.com/sivaramambikasaran>) and you must cite [his paper](http://arxiv.org/abs/1409.7852) if you use this code in your work.

Authors & License
-----------------

Copyright 2016 Dan Foreman-Mackey, Eric Agol, and contributors.

GenRP is free software made available under the MIT License. For details see the LICENSE file.
