#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys
import numpy

from setuptools import setup, Extension

# Publish the library to PyPI.
if "publish" in sys.argv[-1]:
    os.system("python setup.py sdist upload")
    sys.exit()

# The include directory for the celerite headers
localincl = os.path.join("cpp", "include")
if not os.path.exists(os.path.join(localincl, "celerite", "version.h")):
    raise RuntimeError("couldn't find celerite headers")

# Default compile arguments.
compile_args = dict(
    libraries=[],
    define_macros=[("NDEBUG", None)],
)
if os.name == "posix":
    compile_args["libraries"] += ["m", "stdc++"]

compile_args["include_dirs"] = [
    localincl,
    numpy.get_include(),
]

# Check for LAPACK argument
if "--lapack" in sys.argv:
    sys.argv.pop(sys.argv.index("--lapack"))
    compile_args["define_macros"] += [("WITH_LAPACK", None)]

ext = Extension("celerite.solver",
                sources=[os.path.join("celerite", "solver.cpp")],
                language="c++",
                **compile_args)

# Hackishly inject a constant into builtins to enable importing of the
# package before the library is built.
if sys.version_info[0] < 3:
    import __builtin__ as builtins
else:
    import builtins
builtins.__CELERITE_SETUP__ = True
import celerite  # NOQA
from celerite.build import build_ext  # NOQA

setup(
    name="celerite",
    version=celerite.__version__,
    author="Daniel Foreman-Mackey",
    author_email="foreman.mackey@gmail.com",
    url="https://github.com/dfm/celerite",
    license="MIT",
    packages=["celerite"],
    install_requires=["numpy", "pybind11"],
    ext_modules=[ext],
    description="Scalable 1D Gaussian Processes",
    long_description=open("README.rst").read(),
    package_data={"": ["README.rst", "LICENSE",
                       os.path.join(localincl, "*.h"),
                       os.path.join(localincl, "*", "*.h")]},
    include_package_data=True,
    cmdclass=dict(build_ext=build_ext),
    classifiers=[
        # "Development Status :: 5 - Production/Stable",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python",
    ],
    zip_safe=True,
)
