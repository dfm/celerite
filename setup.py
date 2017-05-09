#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys

from setuptools import setup, Extension

# Publish the library to PyPI.
if "publish" in sys.argv[-1]:
    os.system("python setup.py sdist upload")
    sys.exit()

# Default compile arguments.
ext = Extension("celerite.solver",
                sources=[os.path.join("celerite", "solver.cpp")],
                language="c++")

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
    package_data={"": ["README.rst", "LICENSE", "CITATION"]},
    include_package_data=True,
    cmdclass=dict(build_ext=build_ext),
    classifiers=[
        "Development Status :: 5 - Production/Stable",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python",
    ],
    zip_safe=True,
)
