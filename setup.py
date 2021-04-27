#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys

from setuptools import setup, Extension
from pybind11.setup_helpers import Pybind11Extension, build_ext

# Publish the library to PyPI.
if "publish" in sys.argv[-1]:
    os.system("python setup.py sdist upload")
    sys.exit()

# Default compile arguments.
ext = Pybind11Extension(
    "celerite.solver",
    sources=["celerite/solver.cpp"],
    language="c++",
    include_dirs=["cpp/include", "cpp/lib/eigen"],
)

setup(
    name="celerite",
    use_scm_version={
        "write_to": os.path.join("celerite/celerite_version.py"),
        "write_to_template": '__version__ = "{version}"\n',
    },
    author="Daniel Foreman-Mackey",
    author_email="foreman.mackey@gmail.com",
    url="https://github.com/dfm/celerite",
    license="MIT",
    packages=["celerite"],
    install_requires=["numpy"],
    extras_require={
        "test": [
            "autograd",
            "coverage[toml]",
            "pytest",
            "pytest-cov",
        ]
    },
    ext_modules=[ext],
    description="Scalable 1D Gaussian Processes",
    long_description=open("README.rst").read(),
    package_data={"": ["README.rst", "LICENSE", "CITATION"]},
    include_package_data=True,
    python_requires=">=3.6",
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
