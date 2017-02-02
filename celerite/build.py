# -*- coding: utf-8 -*-

from __future__ import division, print_function

import os
import re
import sys
import logging
import tempfile

import setuptools
from setuptools.command.build_ext import build_ext as _build_ext

__all__ = ["build_ext"]

def find_eigen(hint=None):
    """
    Find the location of the Eigen 3 include directory. This will return
    ``None`` on failure.
    """
    # List the standard locations including a user supplied hint.
    search_dirs = [] if hint is None else hint
    if "CONDA_PREFIX" in os.environ:
        search_dirs.append(os.path.join(
            os.environ["CONDA_PREFIX"], "include", "eigen3"))
    search_dirs += [
        "/usr/local/include/eigen3",
        "/usr/local/homebrew/include/eigen3",
        "/opt/local/var/macports/software/eigen3",
        "/opt/local/include/eigen3",
        "/usr/include/eigen3",
        "/usr/include/local",
        "/usr/include",
        "/usr/local/include",
    ]
    print(search_dirs)

    # Loop over search paths and check for the existence of the Eigen/Dense
    # header.
    for d in search_dirs:
        path = os.path.join(d, "Eigen", "Dense")
        if os.path.exists(path):
            # Determine the version.
            vf = os.path.join(d, "Eigen", "src", "Core", "util", "Macros.h")
            if not os.path.exists(vf):
                continue
            src = open(vf, "r").read()
            v1 = re.findall("#define EIGEN_WORLD_VERSION (.+)", src)
            v2 = re.findall("#define EIGEN_MAJOR_VERSION (.+)", src)
            v3 = re.findall("#define EIGEN_MINOR_VERSION (.+)", src)
            if not len(v1) or not len(v2) or not len(v3):
                continue
            v = "{0}.{1}.{2}".format(v1[0], v2[0], v3[0])
            print("Found Eigen version {0} in: {1}".format(v, d))
            return d
    return None

def has_flag(compiler, flagname):
    """Return a boolean indicating whether a flag name is supported on
    the specified compiler.
    """
    with tempfile.NamedTemporaryFile('w', suffix='.cpp') as f:
        f.write('int main (int argc, char **argv) { return 0; }')
        try:
            compiler.compile([f.name], extra_postargs=[flagname])
        except setuptools.distutils.errors.CompileError:
            return False
    return True

def cpp_flag(compiler):
    """Return the -std=c++[11/14] compiler flag.

    The c++14 is prefered over c++11 (when it is available).
    """
    if has_flag(compiler, '-std=c++14'):
        return '-std=c++14'
    elif has_flag(compiler, '-std=c++11'):
        return '-std=c++11'
    else:
        raise RuntimeError('Unsupported compiler -- at least C++11 support '
                           'is needed!')

class build_ext(_build_ext):
    """
    A custom extension builder that finds the include directories for Eigen
    before compiling.

    """

    c_opts = {
        'msvc': ['/EHsc'],
        'unix': [],
    }

    if sys.platform == 'darwin':
        c_opts['unix'] += ['-stdlib=libc++', '-mmacosx-version-min=10.7']

    def build_extensions(self):
        # Add the required Eigen include directory
        dirs = self.compiler.include_dirs
        for ext in self.extensions:
            dirs += ext.include_dirs
        include_dirs = []
        eigen_include = find_eigen(hint=dirs)
        if eigen_include is None:
            logging.warn("Required library Eigen 3 not found.")
            # raise RuntimeError("Required library Eigen 3 not found. "
            #                    "Check the documentation for solutions.")
        else:
            include_dirs += [eigen_include]

        # Add the pybind11 include directory
        import pybind11
        include_dirs += [
            pybind11.get_include(False),
            pybind11.get_include(True),
        ]

        for ext in self.extensions:
            ext.include_dirs += include_dirs

        # Set up pybind11
        ct = self.compiler.compiler_type
        opts = self.c_opts.get(ct, [])
        if ct == 'unix':
            opts.append('-DVERSION_INFO="{0:s}"'
                        .format(self.distribution.get_version()))
            opts.append(cpp_flag(self.compiler))
            if has_flag(self.compiler, '-fvisibility=hidden'):
                opts.append('-fvisibility=hidden')
        elif ct == 'msvc':
            opts.append('/DVERSION_INFO=\\"{0:s}\\"'
                        .format(self.distribution.get_version()))

        for flag in ["-Wno-unused-function", "-Wno-uninitialized", "-O4"]:
            if has_flag(self.compiler, flag):
                opts.append(flag)

        for ext in self.extensions:
            ext.extra_compile_args = opts

        # Run the standard build procedure.
        _build_ext.build_extension(self, ext)
