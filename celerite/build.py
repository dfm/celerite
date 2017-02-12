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

def find_eigen(hint=None, verbose=False):
    """
    Find the location of the Eigen 3 include directory. This will return
    ``None`` on failure.
    """
    # List the standard locations including a user supplied hint.
    search_dirs = [] if hint is None else list(hint)

    # Look in the conda include directory in case eigen was installed using
    # conda.
    if "CONDA_PREFIX" in os.environ:
        search_dirs.append(os.path.join(os.environ["CONDA_PREFIX"], "include"))
        search_dirs.append(os.path.join(os.environ["CONDA_PREFIX"], "Library",
                                        "include"))

    if "CONDA_ENV_PATH" in os.environ:
        search_dirs.append(os.path.join(os.environ["CONDA_ENV_PATH"],
                                        "include"))
        search_dirs.append(os.path.join(os.environ["CONDA_ENV_PATH"],
                                        "Library", "include"))

    # Another hack to find conda include directory if the environment variable
    # doesn't exist. This seems to be necessary for RTD.
    for d in search_dirs:
        el = os.path.split(d)
        if len(re.findall(r"python[0-9\.].+m", el[-1])):
            search_dirs += [os.path.join(*el[:-1])]

    # Some other common installation locations on UNIX-ish platforms
    search_dirs += [
        "/"
        "/usr/local/include",
        "/usr/local/homebrew/include",
        "/opt/local/var/macports/software",
        "/opt/local/include",
        "/usr/include",
        "/usr/include/local",
        "/usr/include",
    ]

    # Common suffixes
    suffixes = ["", "eigen3", "Eigen/include/eigen3", "Eigen3/include/eigen3"]

    # Debugging comments
    if verbose:
        print("Looking for Eigen in:")
        for d in search_dirs:
            print(" - {0}".format(os.path.abspath(d)))
        print("+ suffixes: {0}".format(suffixes))

    # Loop over search paths and check for the existence of the Eigen/Dense
    # header.
    for base in search_dirs:
        for suff in suffixes:
            d = os.path.abspath(os.path.join(base, suff))
            path = os.path.join(d, "Eigen", "Dense")
            if not os.path.exists(path):
                continue

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
            for flag in ["-Wno-unused-function", "-Wno-uninitialized", "-O4"]:
                if has_flag(self.compiler, flag):
                    opts.append(flag)
        elif ct == 'msvc':
            opts.append('/DVERSION_INFO=\\"{0:s}\\"'
                        .format(self.distribution.get_version()))

        for ext in self.extensions:
            ext.extra_compile_args = opts

        # Link to numpy's LAPACK if available
        info = None
        for ext in self.extensions:
            if not any(k[0] == "WITH_LAPACK" for k in ext.define_macros):
                continue
            if info is None:
                import pprint
                import numpy.__config__ as npconf
                info = npconf.get_info("blas_opt_info")
                print("Found LAPACK linking info:")
                pprint.pprint(info)
            for k, v in info.items():
                try:
                    setattr(ext, k, getattr(ext, k) + v)
                except TypeError:
                    continue

        # Run the standard build procedure.
        _build_ext.build_extension(self, ext)
