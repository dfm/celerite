# -*- coding: utf-8 -*-

from __future__ import division, print_function

import os
import sys
import tempfile

import setuptools
from setuptools.command.build_ext import build_ext as _build_ext

__all__ = ["build_ext"]

def has_flag(compiler, flagname):
    """Return a boolean indicating whether a flag name is supported on
    the specified compiler.
    """
    with tempfile.NamedTemporaryFile("w", suffix=".cpp") as f:
        f.write("int main (int argc, char **argv) { return 0; }")
        try:
            compiler.compile([f.name], extra_postargs=[flagname])
        except setuptools.distutils.errors.CompileError:
            return False
    return True

def has_library(compiler, libname):
    """Return a boolean indicating whether a library is found."""
    with tempfile.NamedTemporaryFile("w", suffix=".cpp") as srcfile:
        srcfile.write("int main (int argc, char **argv) { return 0; }")
        srcfile.flush()
        outfn = srcfile.name + ".so"
        try:
            compiler.link_executable(
                [srcfile.name],
                outfn,
                libraries=[libname],
            )
        except setuptools.distutils.errors.LinkError:
            return False
        if not os.path.exists(outfn):
            return False
        os.remove(outfn)
    return True

def cpp_flag(compiler):
    """Return the -std=c++[11/14] compiler flag.

    The c++14 is prefered over c++11 (when it is available).
    """
    if has_flag(compiler, "-std=c++14"):
        return "-std=c++14"
    elif has_flag(compiler, "-std=c++11"):
        return "-std=c++11"
    else:
        raise RuntimeError('Unsupported compiler -- at least C++11 support '
                           'is needed!')

class build_ext(_build_ext):
    """
    A custom extension builder that finds the include directories for Eigen
    before compiling.

    """

    c_opts = {
        "msvc": ["/EHsc", "/DNODEBUG"],
        "unix": ["-DNODEBUG"],
    }

    def build_extensions(self):
        # The include directory for the celerite headers
        localincl = os.path.join("cpp", "include")
        if not os.path.exists(os.path.join(localincl, "celerite",
                                           "version.h")):
            raise RuntimeError("couldn't find celerite headers")

        # Add the pybind11 include directory
        import numpy
        import pybind11
        include_dirs = [
            localincl,
            os.path.join("cpp", "lib", "eigen_3.3.3"),
            numpy.get_include(),
            pybind11.get_include(False),
            pybind11.get_include(True),
        ]
        for ext in self.extensions:
            ext.include_dirs += include_dirs

        # Building on RTDs takes a bit of special care
        if os.environ.get("READTHEDOCS", None) == "True":
            for ext in self.extensions:
                ext.extra_compile_args = ["-std=c++14", "-O0", "-DNO_AUTODIFF"]
            _build_ext.build_extensions(self)
            return

        # Compiler flags
        ct = self.compiler.compiler_type
        opts = self.c_opts.get(ct, [])
        if ct == "unix":
            opts.append("-DVERSION_INFO=\"{0:s}\""
                        .format(self.distribution.get_version()))

            print("testing C++14/C++11 support")
            opts.append(cpp_flag(self.compiler))

            flags = ["-stdlib=libc++", "-fvisibility=hidden",
                     "-Wno-unused-function", "-Wno-uninitialized",
                     "-Wno-unused-local-typedefs", "-funroll-loops"]

            # Mac specific flags and libraries
            if sys.platform == "darwin":
                flags += ["-march=native", "-mmacosx-version-min=10.9"]
                for lib in ["m", "c++"]:
                    for ext in self.extensions:
                        ext.libraries.append(lib)
                for ext in self.extensions:
                    ext.extra_link_args += ["-mmacosx-version-min=10.9",
                                            "-march=native"]
            else:
                libraries = ["m", "stdc++", "c++"]
                for lib in libraries:
                    if not has_library(self.compiler, lib):
                        continue
                    for ext in self.extensions:
                        ext.libraries.append(lib)

            # Check the flags
            print("testing compiler flags")
            for flag in flags:
                if has_flag(self.compiler, flag):
                    opts.append(flag)

        elif ct == "msvc":
            opts.append("/DVERSION_INFO=\\\"{0:s}\\\""
                        .format(self.distribution.get_version()))

        for ext in self.extensions:
            ext.extra_compile_args = opts

        # Run the standard build procedure.
        _build_ext.build_extensions(self)
