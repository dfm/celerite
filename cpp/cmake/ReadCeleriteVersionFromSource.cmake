# Adapted from ceres-solver. Below is the original license:
#
# Ceres Solver - A fast non-linear least squares minimizer
# Copyright 2015 Google Inc. All rights reserved.
# http://ceres-solver.org/
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# * Redistributions of source code must retain the above copyright notice,
#   this list of conditions and the following disclaimer.
# * Redistributions in binary form must reproduce the above copyright notice,
#   this list of conditions and the following disclaimer in the documentation
#   and/or other materials provided with the distribution.
# * Neither the name of Google Inc. nor the names of its contributors may be
#   used to endorse or promote products derived from this software without
#   specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
# ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE
# LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
# CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
# SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
# INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
# CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
# ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.
#
# Original author for ceres: alexs.mac@gmail.com (Alex Stewart)
#

macro(read_celerite_version_from_source CELERITE_SOURCE_ROOT)
    set(CELERITE_VERSION_FILE ${CELERITE_SOURCE_ROOT}/include/celerite/version.h)
    if (NOT EXISTS ${CELERITE_VERSION_FILE})
    message(FATAL_ERROR "Cannot find celerite version.h file in specified "
        "Celerite source directory: ${CELERITE_SOURCE_ROOT}, it is not here: "
        "${CELERITE_VERSION_FILE}")
  endif()

  file(READ ${CELERITE_VERSION_FILE} CELERITE_VERSION_FILE_CONTENTS)

  string(REGEX MATCH "#define CELERITE_VERSION_MAJOR [0-9]+"
      CELERITE_VERSION_MAJOR "${CELERITE_VERSION_FILE_CONTENTS}")
  string(REGEX REPLACE "#define CELERITE_VERSION_MAJOR ([0-9]+)" "\\1"
      CELERITE_VERSION_MAJOR "${CELERITE_VERSION_MAJOR}")
  # NOTE: if (VAR) is FALSE if VAR is numeric and <= 0, as such we cannot use
  #       it for testing version numbers, which might well be zero, at least
  #       for the patch version, hence check for empty string explicitly.
  if ("${CELERITE_VERSION_MAJOR}" STREQUAL "")
      message(FATAL_ERROR "Failed to extract Celerite major version from "
        "${CELERITE_VERSION_FILE}")
  endif()

  string(REGEX MATCH "#define CELERITE_VERSION_MINOR [0-9]+"
      CELERITE_VERSION_MINOR "${CELERITE_VERSION_FILE_CONTENTS}")
  string(REGEX REPLACE "#define CELERITE_VERSION_MINOR ([0-9]+)" "\\1"
      CELERITE_VERSION_MINOR "${CELERITE_VERSION_MINOR}")
  if ("${CELERITE_VERSION_MINOR}" STREQUAL "")
      message(FATAL_ERROR "Failed to extract Celerite minor version from "
        "${CELERITE_VERSION_FILE}")
  endif()

  string(REGEX MATCH "#define CELERITE_VERSION_REVISION [0-9]+"
      CELERITE_VERSION_PATCH "${CELERITE_VERSION_FILE_CONTENTS}")
  string(REGEX REPLACE "#define CELERITE_VERSION_REVISION ([0-9]+)" "\\1"
      CELERITE_VERSION_PATCH "${CELERITE_VERSION_PATCH}")
  if ("${CELERITE_VERSION_PATCH}" STREQUAL "")
      message(FATAL_ERROR "Failed to extract Celerite patch version from "
        "${CELERITE_VERSION_FILE}")
  endif()

  # This is on a single line s/t CMake does not interpret it as a list of
  # elements and insert ';' separators which would result in 3.;2.;0 nonsense.
  set(CELERITE_VERSION "${CELERITE_VERSION_MAJOR}.${CELERITE_VERSION_MINOR}.${CELERITE_VERSION_PATCH}")

  message(STATUS "Detected Celerite version: ${CELERITE_VERSION} from "
      "${CELERITE_VERSION_FILE}")
endmacro()
