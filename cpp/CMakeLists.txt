# Much of this is inspired by the CMake scripts from the BSD licensed
# ceres-solver: https://github.com/ceres-solver/ceres-solver

cmake_minimum_required(VERSION 2.8.0)
project(GenRP C CXX)

# Make CMake aware of the cmake folder for local FindXXX scripts,
# append rather than set in case the user has passed their own
# additional paths via -D.
list(APPEND CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/cmake")

# On OS X, add the Homebrew prefix to the set of prefixes searched by
# CMake in find_path & find_library.  This should ensure that we can
# still build even if Homebrew is installed in a non-standard
# location (not /usr/local).
if (CMAKE_SYSTEM_NAME MATCHES "Darwin")
  find_program(HOMEBREW_EXECUTABLE brew)
  mark_as_advanced(FORCE HOMEBREW_EXECUTABLE)
  if (HOMEBREW_EXECUTABLE)
    # Detected a Homebrew install, query for its install prefix.
    execute_process(COMMAND ${HOMEBREW_EXECUTABLE} --prefix
      OUTPUT_VARIABLE HOMEBREW_INSTALL_PREFIX
      OUTPUT_STRIP_TRAILING_WHITESPACE)
    message(STATUS "Detected Homebrew with install prefix: "
      "${HOMEBREW_INSTALL_PREFIX}, adding to CMake search paths.")
    list(APPEND CMAKE_PREFIX_PATH "${HOMEBREW_INSTALL_PREFIX}")
  endif()
endif()

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

include(ReadGenRPVersionFromSource)
read_genrp_version_from_source(${CMAKE_SOURCE_DIR})

find_package(Eigen REQUIRED)
if (EIGEN_VERSION VERSION_LESS 3.1.0)
message(FATAL_ERROR "-- GenRP requires Eigen version >= 3.1.0 in order "
    "that Eigen/SparseCore be available, detected version of Eigen is: "
    "${EIGEN_VERSION}")
endif (EIGEN_VERSION VERSION_LESS 3.1.0)
enable_testing()

# Includes.
include_directories(include)
include_directories(SYSTEM ${EIGEN_INCLUDE_DIRS})

# Set the default flags to an empty string.
set (GENRP_CXX_FLAGS)
if (CMAKE_COMPILER_IS_GNUCXX)
    # Linux
    if (CMAKE_SYSTEM_NAME MATCHES "Linux")
        if (NOT GCC_VERSION VERSION_LESS 4.2)
            set (GENRP_CXX_FLAGS "${GENRP_CXX_FLAGS} -march=native -mtune=native")
        endif (NOT GCC_VERSION VERSION_LESS 4.2)
    endif (CMAKE_SYSTEM_NAME MATCHES "Linux")
    # Mac OS X
    if (CMAKE_SYSTEM_NAME MATCHES "Darwin")
        set (GENRP_CXX_FLAGS "${GENRP_CXX_FLAGS} -msse3")
        # Use of -fast only applicable for Apple's GCC
        # Assume this is being used if GCC version < 4.3 on OSX
        execute_process(COMMAND ${CMAKE_C_COMPILER}
            ARGS ${CMAKE_CXX_COMPILER_ARG1} -dumpversion
            OUTPUT_VARIABLE GCC_VERSION
            OUTPUT_STRIP_TRAILING_WHITESPACE)
        if (GCC_VERSION VERSION_LESS 4.3)
            set (GENRP_CXX_FLAGS "${GENRP_CXX_FLAGS} -fast")
        endif (GCC_VERSION VERSION_LESS 4.3)
    endif (CMAKE_SYSTEM_NAME MATCHES "Darwin")
endif (CMAKE_COMPILER_IS_GNUCXX)

set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${GENRP_CXX_FLAGS}")

if (UNIX)
  # GCC is not strict enough by default, so enable most of the warnings.
  set(CMAKE_CXX_FLAGS
    "${CMAKE_CXX_FLAGS} -Wno-unknown-pragmas -Wno-sign-compare -Wno-unused-parameter -Wno-missing-field-initializers")
endif (UNIX)

# Use a larger inlining threshold for Clang, since it hobbles Eigen,
# resulting in an unreasonably slow version of the blas routines. The
# -Qunused-arguments is needed because CMake passes the inline
# threshold to the linker and clang complains about it and dies.
if (CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
  set(CMAKE_CXX_FLAGS
    "${CMAKE_CXX_FLAGS} -Qunused-arguments -mllvm -inline-threshold=600 -O3")

  # Older versions of Clang (<= 2.9) do not support the 'return-type-c-linkage'
  # option, so check for its presence before adding it to the default flags set.
  include(CheckCXXCompilerFlag)
  check_cxx_compiler_flag("-Wno-return-type-c-linkage"
                          HAVE_RETURN_TYPE_C_LINKAGE)
  if (HAVE_RETURN_TYPE_C_LINKAGE)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-return-type-c-linkage")
  endif(HAVE_RETURN_TYPE_C_LINKAGE)
endif ()

# Disable asserts for efficiency.
add_definitions(-DNDEBUG)

# Build the examples.
add_subdirectory(src)

# Setup installation of public headers.
file(GLOB GENRP_HDRS ${CMAKE_SOURCE_DIR}/include/genrp/*.h)
install(FILES ${GENRP_HDRS} DESTINATION include/genrp)
