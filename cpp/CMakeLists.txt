# Much of this is inspired by the CMake scripts from the BSD licensed
# ceres-solver: https://github.com/ceres-solver/ceres-solver

cmake_minimum_required(VERSION 2.8.0)
project(celerite C CXX)

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

include(ReadCeleriteVersionFromSource)
read_celerite_version_from_source(${CMAKE_SOURCE_DIR})

find_package(Eigen REQUIRED)
if (EIGEN_VERSION VERSION_LESS 3.1.0)
message(FATAL_ERROR "-- celerite requires Eigen version >= 3.1.0 in order "
    "that Eigen/SparseCore be available, detected version of Eigen is: "
    "${EIGEN_VERSION}")
endif (EIGEN_VERSION VERSION_LESS 3.1.0)

# LAPACK (& BLAS).
set(CELERITE_LIBRARIES)
option(LAPACK "Enable use of LAPACK." OFF)
if (LAPACK)
  find_package(LAPACK QUIET)
  if (LAPACK_FOUND)
    message("-- Found LAPACK library: ${LAPACK_LIBRARIES}")
  else (LAPACK_FOUND)
    message("-- Did not find LAPACK library, disabling LAPACK support.")
  endif (LAPACK_FOUND)

  find_package(BLAS QUIET)
  if (BLAS_FOUND)
    message("-- Found BLAS library: ${BLAS_LIBRARIES}")
  else (BLAS_FOUND)
    message("-- Did not find BLAS library, disabling LAPACK support.")
  endif (BLAS_FOUND)

  if (LAPACK_FOUND AND BLAS_FOUND)
    add_definitions(-DWITH_LAPACK)
    list(APPEND CELERITE_LIBRARIES ${LAPACK_LIBRARIES} ${BLAS_LIBRARIES})
  endif (LAPACK_FOUND AND BLAS_FOUND)
else (LAPACK)
  message("-- Building without LAPACK.")
endif (LAPACK)

enable_testing()

# Includes.
include_directories(include)
include_directories(SYSTEM ${EIGEN_INCLUDE_DIRS})

# Some tweaks for speed.
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O4")
add_definitions(-DNDEBUG)

# Build the examples.
add_subdirectory(src)

# Setup installation of public headers.
#file(GLOB CELERITE_HDRS ${CMAKE_SOURCE_DIR}/include/celerite/*.h)
#install(FILES ${CELERITE_HDRS} DESTINATION include/celerite)
#file(GLOB CELERITE_SOLVER_HDRS ${CMAKE_SOURCE_DIR}/include/celerite/solver/*.h)
#install(FILES ${CELERITE_SOLVER_HDRS} DESTINATION include/celerite/solver)
