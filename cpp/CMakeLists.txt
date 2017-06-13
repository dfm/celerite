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

enable_testing()

# Includes.
include_directories(include)
include_directories(lib/eigen_3.3.3)

# Some tweaks for speed.
#add_definitions(-DNDEBUG)

# Build the examples.
add_subdirectory(src)
