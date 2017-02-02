#!/bin/bash -x

# Download the requested version of Eigen
mkdir -p eigen
cd eigen
wget --quiet "http://bitbucket.org/eigen/eigen/get/${EIGEN_VERSION}.tar.gz"
tar -xf ${EIGEN_VERSION}.tar.gz --strip-components 1
cd ..

# If building the paper, do that here
if [[ $TEST_LANG == paper ]]
then
  if git diff --name-only $TRAVIS_COMMIT_RANGE | grep 'paper/'
  then
    echo "Building the paper..."
    export CELERITE_BUILDING_PAPER=true
    source "$( dirname "${BASH_SOURCE[0]}" )"/setup-texlive.sh
    return
  fi
  export CELERITE_BUILDING_PAPER=false
  return
fi

# If testing C++, deal with that here
if [[ $TEST_LANG == cpp ]]
then
  cd cpp
  cmake . -DEIGEN_CHECK_INCLUDE_DIRS=../eigen
  make
  cd ..
  return
fi

# Conda Python
hash -r
conda config --set always_yes yes --set changeps1 no
conda update -q conda
conda info -a
conda create --yes -n test python=$PYTHON_VERSION numpy=$NUMPY_VERSION Cython setuptools pytest pip
source activate test
pip install pybind11

# Build the extension
CXX=g++-4.8 CC=gcc-4.8 python setup.py build_ext -Ieigen install
