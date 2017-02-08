#!/bin/bash -x

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
conda create --yes -n test python=$PYTHON_VERSION
source activate test
conda install -c conda-forge  numpy=$NUMPY_VERSION setuptools eigen pybind11 pytest

# Build the extension
#CXX=g++-4.8 CC=gcc-4.8 python setup.py install
python setup.py install
