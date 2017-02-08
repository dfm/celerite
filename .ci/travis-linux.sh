#!/bin/bash -x

# Install conda
if [[ $TEST_LANG == python ]]
then
  # http://conda.pydata.org/docs/travis.html#the-travis-yml-file
  wget http://repo.continuum.io/miniconda/Miniconda-latest-Linux-x86_64.sh -O miniconda.sh
  bash miniconda.sh -b -p $HOME/miniconda
  export PATH="$HOME/miniconda/bin:$PATH"
fi

if [[ $TEST_LANG == cpp ]]
then
  # Download the requested version of Eigen
  mkdir -p eigen
  cd eigen
  wget --quiet "http://bitbucket.org/eigen/eigen/get/${EIGEN_VERSION}.tar.gz"
  tar -xf ${EIGEN_VERSION}.tar.gz --strip-components 1
  cd ..
fi

# Install Python dependencies
source "$( dirname "${BASH_SOURCE[0]}" )"/travis.sh
