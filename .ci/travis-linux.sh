#!/bin/bash -x

set -ex

# Install conda
if [[ $TEST_LANG == python ]]
then
  # http://conda.pydata.org/docs/travis.html#the-travis-yml-file
  wget http://repo.continuum.io/miniconda/Miniconda-latest-Linux-x86_64.sh -O miniconda.sh
  bash miniconda.sh -b -p $HOME/miniconda
  export PATH="$HOME/miniconda/bin:$PATH"
fi

# Install Python dependencies
source "$( dirname "${BASH_SOURCE[0]}" )"/travis.sh
