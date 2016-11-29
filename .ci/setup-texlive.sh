#!/bin/bash -x

# Download and extract texlive installer
wget http://mirror.ctan.org/systems/texlive/tlnet/install-tl-unx.tar.gz -O install-tl.tar.gz
mkdir -p install-tl
tar -xf install-tl.tar.gz -C install-tl --strip-components 1

# Install only the necessary packages
mkdir -p texlive
TEXLIVE_INSTALL_PREFIX=`pwd`/texlive install-tl/install-tl -profile .ci/texlive.profile
export PATH=`pwd`/texlive/bin/x86_64-linux:$PATH
tlmgr install lineno revtex textcase epsf xcolor ulem microtype multirow helvetic
