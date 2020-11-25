#!/usr/bin/env zsh

# Usage:
#  ./convert.sh file1.smt2
#
#    -> theory.cu

set -e
cd $(realpath $(dirname))
mkdir -p bin

if [ "$#" = "0" ]; then
    echo 'Supply a theory file' 1>&2
    exit 1
fi


beginswith() { case $2 in "$1"*) true;; *) false;; esac; }
format() { clang-format - }


transpose() {
    sed 's/abort();/return 1;/' | \
        sed 's/extern "C" int/__device__ int/' | \
        sed 's/size != 32/size < 16/' | \
        sed 's/jfs_warning("Wrong sized input tried.\\n");//' | \
        sed 's/SMTLIB\/BitVector.h/SMTLIB\/BufferRef.h/' | \
        sed 's/#include/\/\/ #include/' | \
        sed 's/\/\/ Begin program/#include "theory.h"/'
}


smt2cxx() {
    docker run -it --rm -v $(pwd):/out -u $(id -u):$(id -g) \
           delcypher/jfs_build:fse_2019 /home/user/jfs/build/bin/jfs-smt2cxx \
           /out/${1}
}



for filename in "$@"; do
    if beginswith http "${filename}"; then
        wget "${filename}" -O theory.smt2
        filename=theory.smt2
    fi
    smt2cxx "${filename}" | transpose | format > theory.cu
    make clean smt
    mv smt "bin/smt-$(echo ${filename} | tr '.' '-')"
done
