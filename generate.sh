#!/usr/bin/env zsh

# Usage:
#  ./convert.sh file1.smt2
#
#    -> theory.cu

set -e
cd $(realpath $(dirname $0))
mkdir -p bin
mkdir -p cxx

if [ "$#" = "0" ]; then
    echo 'Supply a theory file' 1>&2
    exit 1
fi


beginswith() { case $2 in "$1"*) true;; *) false;; esac; }
format() { clang-format - }


transpose() {
    while read line; do
        if echo $line | grep "size !=" > /dev/null ; then
            varsize=$(echo $line | awk '{ print $4 }' | tr -d ')')
        fi
        echo $line
        if echo $line | grep "End program" > /dev/null ; then
            echo "int varsize = $varsize;"
        fi
    done | \
    sed 's/abort();/return 1;/' | \
        sed 's/extern "C" int/__device__ int/' | \
        sed '/size != /,+4d' | \
        sed 's/SMTLIB\/BitVector.h/SMTLIB\/BufferRef.h/' | \
        sed 's/#include/\/\/ #include/' | \
        sed 's/\/\/ Begin program/#include "theory.h"/'
}


smt2cxx() {
    docker run --rm -v $(realpath $(dirname ${1})):/out -u $(id -u):$(id -g) \
           delcypher/jfs_build:fse_2019 /home/user/jfs/build/bin/jfs-smt2cxx \
           /out/$(basename ${1})
}



for filename in "$@"; do
    rm -f theory.cu
    if beginswith http "${filename}"; then
        wget "${filename}" -O theory.smt2
        filename=theory.smt2
    fi
    smt2cxx "${filename}" | transpose | format > theory.cu
    cp theory.cu "cxx/smt-$(basename ${filename} | tr '.' '-')".cxx
    make smt
    mv smt "bin/smt-$(basename ${filename} | tr '.' '-')"
done
