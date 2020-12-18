#!/usr/bin/env bash

set -e

# Switch working directory to
# where brenchmark.sh is.
cd $(dirname $(realpath 0))

# Make the things
for rng in CURAND AES CHAM; do
    if [ ! -f ./bin/smt-${rng} ]; then
        echo "Making bin/smt-${rng}"
        make clean rng-${rng} 1> /dev/null
    fi
done

# Backup previous results
if [ -f results.csv ]; then
    mv results.csv results-$(date +%s).csv
fi

# Warm up your GPU before hand
./bin/smt-CHAM &> /dev/null
rm results.csv

# Run benchmarks
iters=10
for i in $(seq ${iters}); do
    echo "Running smt CURAND [${i}/${iters}]"
    ./bin/smt-CURAND &>/dev/null

    echo "Running smt AES [${i}/${iters}]"
    ./bin/smt-AES &> /dev/null

    echo "Running smt CHAM [${i}/${iters}]"
    ./bin/smt-CHAM &> /dev/null
done