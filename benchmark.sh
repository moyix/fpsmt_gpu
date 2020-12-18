#!/usr/bin/env bash

set -e

# Switch working directory to
# where brenchmark.sh is.
cd $(dirname $(realpath 0))

# Make the things
make all &> /dev/null

# Backup previous results
if [ -f results.csv ]; then
    mv results.csv results-$(date +%s).csv
fi

# Warm up your GPU before hand
echo "Warming up GPU"
./bin/smt-CHAM &> /dev/null
rm results.csv

# Run benchmarks
iters=750
for i in $(seq ${iters}); do
    echo "Running smt CURAND [${i}/${iters}]"
    ./bin/smt-CURAND &>/dev/null

    echo "Running smt AES [${i}/${iters}]"
    ./bin/smt-AES &> /dev/null

    echo "Running smt CHAM [${i}/${iters}]"
    ./bin/smt-CHAM &> /dev/null
done
