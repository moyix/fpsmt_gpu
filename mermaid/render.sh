#!/bin/sh

cd $(dirname $(realpath $0))

if [ ! -d node_modules ]; then
    npm i
fi

if [ ! -d ../svg ]; then
    mkdir ../svg
fi

for mmdf in $(find -name '*.mmd'); do
    echo "rendering ${mmdf}"
    npx mmdc -i ${mmdf} -o ../svg/${mmdf}.svg -b transparent -t forest
done
