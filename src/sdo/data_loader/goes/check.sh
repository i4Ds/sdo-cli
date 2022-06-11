#!/bin/bash

# checks whether the hdf5 files in the sunpy data directy are valid
# it can happen that the downloaded hdf5 files are invalid. In this case run the ./check.sh command and remove the corrupt files.
# requires hdf5 utils to be installed. On Mac run `brew install hdf5`
# usage: ./check.sh

for file in ~/sunpy/data/*; do
    h5stat "$file" 2>&1 >/dev/null
done
