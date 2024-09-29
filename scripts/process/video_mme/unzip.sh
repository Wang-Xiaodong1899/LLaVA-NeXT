#!/bin/bash

for zip_file in *.zip
do
    echo "Extracting ${zip_file}..."
    unzip -o "${zip_file}"
done

echo "All files are unziped"
