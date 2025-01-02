#!/bin/bash

for zip_file in *.tar.gz
do
    echo "Extracting ${zip_file}..."
    tar -zxvf "${zip_file}"
    rm "${zip_file}"
done

echo "All files are unziped"
