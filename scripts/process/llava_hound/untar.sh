#!/bin/bash

start=0
end=15


for i in $(seq -f "%g" $start $end)
do
    file_name="chunk_${i}.tar.gz"
    tar -zxvf ${file_name}

    rm ${file_name}

done

