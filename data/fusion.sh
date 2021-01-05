#!/bin/bash

export PYTHONPATH=.

echo 'Test data extraction...'
python test_data_extraction.py -d rico
python test_data_extraction.py -d seq2seq

echo 'Data fusion...'
TYPES=( "add" "cat" )
SCALERS=( "orig" "minmax" "maxabs" "robust" "QT-norm" "QT-uni" "standard" "PT-yj" "normalizer" )
WEIGHTS=( "0.8" "0.8" "0.9" "0.6" "0.4" "0.6" "0.6" "0.6" "0.8" )

for TYPE in ${TYPES[@]}; do
    python data_fusion.py -t ${TYPE}
done

for TYPE in ${TYPES[@]}; do
    for ((i=0; i<9; i++)); do
        python data_fusion.py -t ${TYPE} -s ${SCALERS[i]} -w ${WEIGHTS[i]}
    done
done

echo 'Doen...'
