#!/bin/bash

mkdir log
mkdir log/check_point
mkdir log/plot
mkdir log/pth

export PYTHONPATH=.
echo 'Start seq2seq autuencoder training...'
#python main/train.py

echo 'vector save...'
python main/vector_save.py

echo 'done...'
