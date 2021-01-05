#!/bin/bash

export PYTHONPATH=.
python evaluation_save_all.py -e purity
python evaluation_save_all.py -e nmi
python evaluation_save_all.py -e ari
echo 'Done...'
