#!/usr/bin/env bash

echo "Need pytorch>=1.0.0"
source activate pytorch1.0.0
#source activate pytorch_p36

rm -rf build *.egg-info dist
python3 setup.py install
