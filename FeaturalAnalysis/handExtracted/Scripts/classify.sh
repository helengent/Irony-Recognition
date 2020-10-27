#!/bin/bash

exp_dir=../Data/rastaplp
metaData=../../../AudioData/metaDataPruned.txt

python3 ./classification/main.py $exp_dir $metaData