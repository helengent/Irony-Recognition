#!/bin/bash

exp_dir=../Data/globalVector
metaData=../../../AudioData/metaDataPruned.txt

python3 ./classification/main.py $exp_dir $metaData