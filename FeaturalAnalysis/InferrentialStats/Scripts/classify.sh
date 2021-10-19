#!/bin/bash
#This, and everything in the classification directory, is courtesy of Chase Adams

exp_dir=../Data/globalVector
metaData=../../../AudioData/metaDataPruned.txt

python3 ./classification/main.py $exp_dir $metaData