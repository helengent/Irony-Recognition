#!/bin/bash

wav_dir=$1
txt_dir=$2

# out_dir="$(basename -- $txt_dir)"
# out_dir=data/${out_dir}

out_dir=$2

mkdir -p $out_dir

for i in ${wav_dir}/*.wav; do

    filename="$(basename -- $i)"
    filename="${filename%.wav}"

    t=${txt_dir}/${filename}.txt

    outName=${out_dir}/$filename.TextGrid
    
    python3 lib/Penn/align.py $i $t $outName

    done