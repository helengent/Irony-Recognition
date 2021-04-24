#!/bin/bash

dir=$1
outdir="$(basename -- $dir)"

mkdir -p data/${outdir}_manual

for i in ${dir}/*.wav; do
    play $i
    read transcription

    outfile="$(basename -- $i)"
    outfile="${outfile%.wav}"
    outfile="data/${outdir}_manual/${outfile}.txt"

    echo $transcription >> $outfile

    done