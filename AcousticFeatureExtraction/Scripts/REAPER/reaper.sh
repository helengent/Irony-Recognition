#!/bin/bash

#This script is meant to be executed in the directory where the .wav files are

# split channel

for i in *.wav
do
# name=`basename $i .wav`    
# sox -c 2 $i -c 1 $name_one.wav
reaper -i $i -f $i.f0 -e 0.005 -a
tail -n+8 $i.f0 > tmp
mv tmp $i.f0
done


