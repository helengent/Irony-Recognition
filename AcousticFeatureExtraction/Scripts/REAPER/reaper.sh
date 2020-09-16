#!/bin/bash

#This script is meant to be executed in the directory where the .wav files are

# split channel

for i in *.wav
do
name=`basename $i .wav`    
sox -c 2 $i -c 1 $name_one.wav
reaper -i $name_one.wav -f $name.f0 -a
tail -n+8 $name.f0 > tmp
mv tmp $name.f0
done


