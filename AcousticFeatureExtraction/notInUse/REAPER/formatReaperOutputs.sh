#!/bin/bash

#This script is meant to be executed in the directory where the .f0 files are

for i in *.f0
do
    echo $i
    awk -F" " '{ if($2==1) {print $3;} else {print "NaN";}}' $i >> $i.p
done