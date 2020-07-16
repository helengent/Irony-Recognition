#!/bin/bash

for i in ../TestWaves/*/*/*
do
name="../mono_test/${i:13}"
sox $i $name remix 1,2
done