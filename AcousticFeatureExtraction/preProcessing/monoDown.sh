#!/bin/bash

for i in *.wav; do o="downsampled/$i"; sox "$i" "$o" channels 1 rate 16000; done