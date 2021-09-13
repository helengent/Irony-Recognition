#!/bin/bash

for i in *.wav; do
	play $i
	read decision

	if [ $decision == y ]; then
		mv $i good2/
	elif [ $decision == n ]; then
		mv $i bad/
	fi
done
