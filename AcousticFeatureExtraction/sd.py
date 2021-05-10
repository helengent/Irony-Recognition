#!/usr/bin/env python3

import numpy as np

def sd(list, avg):
	newList = []
	for item in list:
		value = (item - avg) ** 2
		newList.append(value)
	mean = sum(newList)/len(newList)
	return(np.sqrt(mean))

