#!/usr/bin/env python3

import sys
from sd import sd

def vecNoZeros(text, lowerLim, upperLim):
    f0vec = []
    for item in text:
        if (item != "NaN") and (item != "NA"):
            item = float(item)
            if lowerLim <= item <= upperLim:
                f0vec.append(item)
    return f0vec

def f0range(text, lowerLim, upperLim):
    numVec = vecNoZeros(text, lowerLim, upperLim)
    ran = max(numVec) - min(numVec)
    return ran

def meanf0(text, lowerLim, upperLim):
    numVec = vecNoZeros(text, lowerLim, upperLim)
    mean = sum(numVec)/len(numVec)
    return mean

def f0sd(text, lowerLim, upperLim):
    numVec = vecNoZeros(text, lowerLim, upperLim)
    mean = meanf0(text, lowerLim, upperLim)
    fsd = sd(numVec, mean)
    return fsd


#takes text input from a reaper output file
def f0VecTime(text, lowerLim, upperLim, ms=5):
    f0vec = []
    if ms == 5:
        for item in text:
            if (item != "NaN") and (item != "NA"):
                item = float(item)
                if lowerLim <= item <= upperLim:
                    f0vec.append(item)
                else:
                    f0vec.append(0)
            else:
                f0vec.append(0)

    elif ms % 5 != 0:
        print("Error: ms must be divisible by 5")
        sys.exit()

    else:
        for i, item in enumerate(text):
            if (item != "NaN") and (item != "NA"):
                item = float(item)
            
            added = False
            if i % ms == 0:
                if type(item) is float:
                    if lowerLim <= item <= upperLim:
                        f0vec.append(item)
                        added = True

                if added == False:
                    n = 1
                    while n <= (ms/5):
                        if i < (len(text) - n):
                            if type(text[i+n]) is float:
                                if lowerLim <= text[i+n] <= upperLim:
                                    f0vec.append(text[i+n])
                                    n = ms + 1
                                    added = True
                                else:
                                    n+=1
                            else:
                                n+=1
                        else:
                            n+=1
                if added == False:
                    f0vec.append(0)

    return f0vec
    

