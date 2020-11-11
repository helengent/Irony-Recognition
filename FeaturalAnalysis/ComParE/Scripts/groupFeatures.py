#!/usr/bin/env python3

import numpy as np
import pandas as pd 

def groupFeatures(measureList):
    bigDic = dict()
    brokenMeasures = [measure.split("_") for measure in measureList]
    lens = [len(measure) for measure in brokenMeasures]
    longest = np.max(lens)
    valueList = [[] for i in range(longest)]
    for measure in brokenMeasures:
        while len(measure) < longest:
            measure.append("pad")
        for i, m in enumerate(measure):
            valueList[i].append(m)
    
    for i, value in enumerate(valueList):
        bigDic["{}".format(i)] = value
    
    df = pd.DataFrame(bigDic)
    df.to_csv("BaselineMeasureNames.csv")

if __name__ == "__main__":
    row = pd.read_csv("baselineResults/SPPep12_b1.csv", sep=";", engine="python")

    measureList = row.columns
    groupFeatures(measureList)