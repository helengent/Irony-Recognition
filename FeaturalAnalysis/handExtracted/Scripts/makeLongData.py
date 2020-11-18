#!/usr/bin/env python3

import time
import numpy as np
import pandas as pd

def interpolateF0(longData):
    
    fileNames = list(set(longData.filename.tolist()))

    longInterpolated = pd.DataFrame()

    for i, f in enumerate(fileNames):
        subset = longData[longData.filename == f]
        subset = subset.replace(0, np.nan)

        #linspace needs to be called for each filename since there's different lengths of data for each
        indices = np.linspace(0, subset.shape[0]-1, num=75)
        indices = [int(np.round(i)) for i in indices]
        
        sbubset = subset.take(indices)
        sbubset = sbubset.replace(0, np.nan)

        sbubset = sbubset.interpolate()
        if sbubset.isnull().values.any():
            sbubset = sbubset.bfill()

        if sbubset.isnull().values.any():
            print("Okay. Go ahead and cry.")

        longInterpolated = longInterpolated.append(sbubset, ignore_index=True)
    
    return longInterpolated

def interpolate(longData, measureName):
    
    fileNames = list(set(longData.filename.tolist()))
    longInterpolated = pd.DataFrame()

    for i, f in enumerate(fileNames):
        subset = longData[longData.filename == f]
        subset = subset.replace(0, np.nan)

        colName = measureName + "Num"
        subNames = list(set(longData[colName].tolist()))

        for c in subNames:

            subsubset = subset[subset[colName] == c]

            #linspace needs to be called for each filename since there's different lengths of data for each
            indices = np.linspace(0, subsubset.shape[0]-1, num=75)
            indices = [int(np.round(i)) for i in indices]
            
            sbubset = subsubset.take(indices)
            sbubset = sbubset.replace(0, np.nan)

            sbubset = sbubset.interpolate()
            if sbubset.isnull().values.any():
                sbubset = sbubset.bfill()

            if sbubset.isnull().values.any():
                print("Okay. Go ahead and cry.")

            longInterpolated = longInterpolated.append(sbubset, ignore_index=True)
    
    return longInterpolated


def main(measure):

    print(measure)

    long_df = pd.read_csv("../Data/Pruned_10ms/{}_long.csv".format(measure))
    print(long_df.shape)

    if measure == "f0":
        smoothed = interpolateF0(long_df)

    else:
        smoothed = interpolate(long_df, measure)

    print(smoothed.shape)
    print()
    smoothed.to_csv("../Data/Pruned_10ms/{}_ready_for_gamms.csv".format(measure), index=False)


if __name__=="__main__":

    # measures = ["f0", "mfcc", 'ams', 'plp']
    measures = ["ams"]
    t0 = time.time()
    for measure in measures:
        main(measure)
    t1 = time.time()
    print("Data longification completed in {} minutes".format((t1-t0)/60))