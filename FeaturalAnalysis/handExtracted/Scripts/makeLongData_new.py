#!/usr/bin/env python3

import os
import time
import numpy as np
import pandas as pd
from glob import glob
import multiprocessing as mp
from functools import partial


def interpolateF0(subset, name, win_size):

    filename = [name.split("-")[0]] * len(subset)
    speaker = [name.split("_")[1][0]] * len(subset)
    label = [name.split("-")[1]] * len(subset)

    #time is a percentage of the total length of the file
    time = [(i+1)/len(subset) for i in range(len(subset))]

    subset["filename"] = filename
    subset["speaker"] = speaker
    subset["label"] = label
    subset["time"] = time
    
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

    sbubset = sbubset.reindex(columns=["filename", "speaker", "label", "time", measure])
    
    return sbubset


def interpolate(subset, measureName, name, win_size):
    
    subset = subset.replace(0, np.nan)

    colName = measureName + "Num"
    longInterpolated = pd.DataFrame()

    for c, subsubset in subset.iteritems():

        subsubset = pd.DataFrame(subsubset)

        filename = [name.split("-")[0]] * len(subset)
        speaker = [name.split("_")[1][0]] * len(subset)
        label = [name.split("-")[1]] * len(subset)
        measureCount = [(c+1)] * len(subset)

        #time is a percentage of the total length of the file
        time = [(i+1)/len(subset) for i in range(len(subset))]

        subsubset["filename"] = filename
        subsubset["speaker"] = speaker
        subsubset["label"] = label
        subsubset["time"] = time
        subsubset[colName] = measureCount

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

        try:
            sbubset = sbubset.rename(columns={sbubset.columns[0]: measure})
            sbubset = sbubset.reindex(columns=["filename", "speaker", "label", "time", colName, measure])
        except:
            print(sbubset)
            print(sbubset.columns)

        longInterpolated = longInterpolated.append(sbubset, ignore_index=True)
    
    # longInterpolated = longInterpolated.rename(columns={longInterpolated.columns[0]: measure})
    # longInterpolated = longInterpolated.reindex(columns=["filename", "speaker", "label", "time", colName, measure])
    return longInterpolated


def feedInterp(f, measure, win_size):

    name = os.path.basename(f).strip(".wav")
        
    measureFile = pd.read_csv("../../../../Data/AcousticData/{}/{}.csv".format(measure, name), header = None)

    if measureFile.shape[1] == 1:
        measureFile = measureFile.rename(columns={measureFile.columns[0]: measure})
        smoothed = interpolateF0(measureFile, name, win_size)
    else:
        smoothed = interpolate(measureFile, measure, name, win_size)
    
    return smoothed


def main(measure, fileList, win_size):

    print(measure, len(fileList))

    availableCPU = mp.cpu_count()

    with mp.Pool(availableCPU) as p:

        feed = partial(feedInterp, measure=measure, win_size=win_size)

        smoothedFiles = p.map(feed, fileList)

    smoothed = pd.concat(smoothedFiles, ignore_index=True)

    print(smoothed.shape)
    print()
    smoothed.to_csv("../../../../Data/AcousticData/Pruned3_10ms/{}_ready_for_gamms.csv".format(measure), index=False)


if __name__=="__main__":

    measures = ["f0", "mfcc", 'ams', 'plp', 'hnr']
    # measures = ["hnr"]

    fileList = glob("../../../AudioData/GatedPruned3/*.wav")

    t0 = time.time()
    for measure in measures:
        main(measure, fileList, 0.01)
    t1 = time.time()
    print("Data longification completed in {} minutes".format((t1-t0)/60))