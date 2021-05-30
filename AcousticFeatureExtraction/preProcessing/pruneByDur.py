#!/usr/bin/env python3

import os
import sys
import shutil
import numpy as np
import pandas as pd
from glob import glob

sys.path.append(os.path.dirname(sys.path[0]))
from lib.WAVReader import WAVReader as WR
from sd import sd


def getDurationStats(durList):

    durMean = np.mean(durList)
    durSD = sd(durList, durMean)

    return durMean, durSD


def recordDurations(allDir):

    durDict = {"filename": [], "duration": [], "label": []}

    for i, wav in enumerate(glob("{}/*.wav".format(allDir))):

        if i % 100 == 0:
            print(i)

        readr = WR(wav)
        base = os.path.basename(wav).split(".")[0]

        durDict["filename"].append(base)
        durDict["duration"].append(readr.getDuration())
        durDict["label"].append(base[-1])

    return durDict

def main(allDir, outDir):

    # durDict = recordDurations(allDir)
    # durDict = pd.DataFrame(durDict)

    # durDict.to_csv("{}_durations.csv".format(os.path.basename(allDir)))
    durDict = pd.read_csv("{}_durations.csv".format(os.path.basename(allDir)))

    durMean, durSD = getDurationStats(durDict['duration'].tolist())

    upperLim = durMean + (2.5 * durSD)
    lowerLim = durMean - (2.5 * durSD)

    if lowerLim < 0:
        lowerLim = 0

    subSet = durDict[durDict["duration"] < upperLim]
    subSet = subSet[subSet["duration"] > lowerLim]

    print("Excluding {} outliers".format(durDict.shape[0] - subSet.shape[0]))
    print("{} outliers are labeled ironic".format(durDict['label'].tolist().count("I") - subSet['label'].tolist().count("I")))
    print("{} outliers are labeled non-ironic".format(durDict['label'].tolist().count("N") - subSet['label'].tolist().count("N")))
    print("New class balance:\t{}% ironic\t{}% non-ironic".format(np.round(subSet['label'].tolist().count("I") / subSet.shape[0], 2), np.round(subSet['label'].tolist().count("N") / subSet.shape[0], 2)))

    print("Saving pruned dataset to separate directory")

    if not os.path.isdir(outDir):
        os.mkdir(outDir)

    for f in subSet['filename'].tolist():

        shutil.copy("{}/{}.wav".format(allDir, f), "{}/{}.wav".format(outDir, f))


if __name__=="__main__":

    allDir = "../../../All2"
    outDir = "../../../Pruned2"

    main(allDir, outDir)