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

        # if i % 100 == 0:
        #     print(i)

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

    print("Adjusting for Class Balance")

    speakerList = [item.split("_")[1][0] for item in subSet["filename"].tolist()]
    subSet["speaker"] = speakerList

    print("Speaker class distributions:")
    for s in list(set(speakerList)):

        smallSet = subSet[subSet["speaker"] == s]
        labs = smallSet["label"].tolist()

        print("Speaker {}\t{} ironic samples\t{} non-ironic samples".format(s, labs.count("I"), labs.count("N")))

    diff = subSet['label'].tolist().count("N") - subSet['label'].tolist().count("I")
    print("Removing {} samples to achieve class balance")

    subI = subSet[subSet["label"] == "I"]
    durMeanI, durSDI = getDurationStats(subI["duration"].tolist())
    durDiffs = [np.abs(item - durMeanI) for item in subI["duration"].tolist()]
    subI["durDiffs"] = durDiffs
    print("Mean variance from duration mean for ironic samples\t{}".format(np.round(np.mean(subI["durDiffs"].tolist()), 2)))

    subN = subSet[subSet["label"] == "N"]
    durMeanN, durSDN = getDurationStats(subN["duration"].tolist())
    durDiffs = [np.abs(item - durMeanN) for item in subN["duration"].tolist()]
    subN["durDiffs"] = durDiffs
    subN = subN.sort_values("durDiffs")
    print("Mean variance from duration mean for non-ironic samples\t{}".format(np.round(np.mean(subN["durDiffs"].tolist()), 2)))

    newN = pd.DataFrame()
    idx = np.round(np.linspace(0, subN.shape[0] - 1, subI.shape[0])).astype(int)

    i = 0
    for _, row in subN.iterrows():
        if i in idx:
            newN = newN.append(row)
        i += 1

    subN = newN

    print(subI.shape)
    print(subN.shape)

    subSet = subI.append(subN)

    print("Removed extra non-ironic samples with greatest variance from duration mean")
    print("Mean variance from duration mean for ironic samples\t{}".format(np.round(np.mean(subI["durDiffs"].tolist()), 2)))
    print("Mean variance from duration mean for non-ironic samples\t{}".format(np.round(np.mean(subN["durDiffs"].tolist()), 2)))

    print("Saving pruned dataset to separate directory")

    sys.exit()

    if not os.path.isdir(outDir):
        os.mkdir(outDir)

    for f in subSet['filename'].tolist():

        shutil.copy("{}/{}.wav".format(allDir, f), "{}/{}.wav".format(outDir, f))


if __name__=="__main__":

    allDir = "../../AudioData/All3/good"
    outDir = "../../AudioData/newTest"

    main(allDir, outDir)