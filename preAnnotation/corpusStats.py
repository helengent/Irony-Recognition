#!/usr/bin/env python3

import sys
import numpy as np
import pandas as pd

sys.path.append("../AcousticFeatureExtraction/preProcessing")
from pruneByDur import recordDurations


def main():

    print()
    print("**********************************************************")
    print("Full Corpus Statistics")
    print("**********************************************************")
    print()

    durDict = pd.read_csv("../AcousticFeatureExtraction/preProcessing/good_durations.csv")

    #total number of samples in corpus
    print("Total Samples: {}".format(len(durDict)))

    #distribution of ironic vs non-ironic samples
    print("\tIronic Samples: {}".format(durDict["label"].tolist().count("I")))
    print("\tNon-Ironic Samples: {}".format(durDict["label"].tolist().count("N")))
    print()

    #total audio length of corpus
    durations = durDict["duration"].tolist()
    print("Total audio length of corpus: {} hours".format(np.round(np.sum(durations) / 60 / 60, 2)))

    #duration statistics for samples
    print()
    print("Duration Statistics (in seconds)")
    print()
    durStats = {"mean": np.round(np.mean(durations), 2), 
                "sd": np.round(np.std(durations), 2), 
                "min": np.round(np.min(durations), 2),
                "5th percentile": np.round(np.percentile(durations, 5), 2), 
                "1st quartile": np.round(np.percentile(durations, 25), 2), 
                "median":  np.round(np.percentile(durations, 50), 2), 
                "3rd quartile": np.round(np.percentile(durations, 75), 2), 
                "95th percentile": np.round(np.percentile(durations, 95), 2), 
                "max": np.round(np.max(durations), 2)}
    durStats = pd.DataFrame(durStats, index=[0])
    with pd.option_context('display.max_rows', None, 'display.max_columns', None):
        print(durStats)


    ###Pruned dataset###

    print()
    print("**********************************************************")
    print("Pruned Dataset Statistics")
    print("**********************************************************")
    print()

    prunedDurDict = recordDurations("../AudioData/GatedPruned3")
    prunedDurDict = pd.DataFrame(prunedDurDict)

    #total number of samples in corpus
    print("Total Samples: {}".format(len(prunedDurDict)))

    #distribution of ironic vs non-ironic samples
    print("\tIronic Samples: {}".format(prunedDurDict["label"].tolist().count("I")))
    print("\tNon-Ironic Samples: {}".format(prunedDurDict["label"].tolist().count("N")))
    print()

    #rms of samples after normalization

    #total audio length of pruned dataset
    durations = prunedDurDict["duration"].tolist()
    print("Total audio length of pruned dataset: {} hours".format(np.round(np.sum(durations) / 60 / 60, 2)))

    #duration statistics of samples
    print()
    print("Duration Statistics (in seconds)")
    print()
    durStats = {"mean": np.round(np.mean(durations), 2), 
                "sd": np.round(np.std(durations), 2), 
                "min": np.round(np.min(durations), 2),
                "5th percentile": np.round(np.percentile(durations, 5), 2), 
                "1st quartile": np.round(np.percentile(durations, 25), 2), 
                "median":  np.round(np.percentile(durations, 50), 2), 
                "3rd quartile": np.round(np.percentile(durations, 75), 2), 
                "95th percentile": np.round(np.percentile(durations, 95), 2), 
                "max": np.round(np.max(durations), 2)}
    durStats = pd.DataFrame(durStats, index=[0])

    with pd.option_context('display.max_rows', None, 'display.max_columns', None):
        print(durStats)
    print()

    #Distribution by speaker
    speakerList = [item.split("_")[1][0] for item in prunedDurDict["filename"].tolist()]
    prunedDurDict["speaker"] = speakerList

    print("Speaker class distributions:")
    for s in list(set(speakerList)):

        smallSet = prunedDurDict[prunedDurDict["speaker"] == s]
        labs = smallSet["label"].tolist()

        print("Speaker {}\ttotal samples: {}".format(s, len(labs)))
        print("\t\tIronic Samples: {}".format(labs.count("I")))
        print("\t\tNon-Ironic Samples: {}".format(labs.count("N")))
        print("Total audio length for speaker {}:\t{} minutes".format(s, np.round(np.sum(smallSet['duration'].tolist()) / 60, 2)))
        print()

    diff = prunedDurDict['label'].tolist().count("N") - prunedDurDict['label'].tolist().count("I")

    subI = prunedDurDict[prunedDurDict["label"] == "I"]
    durMeanI = np.mean(subI["duration"].tolist())
    durDiffs = [np.abs(item - durMeanI) for item in subI["duration"].tolist()]
    subI["durDiffs"] = durDiffs
    print("Mean variance from duration mean for ironic samples\t{}".format(np.round(np.mean(subI["durDiffs"].tolist()), 2)))

    subN = prunedDurDict[prunedDurDict["label"] == "N"]
    durMeanN = np.mean(subN["duration"].tolist())
    durDiffs = [np.abs(item - durMeanN) for item in subN["duration"].tolist()]
    subN["durDiffs"] = durDiffs
    subN = subN.sort_values("durDiffs")
    print("Mean variance from duration mean for non-ironic samples\t{}".format(np.round(np.mean(subN["durDiffs"].tolist()), 2)))



if __name__=="__main__":

    main()