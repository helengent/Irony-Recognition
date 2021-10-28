#!/usr/bin/env python3

import os
import sys
import numpy as np
import pandas as pd
from glob import glob
from pandas.core import frame
from sklearn.preprocessing import StandardScaler

sys.path.append(os.path.join('/'.join(sys.path[1].split("/")[:-3]), 'AcousticFeatureExtraction'))
from speaker import Speaker
from sd import sd


def Hz2Mels(value):

    return (1/np.log(2)) * (np.log(1 + (value/1000))) * 1000


def normF0(f0, speaker, normType="m"):
    
    #possible normTypes: "m", "z", "d"

    f0_mean = speaker.getSpeakerMeanF0()
    f0_sd = speaker.getSpeakerSDF0()

    if normType == "m":
        normedVec = [(Hz2Mels(value) - f0_mean)/f0_mean for value in f0]
    elif normType == "z":
        normedVec = [(Hz2Mels(value) - f0_mean)/f0_sd for value in f0]
    elif normType == "d":
        normedVec = [Hz2Mels(value) - f0_mean for value in f0]
    else:
        raise ValueError("Invalid normType")

    return np.array(normedVec)


def scaleIt(bigList, scaler):

    chunkSize = np.shape(bigList[0])[0]

    biggestFriend = np.zeros((len(bigList), chunkSize, np.shape(bigList[0])[1]))

    longLad = list()

    for utt in bigList:
        for chunk in utt:
            longLad.append(chunk)
    
    longLad = np.array(longLad)

    scaled = scaler.transform(longLad)

    start = 0
    stop = chunkSize
    for i in range(len(bigList)):

        small = scaled[start:stop]

        biggestFriend[i, :, :] = small

        start += chunkSize
        stop += chunkSize

    return biggestFriend


def makeScaler(nList):

    longLad = list()

    for utt in nList:
        for chunk in utt:
            longLad.append(chunk)
    
    longLad = np.array(longLad)

    scaler = StandardScaler()
    scaler.fit(longLad)

    return(scaler)


def chunkStats(x, percentage):

    if 100 % percentage != 0:
        raise ValueError("percentage must be an integer that 100 is divisible by")

    stats = list()

    numChunks = int(100/percentage)
    chunkSize = int(len(x)/numChunks)
    start = 0

    for n in range(numChunks):

        chunk = x[start:start+chunkSize]

        chunkMean = np.mean(chunk)
        chunkSD = sd(chunk, chunkMean)

        stats.append([chunkMean, chunkSD])

        start += chunkSize

    return stats


# Accepts a pandas dataframe of target files and a list of already extracted acoustic measures
#   Assembles a numpy array with all measures from all files of shape (n, x, y) where
#       n = the length of fileList
#       x = frame_max (the number of frames padded/truncated to per file)
#       y = the number of acoustic measures (possibly multiple per item in measureList e.g. ams=375)
#   the speaker variable is the speaker left OUT of the training and dev data
def assembleArray(listMod, fileList, measureList, fileMod, outDir, speaker=None, f0Normed=False, percentage=10):

    outLabels = list()
    bigFriend = list()
    nfriend = list()

    print("{}\t{} left out\t{}".format(listMod, speaker, fileMod))

    for j, row in fileList.iterrows():

        if (j+1) % 50 == 0:
            print("Working on file {} of {}".format(j+1, fileList.shape[0]))

        s = Speaker(row["speaker"], "../../../../Data/AcousticData/SpeakerMetaData/{}_f0.txt".format(row["speaker"].upper()), "../../../../Data/AcousticData/SpeakerMetaData/{}_avgDur.txt".format(row["speaker"].upper()))

        fileName = row["id"]
        label = row["label"]

        toStack = list()

        if "f0" in measureList:
            f0 = pd.read_csv("../../../../Data/AcousticData/f0/{}.csv".format(fileName))["0"].tolist()
            if f0Normed:
                f0 = normF0(f0, s, normType=f0Normed)
            else:
                f0 = Hz2Mels(np.array(f0))
            f0 = chunkStats(f0, percentage)
            toStack.append(f0)

        if "hnr" in measureList:
            hnr = np.array(pd.read_csv("../../../../Data/AcousticData/hnr/{}.csv".format(fileName))["0"].tolist())
            hnr = chunkStats(hnr, percentage)
            toStack.append(hnr)

        if "mfcc" in measureList:
            for i in range(13):
                mfcc = np.array(pd.read_csv("../../../../Data/AcousticData/mfcc/{}.csv".format(fileName))[str(i)].tolist())
                mfcc = chunkStats(mfcc, percentage)
                toStack.append(mfcc)

        if "plp" in measureList:
            for i in range(9):
                plp = np.array(pd.read_csv("../../../../Data/AcousticData/plp/{}.csv".format(fileName), header=None)[i].tolist())
                plp = chunkStats(plp, percentage)
                toStack.append(plp)

        if "ams" in measureList:
            for i in range(375):
                ams = np.array(pd.read_csv("../../../../Data/AcousticData/ams/{}.csv".format(fileName), header=None)[i].tolist())
                ams = chunkStats(ams, percentage)
                toStack.append(ams)

        assert len(list(set([len(item) for item in toStack]))) == 1

        wholeFriend = np.hstack(toStack)

        bigFriend.append(wholeFriend)
        outLabels.append(label)
        if label == "N":
            nfriend.append(wholeFriend)
    
    #Make the scaler and fit it on nfriend only
    scaler = makeScaler(nfriend)
    biggestFriend = scaleIt(bigFriend, scaler)
    print(np.shape(biggestFriend))

    if speaker:
        with open("{}/{}_{}LeftOut_{}_acoustic.npy".format(outDir, listMod, speaker, fileMod), 'wb') as f:
            np.save(f, biggestFriend)
        with open("{}/{}_{}LeftOut_{}_labels.npy".format(outDir, listMod, speaker, fileMod), 'wb') as f:
            np.save(f, np.array(outLabels))
    else:
        with open("{}/{}_{}_acoustic.npy".format(outDir, listMod, fileMod), 'wb') as f:
            np.save(f, biggestFriend)
        with open("{}/{}_{}_labels.npy".format(outDir, listMod, fileMod), 'wb') as f:
            np.save(f, np.array(outLabels))


# Returns dictionary of pandas dataframes with lists of files in leave on speaker out splits
def LOSOLists(listMod, speakerList):

    LOSOList = {"left_out": [], "train": [], "dev": [], "test": []}

    for s in speakerList:

        train_df = pd.DataFrame()
        dev_df = pd.DataFrame()
        all_df = pd.read_csv("../../../AudioData/splitLists/Gated{}_{}_ALL.csv".format(listMod, s))

        smolList = speakerList[:]
        smolList.remove(s)

        for speaker in smolList:
            train_df = train_df.append(pd.read_csv("../../../AudioData/splitLists/Gated{}_{}_train.csv".format(listMod, speaker)), ignore_index=True)
            dev_df = dev_df.append(pd.read_csv("../../../AudioData/splitLists/Gated{}_{}_dev.csv".format(listMod, speaker)), ignore_index=True)

        LOSOList["left_out"].append(s)
        LOSOList["train"].append(train_df)
        LOSOList["dev"].append(dev_df)
        LOSOList["test"].append(all_df)

    return LOSOList


def main(listMod, speakerList, measureList, outDir, speakerSplit="independent", f0Normed=False):

    if speakerSplit == "independent":
        LOSOList = LOSOLists(listMod, speakerList)

        for i, speaker in enumerate(LOSOList["left_out"]):

            for item in ["dev", "train", "test"]:
                assembleArray(listMod, LOSOList[item][i], measureList, item, outDir, speaker=speaker, f0Normed=f0Normed)

    else:
        train = pd.read_csv("../../../AudioData/splitLists/Gated{}_train.csv".format(listMod))
        dev = pd.read_csv("../../../AudioData/splitLists/Gated{}_dev.csv".format(listMod))
        test = pd.read_csv("../../../AudioData/splitLists/Gated{}_test.csv".format(listMod))

        for item, mod in zip([dev, train, test], ["dev", "train", "test"]):
            assembleArray(listMod, item, measureList, mod, outDir, speaker=False, f0Normed=f0Normed)


if __name__=="__main__":

    measureList = ["f0", "hnr", "mfcc", "plp"]
    speakerList = ["c", "d", "e", "f", "h", "j", "k", "o", "q", "s", "t", "u"]
    listMod = "Pruned3"
    f0Normed = False
    text = "asr"
    speakerSplits = ["dependent", "independent"]
    outDir = "/home/hmgent2/Data/ModelInputs/percentChunks"

    for speakerSplit in speakerSplits:
        main(listMod, speakerList, measureList, outDir, f0Normed=f0Normed, speakerSplit=speakerSplit)