#!/usr/bin/env python3

import numpy as np
import pandas as pd
from glob import glob
from sklearn.preprocessing import StandardScaler


def scaleIt(bigList, scaler, frame_max):

    biggestFriend = np.zeros((len(bigList), frame_max, np.shape(bigList[0])[1]))
    print(np.shape(biggestFriend))

    longLad = list()

    for utt in bigList:
        for frame in utt:
            longLad.append(frame)
    
    longLad = np.array(longLad)
    print(longLad.shape)

    scaled = scaler.transform(longLad)

    start = 0
    stop = frame_max
    for i in range(len(bigList)):

        small = scaled[start:stop]

        biggestFriend[i, :, :] = small

        start += frame_max
        stop += frame_max

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


# Accepts a pandas dataframe of target files and a list of already extracted acoustic measures
#   Assembles a numpy array with all measures from all files of shape (n, x, y) where
#       n = the length of fileList
#       x = frame_max (the number of frames padded/truncated to per file)
#       y = the number of acoustic measures (possibly multiple per item in measureList e.g. ams=375)
#   the speaker variable is the speaker left OUT of the training and dev data
def assembleArray_Raw(listMod, fileList, measureList, frame_max, outDir, fileMod, speaker=None):

    outLabels = list()
    bigFriend = list()
    nfriend = list()

    if speaker:
        print("{}\t{} left out\t{}".format(listMod, speaker, fileMod))
    else:
        print("{}\t{}".format(listMod, fileMod))

    for j, row in fileList.iterrows():

        print("Working on file {} of {}".format(j+1, fileList.shape[0]))

        fileName = row["id"]
        label = row["label"]

        toStack = list()

        if "f0" in measureList:
            f0 = np.array(pd.read_csv("../../../../Data/AcousticData/f0/{}.csv".format(fileName))["0"].tolist())
            while len(f0) < frame_max:
                f0 = np.append(f0, 0)
            if len(f0) > frame_max:
                f0 = f0[:frame_max]
            f0 = f0.reshape((-1, 1))
            toStack.append(f0)

        if "hnr" in measureList:
            hnr = np.array(pd.read_csv("../../../../Data/AcousticData/hnr/{}.csv".format(fileName))["0"].tolist())
            while len(hnr) < frame_max:
                hnr = np.append(hnr, 0)
            if len(hnr) > frame_max:
                hnr = hnr[:frame_max]
            hnr = hnr.reshape((-1, 1))
            toStack.append(hnr)

        if "mfcc" in measureList:
            for i in range(13):
                mfcc = np.array(pd.read_csv("../../../../Data/AcousticData/mfcc/{}.csv".format(fileName))[str(i)].tolist())
                while len(mfcc) < frame_max:
                    mfcc = np.append(mfcc, 0)
                if len(mfcc) > frame_max:
                    mfcc = mfcc[:frame_max]
                mfcc = mfcc.reshape((-1, 1))
                toStack.append(mfcc)

        if "plp" in measureList:
            for i in range(9):
                plp = np.array(pd.read_csv("../../../../Data/AcousticData/plp/{}.csv".format(fileName), header=None)[i].tolist())
                while len(plp) < frame_max:
                    plp = np.append(plp, 0)
                if len(plp) > frame_max:
                    plp = plp[:frame_max]
                plp = plp.reshape((-1, 1))
                toStack.append(plp)

        if "ams" in measureList:
            for i in range(375):
                ams = np.array(pd.read_csv("../../../../Data/AcousticData/ams/{}.csv".format(fileName), header=None)[i].tolist())
                while len(ams) < frame_max:
                    ams = np.append(ams, 0)
                if len(ams) > frame_max:
                    ams = ams[:frame_max]
                ams = ams.reshape((-1, 1))
                toStack.append(ams)

        wholeFriend = np.hstack(toStack)
        bigFriend.append(wholeFriend)
        outLabels.append(label)
        if label == "N":
            nfriend.append(wholeFriend)

    scaler = makeScaler(nfriend)
    biggestFriend = scaleIt(bigFriend, scaler, frame_max)

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


def main(listMod, speakerList, measureList, outDir, frame_max, speakerSplit="independent"):

    if speakerSplit == "independent":
        LOSOList = LOSOLists(listMod, speakerList)

        for i, speaker in enumerate(LOSOList["left_out"]):

            for item in ["dev", "train", "test"]:
                assembleArray_Raw(listMod, LOSOList[item][i], measureList, frame_max, outDir, item, speaker=speaker)

    else:
        train = pd.read_csv("../../../AudioData/splitLists/Gated{}_train.csv".format(listMod))
        dev = pd.read_csv("../../../AudioData/splitLists/Gated{}_dev.csv".format(listMod))
        test = pd.read_csv("../../../AudioData/splitLists/Gated{}_test.csv".format(listMod))

        for df, item in zip([dev, train, test], ["dev", "train", "test"]):
            assembleArray_Raw(listMod, df, measureList, frame_max, outDir, item)


if __name__=="__main__":

    measureList = ["f0", "hnr", "mfcc", "plp"]
    frame_max = 550 #Chosen because it's roughly the 90th percentile for file length in what I have so far
    speakerList = ["c", "d", "e", "f", "h", "j", "k", "o", "q", "s", "t", "u"]
    listMod = "Pruned3"
    speakerSplits = ["dependent", "independent"]
    outDir = "/home/hmgent2/Data/ModelInputs/rawSequential"

    for speakerSplit in speakerSplits:
        main(listMod, speakerList, measureList, outDir, frame_max, speakerSplit=speakerSplit)