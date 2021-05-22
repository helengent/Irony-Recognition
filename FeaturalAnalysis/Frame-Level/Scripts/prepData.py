#!/usr/bin/env python3

import numpy as np
import pandas as pd
from glob import glob
from sklearn.preprocessing import StandardScaler


def scaleIt(bigList, frame_max):

    biggestFriend = np.zeros((len(bigList), frame_max, np.shape(bigList[0])[1]))
    print(np.shape(biggestFriend))

    longLad = list()

    for utt in bigList:
        for frame in utt:
            longLad.append(frame)
    
    longLad = np.array(longLad)
    print(longLad.shape)

    scaler = StandardScaler()
    scaled = scaler.fit_transform(longLad)

    start = 0
    stop = frame_max
    for i in range(len(bigList)):

        small = scaled[start:stop]

        biggestFriend[i, :, :] = small

        start += frame_max
        stop += frame_max

    return biggestFriend


# Accepts a pandas dataframe of target files and a list of already extracted acoustic measures
#   Assembles a numpy array with all measures from all files of shape (n, x, y) where
#       n = the length of fileList
#       x = frame_max (the number of frames padded/truncated to per file)
#       y = the number of acoustic measures (possibly multiple per item in measureList e.g. ams=375)
#   the speaker variable is the speaker left OUT of the training and dev data
def assembleArray_Raw(listMod, fileList, measureList, frame_max, fileMod, speaker=None):

    outLabels = list()
    bigFriend = list()

    print("{}\t{} left out\t{}".format(listMod, speaker, fileMod))

    for j, row in fileList.iterrows():

        print("Working on file {} of {}".format(j+1, fileList.shape[0]))

        fileName = row["id"]
        label = row["label"]

        toStack = list()

        if "f0" in measureList:
            f0 = np.array(pd.read_csv("../../../AcousticData/f0/{}.csv".format(fileName))["0"].tolist())
            while len(f0) < frame_max:
                f0 = np.append(f0, 0)
            if len(f0) > frame_max:
                f0 = f0[:frame_max]
            f0 = f0.reshape((-1, 1))
            toStack.append(f0)

        if "hnr" in measureList:
            hnr = np.array(pd.read_csv("../../../AcousticData/hnr/{}.csv".format(fileName))["0"].tolist())
            while len(hnr) < frame_max:
                hnr = np.append(hnr, 0)
            if len(hnr) > frame_max:
                hnr = hnr[:frame_max]
            hnr = hnr.reshape((-1, 1))
            toStack.append(hnr)

        if "mfccs" in measureList:
            for i in range(13):
                mfcc = np.array(pd.read_csv("../../../AcousticData/mfccs/{}.csv".format(fileName))[str(i)].tolist())
                while len(mfcc) < frame_max:
                    mfcc = np.append(mfcc, 0)
                if len(mfcc) > frame_max:
                    mfcc = mfcc[:frame_max]
                mfcc = mfcc.reshape((-1, 1))
                toStack.append(mfcc)

        if "rastaplp" in measureList:
            for i in range(9):
                plp = np.array(pd.read_csv("../../../AcousticData/rastaplp/{}.csv".format(fileName), header=None).transpose()[i].tolist())
                while len(plp) < frame_max:
                    plp = np.append(plp, 0)
                if len(plp) > frame_max:
                    plp = plp[:frame_max]
                plp = plp.reshape((-1, 1))
                toStack.append(plp)

        if "ams" in measureList:
            for i in range(375):
                ams = np.array(pd.read_csv("../../../AcousticData/ams/{}.csv".format(fileName), header=None).transpose()[i].tolist())
                while len(ams) < frame_max:
                    ams = np.append(ams, 0)
                if len(ams) > frame_max:
                    ams = ams[:frame_max]
                ams = ams.reshape((-1, 1))
                toStack.append(ams)

        wholeFriend = np.hstack(toStack)
        bigFriend.append(wholeFriend)

        outLabels.append(label)

    
    biggestFriend = scaleIt(bigFriend, frame_max)

    with open("../Data/{}_{}LeftOut_{}_acoustic.npy".format(listMod, speaker, fileMod), 'wb') as f:
        np.save(f, biggestFriend)
    with open("../Data/{}_{}LeftOut_{}_labels.npy".format(listMod, speaker, fileMod), 'wb') as f:
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


def main(listMod, speakerList, measureList, frame_max, speakerSplit="independent"):

    if speakerSplit == "independent":
        LOSOList = LOSOLists(listMod, speakerList)

        for i, speaker in enumerate(LOSOList["left_out"]):

            for item in ["dev", "train", "test"]:
                assembleArray_Raw(listMod, LOSOList[item][i], measureList, frame_max, item, speaker=speaker)

    else:
        train = pd.read_csv("../../../AudioData/splitLists/Gated{}_train.csv".format(listMod))
        dev = pd.read_csv("../../../AudioData/splitLists/Gated{}_dev.csv".format(listMod))
        test = pd.read_csv("../../../AudioData/splitLists/Gated{}_test.csv".format(listMod))

    




if __name__=="__main__":

    measureList = ["ams", "f0", "hnr", "mfccs", "rastaplp"]
    frame_max = 550 #Chosen because it's roughly the 90th percentile for file length in what I have so far
    
    speakerLists = [["c", "d", "e"], ["b", "g", "p", "r", "y"], ["b", "g", "p", "r", "y"]]
    listMods = ["ANH", "Pruned", "All"]

    # speakerList = ["b", "g", "p", "r", "y"]
    # listMod ="All"

    for speakerList, listMod in zip(speakerLists, listMods):

        main(listMod, speakerList, measureList, frame_max)