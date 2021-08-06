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
from preProcessing.ASR.parseTextGrid import parse


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


def scaleIt(bigList, shapeList, frame_max, tokMax):

    # biggestFriend = np.zeros((len(bigList), tokMax, frame_max, np.shape(bigList[0])[1]))
    # print(np.shape(biggestFriend))

    padStack = list()
    longLad = list()

    for utt in bigList:
        for word in utt:
            for features in word:
                longLad.append(features)
    
    longLad = np.array(longLad)

    scaler = StandardScaler()
    scaled = scaler.fit_transform(longLad)

    for uttShapes in shapeList:
        uttStack = list()
        for shape in uttShapes:
            lilChunk = longLad[:shape[0]]
            longLad = longLad[shape[0]:]

            if lilChunk.shape[0] < frame_max:
                pad = np.zeros((frame_max-lilChunk.shape[0], lilChunk.shape[1]))
                lilChunk = np.concatenate((lilChunk, pad))
            elif lilChunk.shape[0] > frame_max:
                lilChunk = lilChunk[:frame_max]
            
            uttStack.append(lilChunk)
        
        uttStack = np.stack(uttStack)
        if uttStack.shape[0] < tokMax:
            pad = np.zeros((tokMax-uttStack.shape[0], uttStack.shape[1], uttStack.shape[2]))
            uttStack = np.concatenate((uttStack, pad))
        elif uttStack.shape[0] > tokMax:
            uttStack = uttStack[:tokMax]

        padStack.append(uttStack)

    biggestFriend = np.stack(padStack)

    return biggestFriend


def lineUp(data, words):

    matched = list()

    for w in words:
        start_idx = int(w[1] * 100)
        end_idx = int(w[2] * 100)

        if end_idx <= len(data):
            lilChunk = data[start_idx:end_idx]
            matched.append(lilChunk)

        elif start_idx < len(data):
            lilChunk = data[start_idx:]
            matched.append(lilChunk)

    return matched


# Accepts a pandas dataframe of target files and a list of already extracted acoustic measures
#   Assembles a numpy array with all measures from all files of shape (n, x, y) where
#       n = the length of fileList
#       x = frame_max (the number of frames padded/truncated to per file)
#       y = the number of acoustic measures (possibly multiple per item in measureList e.g. ams=375)
#   the speaker variable is the speaker left OUT of the training and dev data
def assembleArray_Raw(listMod, fileList, measureList, frame_max, tokMax, fileMod, speaker=None, f0Normed=False, text="asr",):

    badList = list()
    outLabels = list()
    bigFriend = list()
    shapeList = list()

    print("{}\t{} left out\t{}".format(listMod, speaker, fileMod))

    for j, row in fileList.iterrows():

        if (j+1) % 50 == 0:
            print("Working on file {} of {}".format(j+1, fileList.shape[0]))

        s = Speaker(row["speaker"], "../../../../Data/AcousticData/SpeakerMetaData/{}_f0.txt".format(row["speaker"].upper()), "../../../../Data/AcousticData/SpeakerMetaData/{}_avgDur.txt".format(row["speaker"].upper()))

        fileName = row["id"]
        label = row["label"]

        try:
            _, _, words, phones = parse("../../../../Data/TextData/{}_{}/{}.TextGrid".format(listMod, text, fileName))
        except:
            print("MISSING TEXTGRID {}".format(fileName))
            words = []
            phones = []

        toStack = list()

        if "f0" in measureList:
            f0 = pd.read_csv("../../../../Data/AcousticData/f0/{}.csv".format(fileName))["0"].tolist()
            if f0Normed:
                f0 = normF0(f0, s, normType=f0Normed)
            else:
                f0 = np.array(Hz2Mels(f0))
            f0 = f0.reshape((-1, 1))
            toStack.append(f0)

        if "hnr" in measureList:
            hnr = np.array(pd.read_csv("../../../../Data/AcousticData/hnr/{}.csv".format(fileName))["0"].tolist())
            hnr = hnr.reshape((-1, 1))
            toStack.append(hnr)

        if "mfcc" in measureList:
            for i in range(13):
                mfcc = np.array(pd.read_csv("../../../../Data/AcousticData/mfcc/{}.csv".format(fileName))[str(i)].tolist())
                mfcc = mfcc.reshape((-1, 1))
                toStack.append(mfcc)

        if "plp" in measureList:
            for i in range(9):
                plp = np.array(pd.read_csv("../../../../Data/AcousticData/plp/{}.csv".format(fileName), header=None)[i].tolist())
                plp = plp.reshape((-1, 1))
                toStack.append(plp)

        if "ams" in measureList:
            for i in range(375):
                ams = np.array(pd.read_csv("../../../../Data/AcousticData/ams/{}.csv".format(fileName), header=None)[i].tolist())
                ams = ams.reshape((-1, 1))
                toStack.append(ams)

        arrayShapes = [item.shape[0] for item in toStack]
        maxLen = np.max(arrayShapes)
        newToStack = list()

        if len(list(set(arrayShapes))) != 1:
            for item in toStack:
                if item.shape[0] < maxLen:
                    pad = np.zeros((maxLen - item.shape[0], item.shape[1]))
                    item = np.concatenate((item, pad))
                newToStack.append(item)
        else:
            newToStack = toStack

        wholeFriend = np.hstack(newToStack)
        wholeFriend = lineUp(wholeFriend, words)
        shapes = [item.shape for item in wholeFriend]

        if len(wholeFriend) > 0:
            bigFriend.append(wholeFriend)
            shapeList.append(shapes)
            outLabels.append(label)
        else:
            badList.append(fileName)
    
    biggestFriend = scaleIt(bigFriend, shapeList, frame_max, tokMax)
    print(np.shape(biggestFriend))

    if speaker:
        with open("../Data/{}_{}LeftOut_{}_acoustic.npy".format(listMod, speaker, fileMod), 'wb') as f:
            np.save(f, biggestFriend)
        with open("../Data/{}_{}LeftOut_{}_labels.npy".format(listMod, speaker, fileMod), 'wb') as f:
            np.save(f, np.array(outLabels))
    else:
        with open("../Data/{}_{}_acoustic.npy".format(listMod, fileMod), 'wb') as f:
            np.save(f, biggestFriend)
        with open("../Data/{}_{}_labels.npy".format(listMod, fileMod), 'wb') as f:
            np.save(f, np.array(outLabels))

    return badList


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


def main(listMod, speakerList, measureList, frame_max=200, tokMax=50, speakerSplit="independent", f0Normed=False, text="asr"):

    if speakerSplit == "independent":
        LOSOList = LOSOLists(listMod, speakerList)

        for i, speaker in enumerate(LOSOList["left_out"]):

            for item in ["dev", "train", "test"]:
                badList = assembleArray_Raw(listMod, LOSOList[item][i], measureList, frame_max, tokMax, item, speaker=speaker, f0Normed=f0Normed, text=text)
                with open("BAD.txt", "a+") as f:
                    for item in badList:
                        f.write("{}\n".format(item))

    else:
        train = pd.read_csv("../../../AudioData/splitLists/Gated{}_train.csv".format(listMod))
        dev = pd.read_csv("../../../AudioData/splitLists/Gated{}_dev.csv".format(listMod))
        test = pd.read_csv("../../../AudioData/splitLists/Gated{}_test.csv".format(listMod))

        for item, mod in zip([dev, train, test], ["dev", "train", "test"]):
            badList = assembleArray_Raw(listMod, item, measureList, frame_max, tokMax, mod, speaker=False, f0Normed=f0Normed, text=text)
            with open("BAD.txt", "a+") as f:
                for item in badList:
                    f.write("{}\n".format(item))


if __name__=="__main__":

    # measureList = ["ams", "f0", "hnr", "mfcc", "plp"]
    measureList = ["f0", "hnr"]
    speakerList = ["c", "d", "e", "j", "o", "s", "u"]
    listMod = "Pruned2"
    f0Normed = "m"
    text = "asr"

    main(listMod, speakerList, measureList, f0Normed=f0Normed, text=text, speakerSplit="dependent")