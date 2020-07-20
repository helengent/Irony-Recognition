#!/usr/bin/env python3

import sys
import pickle
import pandas as pd
from glob import glob

#Create consolidated dataframe that incorporates speaker information and irony label
def createConsolidated():
    bigDF = pd.DataFrame()
    speakerList, labelList, genderList = list(), list(), list()

    for f in glob("baselineResults/*"):

        #Set speaker
        speaker = f[16]

        #Set speaker gender
        if (speaker == "b") or (speaker == "y"):
            gender = "m"
        elif (speaker == "g") or (speaker == "p"):
            gender = "f"
        elif speaker == "r":
            gender = "nb"
        else:
            print("Helen, wtf did you do?")
            print(f, speaker)
            sys.exit()

        #Set irony label
        if f[-5] == "I":
            irony = "i"
        else:
            irony = "n"

        speakerList.append(speaker)
        labelList.append(irony)
        genderList.append(gender)

        row = pd.read_csv(f, sep=";", engine="python")
        bigDF = bigDF.append(row, ignore_index=True)
        print(bigDF.shape)

    bigDF['label'] = labelList
    bigDF['speaker'] = speakerList
    bigDF['gender'] = genderList

    print(bigDF.shape)
    
    with open("baseline_consolidated.pkl", "wb") as p:
        pickle.dump(bigDF, p)

if __name__ == "__main__":

    # createConsolidated()

    with open("baseline_consolidated.pkl", "rb") as p:
        bigDF = pd.read_pickle(p)

    for item in bigDF.columns:
        print(item)