#!/usr/bin/env python3

import sys
import pickle
import pandas as pd
from glob import glob

#Create consolidated dataframe that incorporates speaker information and irony label
def main():
    bigDF = pd.DataFrame()
    speakerList, labelList, genderList = list(), list(), list()

    for f in glob("baselineResults/*"):

        #Set speaker
        speaker = f[16]

        #Set speaker gender (numerified)
        if speaker == "b":
            speaker = 1
            gender = 1
        elif speaker == "y":
            speaker = 5
            gender = 1
        elif speaker == "g":
            speaker = 2
            gender = 2
        elif speaker == "p":
            speaker = 3
            gender = 2
        elif speaker == "r":
            speaker = 4
            gender = 3
        else:
            print("Helen, wtf did you do?")
            print(f, speaker)
            sys.exit()

        #Set irony label
        if f[-5] == "I":
            irony = 0
        else:
            irony = 1

        speakerList.append(speaker)
        labelList.append(irony)
        genderList.append(gender)

        row = pd.read_csv(f, sep=";", engine="python")
        bigDF = bigDF.append(row, ignore_index=True)
        print(bigDF.shape)

    bigDF['speaker'] = speakerList
    bigDF['gender'] = genderList
    bigDF['label'] = labelList
    try:
        bigDF.pop('name')
    except:
        pass

    print(bigDF.shape)
    
    with open("baseline_consolidated.pkl", "wb") as p:
        pickle.dump(bigDF, p, protocol=4)

if __name__=="__main__":
    main()