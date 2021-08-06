#!/usr/bin/env python3

import os
import sys
import pickle
import pandas as pd
from glob import glob

#Create consolidated dataframe that incorporates speaker information and irony label
def main(data_dir):

    bigDF = pd.DataFrame()
    speakerList, labelList, genderList = list(), list(), list()
    genders = pd.read_csv("../../../../Data/AcousticData/SpeakerMetaData/speakersGenders.txt")
    speakerDict = dict()

    out_dir = os.path.dirname(data_dir)
    mod = data_dir.split("/")[-1].strip("baseline")

    speakerCounter = 0
    for f in glob("{}/*.csv".format(data_dir)):

        #Set speaker
        speaker = os.path.basename(f).split("_")[1][0]

        if speaker in speakerDict.keys():
            gender = speakerDict[speaker][1]
            speaker = speakerDict[speaker][0]
        else:
            gender = genders[genders["speaker"] == speaker.upper()]["gender"].tolist()[0]
            if gender == "m":
                gender = 0
            elif gender == "f":
                gender = 1
            else:
                gender = 2
            speakerDict[speaker] = (speakerCounter, gender)
            speaker = speakerCounter 
            speakerCounter += 1

        #Set irony label
        if f[-5] == "I":
            irony = 0
        else:
            irony = 1

        speakerList.append(speaker)
        labelList.append(irony)
        genderList.append(gender)

        row = pd.read_csv(f)
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
    
    with open("{}/baseline_consolidated_{}.pkl".format(out_dir, mod), "wb") as p:
        pickle.dump(bigDF, p, protocol=4)

if __name__=="__main__":

    data_dir = "../../../../Data/AcousticData/ComParE/baselinePruned2"

    main(data_dir)