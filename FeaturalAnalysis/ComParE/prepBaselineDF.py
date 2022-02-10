#!/usr/bin/env python3

import os
import sys
import pickle
import pandas as pd
from glob import glob

#Create consolidated dataframe that incorporates speaker information and irony label
def main(data_dir):

    bigDF = pd.DataFrame()
    fileNameList, speakerList, labelList = list(), list(), list()

    out_dir = os.path.dirname(data_dir)
    mod = data_dir.split("/")[-1].strip("baseline")

    for f in glob("{}/*.csv".format(data_dir)):

        #Set speaker
        speaker = os.path.basename(f).split("_")[1][0]

        #Set irony label
        if f[-5] == "I":
            irony = "I"
        else:
            irony = "N"

        fileNameList.append(os.path.basename(f)[:-4])
        speakerList.append(speaker)
        labelList.append(irony)

        row = pd.read_csv(f)
        bigDF = bigDF.append(row, ignore_index=True)

    bigDF["fileName"] = fileNameList
    bigDF['speaker'] = speakerList
    bigDF['label'] = labelList
    try:
        bigDF.pop('name')
    except:
        pass

    print(bigDF.shape)
    print(bigDF)

    bigDF.to_csv("/home/hmgent2/Data/ModelInputs/ComParE/all_inputs.csv".format(out_dir))
    

if __name__=="__main__":

    data_dir = "../../../../Data/AcousticData/ComParE/baselinePruned3"

    main(data_dir)