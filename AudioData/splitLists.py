#!/usr/bin/env python3

import os
import pandas as pd
from glob import glob
from sklearn.model_selection import train_test_split


def main(input_dir, speakerSplit="independent"):

    fileList = glob("{}/*.wav".format(input_dir))

    masterDict = {"id": [], "speaker": [], "label": []}

    for f in fileList:

        s = os.path.basename(f)

        masterDict["id"].append(s.split(".")[0])
        masterDict["speaker"].append(s.split("_")[1][0])
        masterDict["label"].append(s.split(".")[0][-1])

    masterList = pd.DataFrame(masterDict)

    if speakerSplit == "independent":

        speakers = list(set(masterList["speaker"].tolist()))

        for speaker in speakers:

            speakerList = masterList[masterList["speaker"] == speaker]

            train_df, dev_df = train_test_split(speakerList, test_size=0.1, shuffle=True, stratify=speakerList["label"], random_state=6)
            # train_df, test_df = train_test_split(train_df, test_size=0.1, shuffle=True, stratify=train_df["label"], random_state=6)

            train_df.to_csv("splitLists/{}_{}_train.csv".format(input_dir, speaker), index=False)
            dev_df.to_csv("splitLists/{}_{}_dev.csv".format(input_dir, speaker), index=False)
            # test_df.to_csv("splitLists/{}_{}_test.csv".format(input_dir, speaker), index=False)
            speakerList.to_csv("splitLists/{}_{}_ALL.csv".format(input_dir, speaker), index=False)

    else:
        train_df, dev_df = train_test_split(masterList, test_size=0.1, shuffle=True, stratify=masterList["label"], random_state=6)
        train_df, test_df = train_test_split(train_df, test_size=0.1, shuffle=True, stratify=train_df["label"], random_state=6)

        train_df.to_csv("splitLists/{}_train.csv".format(input_dir), index=False)
        dev_df.to_csv("splitLists/{}_dev.csv".format(input_dir), index=False)
        test_df.to_csv("splitLists/{}_test.csv".format(input_dir), index=False)

if __name__=="__main__":

    input_dir = "GatedAll"

    main(input_dir)