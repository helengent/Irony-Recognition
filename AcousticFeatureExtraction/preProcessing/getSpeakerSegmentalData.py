#!/usr/bin/env python3

import os
import numpy as np
import pandas as pd


def speakerSegmentalData(df):

    df = pd.read_csv(df, index_col=0)
    speakerList = list(set(df["speaker"].tolist()))

    for speaker in speakerList:

        wordDict = dict()
        phoneDict = dict()

        subset = df[df["speaker"] == speaker]

        with open("../../Data/AcousticData/SpeakerMetaData/{}_segmental.txt".format(speaker.upper()), "w") as f:
            f.write("speaker\t{}\n".format(speaker))

            for c in subset.columns[25:]:
                if c != "Avg. Voca. Dur." and c != "Avg. Cons. Dur.":
                    f.write("{}\t{}\n".format(c, np.nanmean(df[c])))
                   