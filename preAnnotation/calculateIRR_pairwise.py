#!/usr/bin/env python3

import sys
import numpy as np
import pandas as pd
from glob import glob
from itertools import combinations
from krippendorf_alpha import krippendorff_alpha
from statsmodels.stats.inter_rater import fleiss_kappa

from calculateIRR_all import parseFileNames, getReadyForKrippendorf, getAverages


if __name__=="__main__":

    annPairs = ["ANmy_ANri"]

    for pair in annPairs:

        sourceDir = "../../forIRR/{}".format(pair)
        df, annotators, Fdf = parseFileNames(sourceDir)

        kList = getReadyForKrippendorf(df)
        kPairs = combinations(kList, 2)
        annPairs = combinations(annotators, 2)

        df = getAverages(df)

        sharedSamples = len(glob("{}/{}/*.wav".format(sourceDir, pair.split("_")[0])))

        print(pair)

        print("Total Shared Samples: {}".format(sharedSamples))

        print("Inter-Rater Reliability: {}%".format(np.round((sum(list(df['IRR']))/len(df) * 100), 2)))

        print("Krippendorf's alpha: {}".format(krippendorff_alpha(kList)))

        print("Fleiss' Kappa: {}".format(fleiss_kappa(Fdf, method="fleiss")))

        print()