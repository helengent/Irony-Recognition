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
    sourceDir = "../../forIRR/ANfj_ANom"
    df, annotators, Fdf = parseFileNames(sourceDir)

    kList = getReadyForKrippendorf(df)
    kPairs = combinations(kList, 2)
    annPairs = combinations(annotators, 2)

    df = getAverages(df)

    print("Inter-Rater Reliability: {}%".format(np.round((sum(list(df['IRR']))/len(df) * 100), 2)))

    print("Krippendorf's alpha: {}".format(krippendorff_alpha(kList)))

    print("Fleiss' Kappa: {}".format(fleiss_kappa(Fdf, method="fleiss")))