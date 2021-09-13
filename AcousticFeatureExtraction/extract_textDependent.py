#!/usr/bin/env python3

import os
import sys
import numpy as np
import pandas as pd
from glob import glob
from extractor_textDependent import Extractor


def main(wavPath, tg_mod, saveWhole = False):

    fileList = glob("../AudioData/Gated{}/*.wav".format(wavPath))

    if saveWhole:
        lorgeDF = pd.DataFrame()
        labs, speakers, fNames = list(), list(), list()
        durs, f0globalMeans, f0globalSDs, sound2sil, totalPauses = list(), list(), list(), list(), list()

    for f in fileList:
        tg = "../../Data/TextData/{}_{}/{}.TextGrid".format(wavPath, tg_mod, os.path.basename(f).split(".")[0])

        print(os.path.basename(f).split(".")[0])

        E = Extractor(f, tg)

        m = E.returnMatrix()
        ml = E.matrix_labels

        df = pd.DataFrame([m], columns = ml)

        sys.exit()

        df.to_csv("../../Data/AcousticData/text_feats/{}/{}.csv".format(tg_mod, os.path.basename(f).split(".")[0]), index=False)

        if saveWhole:
            lorgeDF = lorgeDF.append(df)
            fNames.append(os.path.basename(f).split(".")[0])
            labs.append(os.path.basename(f).split(".")[0][-1])
            speakers.append(os.path.basename(f).split(".")[0].split("_")[1][0])

            try:
                globAcoustic = pd.read_csv("../../Data/AcousticData/globalVector/{}.csv".format(os.path.basename(f).split(".")[0]))
                durs.append(globAcoustic["duration"].item())
                f0globalMeans.append(globAcoustic["f0globalMean"].item())
                f0globalSDs.append(globAcoustic["f0globalSD"].item())
                sound2sil.append(globAcoustic["sound2silenceRatio"].item())
                totalPauses.append(globAcoustic["totalPauses"].item())
            except:
                durs.append(np.nan)
                f0globalMeans.append(np.nan)
                f0globalSDs.append(np.nan)
                sound2sil.append(np.nan)
                totalPauses.append(np.nan)


    if saveWhole:
        lorgeDF["fileName"] = fNames
        lorgeDF["speaker"] = speakers
        lorgeDF["label"] = labs
        lorgeDF["duration"] = durs
        lorgeDF["f0globalMean"] = f0globalMeans
        lorgeDF["f0globalSDs"] = f0globalSDs
        lorgeDF["sound2silenceRatio"] = sound2sil
        lorgeDF["totalPauses"] = totalPauses

        cols = lorgeDF.columns.tolist()
        cols = cols[-8:] + cols[:-8]
        lorgeDF = lorgeDF[cols]

        lorgeDF.to_csv("../../Data/AcousticData/text_feats/{}_{}_text_feats.csv".format(wavPath, tg_mod))


if __name__=="__main__":

    main("Pruned3", "asr", saveWhole=True)