#!/usr/bin/env python3

import os
import sys
import numpy as np
import pandas as pd
from glob import glob
from extractor_textDependent_new import Extractor


def main(wavPath, tg_mod, saveWhole = False):

    fileList = glob("../AudioData/Gated{}/*.wav".format(wavPath))

    if saveWhole:
        lorgeDF = pd.DataFrame()
        labs, speakers, fNames = list(), list(), list()
        f0globalMeans, f0globalRanges, f0globalSDs, f0globalMedians, = list(), list(), list(), list()
        hnrglobalMeans, hnrglobalRanges, hnrglobalSDs = list(), list(), list()
        energyRanges, energySDs = list(), list()
        durs, sound2sil, totalPauses = list(), list(), list()


    for f in fileList:
        tg = "../../Data/TextData/{}_{}/{}.TextGrid".format(wavPath, tg_mod, os.path.basename(f).split(".")[0])

        print(os.path.basename(f).split(".")[0])

        E = Extractor(f, tg)

        m = E.returnMatrix()
        ml = E.matrix_labels

        df = pd.DataFrame([m], columns = ml)

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
                f0globalRanges.append(globAcoustic["f0globalRange"].item())
                f0globalSDs.append(globAcoustic["f0globalSD"].item())
                f0globalMedians.append(globAcoustic["f0globalMedian"].item())
                hnrglobalMeans.append(globAcoustic["hnrglobalMean"].item())
                hnrglobalRanges.append(globAcoustic["hnrglobalRange"].item())
                hnrglobalSDs.append(globAcoustic["hnrglobalSD"].item())
                energyRanges.append(globAcoustic["energyRange"].item())
                energySDs.append(globAcoustic["energySD"].item())
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
        lorgeDF["f0globalRange"] = f0globalRanges
        lorgeDF["f0globalSD"] = f0globalSDs
        lorgeDF["f0globalMedian"] = f0globalMedians
        lorgeDF["sound2silenceRatio"] = sound2sil
        lorgeDF["totalPauses"] = totalPauses
        lorgeDF["hnrglobalMean"] = hnrglobalMeans
        lorgeDF["hnrglobalRange"] = hnrglobalRanges
        lorgeDF["hnrglobalSD"] = hnrglobalSDs
        lorgeDF["energyRange"] = energyRanges
        lorgeDF["energySD"] = energySDs

        cols = lorgeDF.columns.tolist()
        cols = cols[-15:] + cols[:-15]
        lorgeDF = lorgeDF[cols]

        lorgeDF.to_csv("../../Data/AcousticData/text_feats/{}_{}_text_feats.csv".format(wavPath, tg_mod))


if __name__=="__main__":

    main("Pruned3", "asr", saveWhole=True)