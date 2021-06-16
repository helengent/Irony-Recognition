#!/usr/bin/env python3

import os
from glob import glob
from extractor_textDependent import Extractor


def main(wavPath, tg_mod):

    fileList = glob("../AudioData/Gated{}/*.wav".format(wavPath))

    for f in fileList:
        tg = "../../Data/TextData/data/{}_{}/{}.TextGrid".format(wavPath, tg_mod, os.path.basename(f).split(".")[0])

        E = Extractor(f, tg)

        m = E.returnMatrix()


if __name__=="__main__":
    main("Pruned2", "ASR")