#!/usr/bin/env python3

import subprocess
from preProcessing.ASR import asr


def main(wavPath, haveManualT=False):

    asr.main("../AudioData/Gated{}".format(wavPath), "../../Data/TextData/{}_asr".format(wavPath))

    bashCommand = "cd preProcessing/ASR; ./run_Penn.sh ../../../AudioData/Gated{} ../../../../Data/TextData/{}_asr; cd ../..".format(wavPath, wavPath)
    subprocess.run(bashCommand, shell=True)

    if haveManualT == True:
        bashCommand = "cd preProcessing/ASR; ./run_Penn.sh ../../../AudioData/Gated{} ../../../../Data/TextData/{}_manual; cd ../..".format(wavPath, wavPath)
        subprocess.run(bashCommand, shell=True)

if __name__=="__main__":

    main("ANH", haveManualT=True)