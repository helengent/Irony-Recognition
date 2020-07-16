#!/usr/bin/env python3

from WAVReader import WAVReader as WR
from WAVWriter import WAVWriter as WW
import numpy as np
import os, shutil, sys
from glob import glob
    
def makeItMono(stereoInput, name, sampRate, bts):
    newAudio = np.zeros((len(stereoInput), 1))
    for i in range(len(stereoInput)):
        newAudio[i] = ((stereoInput[i][0]/2)+(stereoInput[i][1]/2))
    fileName = "../mono_test" + name[12:]
    writr = WW(fileName, newAudio)
    writr.write()

if __name__=="__main__":
    wavs = glob('../TestWaves/*/*/*.wav')
    for wav in wavs:

        readr = WR(wav)
        data = readr.getData()
        fs = readr.getSamplingRate()
        bits = readr.getBitsPerSample()
        makeItMono(data, wav, fs, bits)


