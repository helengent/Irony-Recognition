#!/usr/bin/env python3

import os
import sys
import time
import subprocess
import numpy as np
from preProcessing import preProcessAudio, asrFA, limitsUpperLower, getSpeakerDurationData

import extract
import extract_textDependent


def preProcess(wavPath, speakerList, winSize=10, needAMS=False, needPLP=False, haveManualT=False):

    if not os.path.isdir("../AudioData/Gated{}".format(wavPath)):
        # Downsample to 16000 Hz and convert to mono
        # Normalize rms and trim leading and trailing silence
        print("Beginning Initial Preprocessing")
        preProcessAudio.preMain(wavPath)

    # Generates ASR transcriptions for all audio files
    # Runs forced alignment for ASR transcriptions
    # If manual transcriptsions are available, runs forced alignment on them as well
    print("Performing ASR and Forced Alignment")
    asrFA.main(wavPath, haveManualT=haveManualT)

    # Finds and records upper and lower limits on F0 for each speaker, as well as mean and sd
    limitsUpperLower.main(wavPath, winSize, speakerList)

    # Generates and records duration-based metadata for each speaker
    if haveManualT == True:
        getSpeakerDurationData.speakerDurationData("../../Data/TextData/{}_manual".format(wavPath), speakerList)
    else:
        getSpeakerDurationData.speakerDurationData("../../Data/TextData/{}_asr".format(wavPath), speakerList)

    #TODO
    if needAMS == True:
        pass

    if needPLP == True:
        pass


def extractFeats(wavPath, speakerList, outputType, winSize=10, tg_mod="asr", saveWhole=False):

    #Extract acoustic features

    extract.main(wavPath, speakerList, outputType, winSize=winSize)

    extract_textDependent.main(wavPath, tg_mod, saveWhole=saveWhole)


if __name__=="__main__":

    # wavPath = "Pruned3"
    wavPath = "tmp"
    speakerList = ["C", "D", "E", "F", "H", "J", "K", "O", "Q", "S", "T", "U"]
    outputList = ['individual', 'global']

    tg_mod = "asr"
    saveWhole = True

    t0 = time.time()
    preProcess(wavPath, speakerList, haveManualT=False)
    # extractFeats(wavPath, speakerList, outputList, tg_mod=tg_mod, saveWhole=saveWhole)
    print("All processes completed in {} minutes".format(np.round((time.time() - t0) / 60), 2))