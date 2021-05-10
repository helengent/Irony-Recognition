#!/usr/bin/env python3

import os
import sys
import time
import extract
import subprocess
import numpy as np
from preProcessing import preProcessAudio, asrFA, limitsUpperLower, getSpeakerDurationData

sys.path.append(os.path.join(os.path.dirname(sys.path[0], 'ASR')))
import asr

def main(wavPath, speakerList, outputType, winSize=10, prune=True, needReaper=False, needAMS=False, needPLP=False, needASR=False, needFA=False, haveManualT=False):

    # Downsample to 16000 Hz and convert to mono
    # Normalize rms and trim leading and trailing silence
    preProcessAudio.preMain(wavPath)

    # Generates ASR transcriptions for all audio files
    # Runs forced alignment for ASR transcriptions
    # If manual transcriptsions are available, runs forced alignment on them as well
    asrFA.main(wavPath, haveManualT=haveManualT)

    # Finds and records upper and lower limits on F0 for each speaker, as well as mean and sd
    limitsUpperLower.main(wavPath, winSize, speakerList)

    # Generates and records duration-based metadata for each speaker
    if haveManualT == True:
        getSpeakerDurationData.speakerDurationData("../TextData/{}_manual".format(wavPath), speakerList)
    else:
        getSpeakerDurationData.speakerDurationData("../TextData/{}_asr".format(wavPath), speakerList)

    #All Moved to preProcessing/preProcessAudio.py

    # if not os.path.isdir("../../AudioData/Gated{}".format(wavPath)):

    #     #convert stereo audio to mono
    #     #downsample to 16000 Hz
    #     bashCommand = "cd ../../AudioData/{}; mkdir downsampled; ../../AcousticFeatureExtraction/Scripts/monoDown.sh; mkdir ../temp{}; mv downsampled/* ../temp{}; rm -r downsampled; cd ../../AcousticFeatureExtraction/Scripts".format(wavPath, wavPath, wavPath)
    #     subprocess.run(bashCommand, shell=True)

    #     #preProcess.py normalizes rms to the average for all files (time-consuming), and trims leading and trailing silences
    #     preProcess.main(wavPath)

    #     #Get rid of temp folder. Files now live in Gated{wavPath}
    #     subprocess.run("rm -r ../../AudioData/temp{}".format(wavPath), shell=True)

    # All moved to preProcessing/asrFA.py
    #This is a good place to include ASR and forced alignment
    # if needASR == True:
    #     asr.main("../../AudioData/Gated{}".format(wavPath), "../../ASR/data/{}_asr".format(wavPath))

    # if needFA == True:
    #     bashCommand = "cd ../../ASR; ./run_Penn.sh ../AudioData/Gated{} data/{}_asr; cd ../AcousticFeatureExtraction/Scripts".format(wavPath, wavPath)
    #     subprocess.run(bashCommand, shell=True)

    #     if haveManualT == True:
    #         bashCommand = "cd ../../ASR; ./ASR/run_Penn.sh ../../AudioData/Gated{} ../../ASR/data/{}_manual; cd ../AcousticFeatureExtraction/Scripts".format(wavPath, wavPath)
    #         subprocess.run(bashCommand, shell=True)


    # if needReaper == True:
    #     bashCommand = "mkdir ../ReaperTxtFiles/{}_{}ms_ReaperF0Results; cd ../../AudioData/Gated{}; ../../AcousticFeatureExtraction/Scripts/REAPER/reaper.sh; ../../AcousticFeatureExtraction/Scripts/REAPER/formatReaperOutputs.sh; mv *.p ../../AcousticFeatureExtraction/ReaperTxtFiles/{}_{}ms_ReaperF0Results/; rm *.f0; cd ../../AcousticFeatureExtraction/Scripts".format(wavPath, winSize, wavPath, wavPath, winSize)
    #     subprocess.run(bashCommand, shell=True)

    #TODO
    if needAMS == True:
        pass

    if needPLP == True:
        pass

    # #Calculates speaker-dependent f0 statistics for outlier removal
    # limitsUpperLower.main(wavPath, winSize, speakerList)

    #Extract acoustic features

    extract.main(wavPath, speakerList, outputType, winSize=winSize, prune=prune)


if __name__=="__main__":
    wavPath = "ANH"
    # wavPath = "Pruned"
    speakerList = ["C", "D", "E"]
    # speakerList = ["B", "G", "P", "R", "Y"]
    # outputList = ['global', 'sequential', 'long', 'individual']
    # outputList = ['global', 'long', 'individual']
    outputList = ['individual']
    t0 = time.time()
    main(wavPath, speakerList, outputList, prune=False, needReaper=False, needASR=False, needFA=False, haveManualT=True)
    print("All processes completed in {} minutes".format(np.round((time.time() - t0) / 60), 2))