#!/usr/bin/env python3

import os
import time
import shutil
import pickle
from sd import sd
import numpy as np
import pandas as pd
from glob import glob
from speaker import Speaker
from extractor import Extractor


def extractVectors(wav, speakers, wavPath, winSize, saveIndv=False):

    wavfile = os.path.basename(wav).split(".")[0]

    #set speaker variable
    speaker = "NULL"
    for s in speakers:
        if wavfile.split("_")[1][0].upper() == s.getSpeaker():
            speaker = s

    #set irony variable
    if wavfile[-1] == "I":
        irony = "i"
    elif wavfile[-1] == "N":
        irony = "n"
    else:
        print("Oh no.")

    extractor = Extractor(wav, speaker, irony, winSize=int(winSize))
    f0, _ = extractor.getF0Contour()
    if len(list(set(f0))) == 1:
        with open("bad_files.txt", "w+") as f:
            f.write("Bad file: {}".format(wavfile))
        return
    else:
        mfccs = extractor.getMFCCs()
        dur = extractor.dur
        hnr = extractor.getHNR()
        hnrMean, hnrRange, hnrSD = extractor.getHNRstats()
        meanF0 = extractor.getMeanf0()
        F0sd = extractor.getSDf0()
        rangeF0 = extractor.getRangef0()
        medianF0 = extractor.getMedianf0()
        energyRange, energySD = extractor.getEnergyStats()
        apl, s2s, tp = extractor.getTimingStats()
        ams = extractor.getAMS()
        plp = extractor.getPLP()

        fileID = wavfile

        #Append to GLOBALDICT
        GLOBALDICT['filename'].append(fileID)
        GLOBALDICT['label'].append(irony)
        GLOBALDICT['speaker'].append(speaker.getSpeaker())
        GLOBALDICT['gender'].append(speaker.getGender())
        GLOBALDICT['duration'].append(dur)
        GLOBALDICT['f0globalMean'].append(meanF0)
        GLOBALDICT['f0globalRange'].append(rangeF0)
        GLOBALDICT['f0globalSD'].append(F0sd)
        GLOBALDICT["f0globalMedian"].append(medianF0)
        GLOBALDICT['avgPauseLength'].append(apl)
        GLOBALDICT['sound2silenceRatio'].append(s2s)
        GLOBALDICT['totalPauses'].append(tp)
        GLOBALDICT['hnrglobalMean'].append(hnrMean)
        GLOBALDICT['hnrglobalRange'].append(hnrRange)
        GLOBALDICT['hnrglobalSD'].append(hnrSD)
        GLOBALDICT['energyRange'].append(energyRange)
        GLOBALDICT['energySD'].append(energySD)

        #Append sequential information
        SEQUENTIALDICT['filename'].append(fileID)
        SEQUENTIALDICT['label'].append(irony)
        
        #Append f0 to F0CONTOURS
        F0CONTOURS.append(f0)

        #Append mfccs to MFCCS
        MFCCS.append(mfccs)

        #Append hnr to HNR
        HNR.append(hnr)

        #Append ams to AMSLIST
        AMSLIST.append(ams)

        #Append plp to RASTAPLPLIST
        RASTAPLPLIST.append(plp) 

        ##This code saves out individual csv files for each sequential measure and for the global measure vector for each .wav file
        if saveIndv == True:

            f0 = pd.DataFrame(np.array(f0))
            f0.to_csv("../../Data/AcousticData/f0/{}.csv".format(fileID), index=False)

            mfccs = pd.DataFrame(mfccs)
            mfccs.to_csv("../../Data/AcousticData/mfcc/{}.csv".format(fileID), index=False)

            hnr = pd.DataFrame(hnr)
            hnr.to_csv("../../Data/AcousticData/hnr/{}.csv".format(fileID), index=False)

            ams = pd.DataFrame(np.transpose(ams))
            ams.to_csv("../../Data/AcousticData/ams/{}.csv".format(fileID), index=False)

            plp = pd.DataFrame(np.transpose(plp))
            plp.to_csv("../../Data/AcousticData/plp/{}.csv".format(fileID), index=False)

            smolDict = {'f0globalMean': [meanF0], 'f0globalRange': [rangeF0], 'f0globalSD': [F0sd], 'f0globalMedian': [medianF0], 
                        'hnrglobalMean': [hnrMean], 'hnrglobalRange': [hnrRange], 'hnrglobalSD': [hnrSD], 
                        'energyRange': [energyRange], 'energySD': [energySD], 
                        'duration': [dur], 'avgPauseLength': [apl], 'sound2silenceRatio': [s2s], 'totalPauses': [tp]}
            smolDict = pd.DataFrame(smolDict)
            smolDict.to_csv("../../Data/AcousticData/globalVector/{}.csv".format(fileID), index=False)


def makeSpeakerList(s):
    speakers = list()
    for speaker in s:
        speakers.append(Speaker(speaker, "../../Data/AcousticData/SpeakerMetaData/{}_f0.txt".format(speaker)))
    return speakers
  

def main(wavPath, speakerList, output, winSize="10"):

    global GLOBALDICT
    GLOBALDICT = {"filename": [], "label": [], "speaker": [], "gender": [], "duration": [], "f0globalMean": [], "f0globalRange": [],
                  "f0globalSD": [], "f0globalMedian": [], "avgPauseLength": [], "sound2silenceRatio": [], "totalPauses": [], 
                  "hnrglobalMean": [], "hnrglobalRange": [], "hnrglobalSD": [], "energyRange": [], "energySD": []}

    global SEQUENTIALDICT
    SEQUENTIALDICT = {"filename": [], "label": []}

    global F0CONTOURS
    F0CONTOURS = list()

    global MFCCS
    MFCCS = list()

    global HNR
    HNR = list()

    global AMSLIST
    AMSLIST = list()

    global RASTAPLPLIST
    RASTAPLPLIST = list()

    speakers = makeSpeakerList(speakerList)

    wavs = glob('../AudioData/Gated{}/*.wav'.format(wavPath))
    wavs.sort()

    for i, wav in enumerate(wavs):

        print("Working on file {} of {}".format(i+1, len(wavs)))
        sI = "individual" in output
        extractVectors(wav, speakers, wavPath, winSize, saveIndv=sI)
    
    if "global" in output:
        global_df = pd.DataFrame(GLOBALDICT)
        global_df.to_csv("../../Data/AcousticData/{}_global_measures.csv".format(wavPath), index=False)

