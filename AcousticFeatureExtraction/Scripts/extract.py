#!/usr/bin/env python3

import os
from sd import sd
import numpy as np
import pandas as pd
from glob import glob
from speaker import Speaker
from extractor import Extractor

def pruneAndSave():
    meanDur = sum(GLOBALDICT["duration"])/len(GLOBALDICT["duration"])
    sdDur = sd(GLOBALDICT["duration"], meanDur)

    durUpperLim = meanDur + (2.5 * sdDur)
    durLowerLim = meanDur - (2.5 * sdDur)

    global_df = pd.DataFrame(GLOBALDICT)
    global_df = global_df[global_df.duration < durUpperLim]

    f0Pruned = list()
    mfccsPruned = list()
    newSequentialDict = {"filename": [], "label": []}
    for i, (contour, mfcc) in enumerate(zip(F0CONTOURS, MFCCS)):
        if SEQUENTIALDICT["filename"][i] in global_df.filename.tolist():
            f0Pruned.append(contour)
            mfccsPruned.append(mfcc)
            newSequentialDict["filename"].append(SEQUENTIALDICT["filename"][i])
            newSequentialDict["label"].append(SEQUENTIALDICT["label"][i])

    longList = [np.max([len(contour) for contour in f0Pruned]), np.max([len(mfcc) for mfcc in mfccsPruned])]
    longest = np.max(longList)

    for contour, mfcc in zip(f0Pruned, mfccsPruned):
        while len(contour) < longest:
            contour.append(-1)
        while len(mfcc) < longest:
            padding = np.full((1,13), -1)
            mfcc = np.append(mfcc, padding, axis=0)

    for i in range(longest):
        f0colName = "frame_" + str(i) + "_f0"
        newSequentialDict[f0colName] = [contour[i] for contour in f0Pruned]
        ms = [mfcc[i] for mfcc in mfccsPruned]
        for j in range(13):
            colName = "frame_" + str(i) + "_mfcc_" + str(j)
            newSequentialDict[colName] = [m[j] for m in ms]

    # longest = np.max([len(contour) for contour in f0Pruned])
    # for contour in f0Pruned:
    #     while len(contour) < longest:
    #         contour.append(-1)

    # for i in range(longest):
    #     colName = "frame_" + str(i) + "_f0"
    #     newSequentialDict[colName] = [contour[i] for contour in f0Pruned]

    # Definitely need to break mfccs up into format like frame_i_mfcc_j
    # longest = np.max([len(mfcc) for mfcc in mfccsPruned])
    # for mfcc in mfccsPruned:
    #     while len(mfcc) < longest:
    #         mfcc.append([-1]*13)

    # for i in range(longest):
    #     ms = [mfcc[i] for mfcc in mfccsPruned]
    #     for j in range(13):
    #         colName = "frame_" + str(i) + "_mfcc_" + str(j)
    #         newSequentialDict[colName] = [m[j] for m in ms]

    sequential_df = pd.DataFrame(newSequentialDict)

    global_df.to_csv("../../FeaturalAnalysis/handExtracted/global_measures.csv")
    sequential_df.to_csv("../../FeaturalAnalysis/handExtracted/sequential_measures.csv")

def extractVectors(wav, speakers):
    f = open("../ReaperF0Results/{}.f0.p".format(os.path.basename(wav).split(".")[0]), "r")
    f0text = f.read()
    f.close()
    f0text = f0text.split()

    wavfile = os.path.basename(wav).split(".")[0]

    #set speaker variable
    speaker = "NULL"
    for s in speakers:
        if wavfile[8].upper() == s.getSpeaker():
            speaker = s

    #set irony variable
    if wavfile[-1] == "I":
        irony = "i"
    else:
        irony = "n"

    extractor = Extractor(wav, f0text, speaker, irony)
    f0 = extractor.getF0Contour()
    if len(list(set(f0))) == 1:
        print("Bad file: {}".format(wavfile))
        return
    else:
        mfccs = extractor.getMFCCs()
        dur = extractor.dur
        hnr = extractor.getHNR()
        meanF0 = extractor.getMeanf0()
        F0sd = extractor.getSDF0()
        apl, s2s, tp = extractor.getTimingStats()

        fileID = wavfile.split("_")[1]

        #Append to GLOBALDICT
        GLOBALDICT['filename'].append(fileID)
        GLOBALDICT['label'].append(irony)
        GLOBALDICT['speaker'].append(speaker.getSpeaker())
        GLOBALDICT['gender'].append(speaker.getGender())
        GLOBALDICT['duration'].append(dur)
        GLOBALDICT['hnr'].append(hnr)
        GLOBALDICT['f0globalMean'].append(meanF0)
        GLOBALDICT['f0globalSD'].append(F0sd)
        GLOBALDICT['avgPauseLength'].append(apl)
        GLOBALDICT['sound2silenceRatio'].append(s2s)
        GLOBALDICT['totalPauses'].append(tp)

        #Append sequential information
        SEQUENTIALDICT['filename'].append(fileID)
        SEQUENTIALDICT['label'].append(irony)
        
        #Append f0 to F0CONTOURS
        F0CONTOURS.append(f0)

        #Append mfccs to MFCCS
        MFCCS.append(mfccs)

def makeSpeakerList():
    #initiate speakers list:
    B = Speaker("B", "../SpeakerF0Stats/B.txt", gender="m")
    G = Speaker("G", "../SpeakerF0Stats/G.txt", gender="f")
    P = Speaker("P", "../SpeakerF0Stats/P.txt", gender="f")
    R = Speaker("R", "../SpeakerF0Stats/R.txt", gender="nb")
    Y = Speaker("Y", "../SpeakerF0Stats/Y.txt", gender="m")
    speakers = [B, G, P, R, Y]
    return speakers
    
def main():

    global GLOBALDICT
    GLOBALDICT = {"filename": [], "label": [], "speaker": [], "gender": [], "duration": [], "hnr": [], "f0globalMean": [], 
                  "f0globalSD": [], "avgPauseLength": [], "sound2silenceRatio": [], "totalPauses": []}

    global SEQUENTIALDICT
    SEQUENTIALDICT = {"filename": [], "label": []}

    global F0CONTOURS
    F0CONTOURS = list()

    global MFCCS
    MFCCS = list()

    speakers = makeSpeakerList()

    # wavs = glob('../../AudioData/SmolWaves/*/*/*.wav')
    # wavs = glob('../../AudioData/TestWaves/*/*/*.wav')
    wavs = glob('../../AudioData/GatedAll/*.wav')
    wavs.sort()

    for i, wav in enumerate(wavs):

        print("Working on file {} of {}".format(i, len(wavs)))
        extractVectors(wav, speakers)
    
    pruneAndSave()

if __name__ == "__main__":
    main()