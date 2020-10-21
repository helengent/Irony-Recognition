#!/usr/bin/env python3

import os
import time
import pickle
from sd import sd
import numpy as np
import pandas as pd
from glob import glob
from speaker import Speaker
from extractor import Extractor

def make3Ddf(seqDict, listTup):

    print("Compiling 3D time-series data")

    #takes a tuple of lists, unpacks, and converts into a 3D dataframe

    f0List, mfccs, amsList, plpList = listTup

    fileNameList = seqDict["filename"]
    labList = seqDict["label"]
    speakerList = [filename[0] for filename in fileNameList]

    timeDict = dict()
    innerColNames = ["filename", "label", "speaker", "f0"]

    #Append all variable names to innerColNames
    for i in range(13):
        string = "mfcc" + str(i)
        innerColNames.append(string)
    for i in range(225):
        string = "ams" + str(i)
        innerColNames.append(string)
    for i in range(9):
        string = "plp" + str(i)
        innerColNames.append(string)

    print("Assembling 3D array")

    #Each element in timeDict will have a list of all variables for that element
    #IN THE ORDER THEY APPEAR IN innerColNames!!!
    for i in range(len(f0List[0])):
        timeString = "t" + str(i)

        timeDict[timeString] = [fileNameList, labList, speakerList]

        #All f0 measures at timestep i for all samples
        allTf0 = [f0[i] for f0 in f0List]
        timeDict[timeString].append(allTf0)

        #All mfcc coefficients
        for j in range(13):
            #All mfcc[j] coefficients at timestep i for all samples
            allTmfcc = [mfcc[i][j] for mfcc in mfccs]
            timeDict[timeString].append(allTmfcc)

        #All ams measures
        for j in range(225):
            #All ams[j] measures at timestep i for all samples
            allTams = [ams[j][i] for ams in amsList]
            timeDict[timeString].append(allTams)
        
        #All plp measures
        for j in range(9):
            #All plp[j] measures at timestep i for all samples
            allTplp = [plp[j][i] for plp in plpList]
            timeDict[timeString].append(allTplp)

    # timeList = list(timeDict.keys())

    # toArray = list()
    # for k in timeDict.keys():
    #     for item in timeDict[k]:
    #         toArray.append(item)

    # t = timeList * len(innerColNames)
    # t.sort()
    
    # C = np.array(toArray)

    # A = np.array(t)
    # B = np.array(innerColNames*len(timeList))

    # print("Saving 3D data")

    # threeDdf = pd.DataFrame(C.T, columns=pd.MultiIndex.from_tuples(zip(A, B)))
    # threeDdf.to_csv("../../FeaturalAnalysis/handExtracted/Data/CSVs/3Dsequential.csv", index=False)

    with open("../../FeaturalAnalysis/handExtracted/Data/Pickles/various3D/Pruned_timeDict.pkl", "wb") as f:
        pickle.dump(timeDict, f)
    with open("../../FeaturalAnalysis/handExtracted/Data/Pickles/various3D/Pruned_innerColNames.pkl", "wb") as f:
        pickle.dump(innerColNames, f)
    # with open("../../FeaturalAnalysis/handExtracted/Data/Pickles/various3D/A.pkl", "wb") as f:
    #     pickle.dump(A, f)
    # with open("../../FeaturalAnalysis/handExtracted/Data/Pickles/various3D/B.pkl", "wb") as f:
    #     pickle.dump(B, f)
    # with open("../../FeaturalAnalysis/handExtracted/Data/Pickles/various3D/C.pkl", "wb") as f:
    #     pickle.dump(C, f)
    # with open("../../FeaturalAnalysis/handExtracted/Data/Pickles/various3D/3Dsequential.pkl", "wb") as f:
    #     pickle.dump(threeDdf, f)

    print("3D processes complete")


def pruneAndSave():
    meanDur = sum(GLOBALDICT["duration"])/len(GLOBALDICT["duration"])
    sdDur = sd(GLOBALDICT["duration"], meanDur)

    # durUpperLim = meanDur + (2.5 * sdDur)
    # durLowerLim = 0.855 #hard-coded to ensure enough sequential samples.

    durUpperLim = 1000
    durLowerLim = 0 #These samples are already pruned for duration.

    global_df = pd.DataFrame(GLOBALDICT)
    global_df = global_df[global_df.duration < durUpperLim]
    global_df = global_df[global_df.duration >= durLowerLim]

    f0Pruned = list()
    mfccsPruned = list()
    amsPruned = list()
    plpPruned = list()

    print("Pruning sequential data to match pruned global data")

    newSequentialDict = {"filename": [], "speaker": [], "label": []}
    f0SequentialDict = {"filename": [], "speaker": [], "label": []}
    mfccSequentialDict = {"filename": [], "speaker": [], "label": []}
    amsSequentialDict = {"filename": [], "speaker": [], "label": []}
    plpSequentialDict = {"filename": [], "speaker": [], "label": []}

    seqDictList = [newSequentialDict, f0SequentialDict, mfccSequentialDict, amsSequentialDict, plpSequentialDict]

    for i, (contour, mfcc, ams, plp) in enumerate(zip(F0CONTOURS, MFCCS, AMSLIST, RASTAPLPLIST)):
        if SEQUENTIALDICT["filename"][i] in global_df.filename.tolist():
            f0Pruned.append(contour)
            mfccsPruned.append(mfcc)
            amsPruned.append(ams)
            plpPruned.append(plp)
            for dictionary in seqDictList:
                dictionary["filename"].append(SEQUENTIALDICT["filename"][i])
                dictionary["speaker"].append(SEQUENTIALDICT["filename"][i][0])
                dictionary["label"].append(SEQUENTIALDICT["label"][i])

    print("Padding sequential data to uniform length")

    longList = [np.max([len(contour) for contour in f0Pruned]), np.max([len(mfcc) for mfcc in mfccsPruned]), 
                np.max([np.shape(ams)[1] for ams in amsPruned]), np.max([np.shape(plp)[1] for plp in plpPruned])]
    longest = np.max(longList)

    #All f0 contours are 1xtime
    #All MFCC grids are timex13
    #All ams measures are 225xtime
    #All plp measures are 9xtime
    newMFCCsPruned, newAMSpruned, newPLPpruned = list(), list(), list()
    for contour, mfcc, ams, plp in zip(f0Pruned, mfccsPruned, amsPruned, plpPruned):
        while len(contour) < longest:
            contour.append(np.nan)
            
        while len(mfcc) < longest:
            padding = np.full((1,13), np.nan)
            mfcc = np.append(mfcc, padding, axis=0)
        newMFCCsPruned.append(mfcc)
        
        while np.shape(ams)[1] < longest:
            padding = np.full((225, 1), np.nan)
            ams = np.append(ams, padding, axis=1)
        newAMSpruned.append(ams)
        
        while np.shape(plp)[1] < longest:
            padding = np.full((9, 1), np.nan)
            plp = np.append(plp, padding, axis=1)
        newPLPpruned.append(plp)

    # This function makes it take FOREVER.
    # Only uncomment if you need the 3D dataframe
    make3Ddf(newSequentialDict, (f0Pruned, newMFCCsPruned, newAMSpruned, newPLPpruned))

    for i in range(longest):
        f0colName = "frame_" + str(i) + "_f0"
        newSequentialDict[f0colName] = [contour[i] for contour in f0Pruned]
        f0SequentialDict[f0colName] = [contour[i] for contour in f0Pruned]
        ms = [mfcc[i] for mfcc in newMFCCsPruned]
        for j in range(13):
            colName = "frame_" + str(i) + "_mfcc_" + str(j)
            newSequentialDict[colName] = [m[j] for m in ms]
            mfccSequentialDict[colName] = [m[j] for m in ms]
        for j in range(225):
            colName = "frame_" + str(i) + "_ams_" + str(j)
            allTams = [ams[j][i] for ams in newAMSpruned]
            newSequentialDict[colName] = allTams
            amsSequentialDict[colName] = allTams
        for j in range(9):
            colName = "frame_" + str(i) + "_plp_" + str(j)
            allTplp = [plp[j][i] for plp in newPLPpruned]
            newSequentialDict[colName] = allTplp
            plpSequentialDict[colName] = allTplp

    print("Saving global and 2D sequential dataframes")

    global_df.to_csv("../../FeaturalAnalysis/handExtracted/Data/CSVs/Pruned_global_measures.csv", index=False)
    with open("../../FeaturalAnalysis/handExtracted/Data/Pickles/Pruned_global.pkl", "wb") as f:
        pickle.dump(global_df, f)

    fileNames = ["all", "f0", "mfcc", "ams", "plp"]

    for d, f in zip(seqDictList, fileNames):

        print(f)

        sequential_df = pd.DataFrame(d)

        sequential_df.to_csv("../../FeaturalAnalysis/handExtracted/Data/CSVs/Pruned_10ms_{}_Seqmeasures.csv".format(f), index=False)

        with open("../../FeaturalAnalysis/handExtracted/Data/Pickles/Pruned_10ms_{}_Seqmeasures.pkl".format(f), "wb") as f:
            pickle.dump(sequential_df, f)


def extractVectors(wav, speakers):
    f = open("../Pruned_10ms_ReaperF0Results/{}.wav.f0.p".format(os.path.basename(wav).split(".")[0]), "r")
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
        ams = extractor.getAMS()
        plp = extractor.getPLP()

        fileID = wavfile.split("SPPep12_")[1]

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

        #Append ams to AMSLIST
        AMSLIST.append(ams)

        #Append plp to RASTAPLPLIST
        RASTAPLPLIST.append(plp) 


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

    global AMSLIST
    AMSLIST = list()

    global RASTAPLPLIST
    RASTAPLPLIST = list()

    speakers = makeSpeakerList()

    wavs = glob('../../AudioData/GatedPruned/*.wav')
    wavs.sort()

    for i, wav in enumerate(wavs):

        print("Working on file {} of {}".format(i, len(wavs)))
        extractVectors(wav, speakers)

    pruneAndSave()

if __name__ == "__main__":
    t0 = time.time()
    main()
    t1 = time.time()
    print("All processes completed in {} minutes".format((t1-t0)/60))
