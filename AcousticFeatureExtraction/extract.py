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

# # This is baaaaaaad. Get rid of it!
# def pruneAndSave(wavPath, winSize, dirPath, prune=True):

#     newSequentialDict = {"filename": [], "speaker": [], "label": []}
#     f0SequentialDict = {"filename": [], "speaker": [], "label": []}
#     mfccSequentialDict = {"filename": [], "speaker": [], "label": []}
#     amsSequentialDict = {"filename": [], "speaker": [], "label": []}
#     plpSequentialDict = {"filename": [], "speaker": [], "label": []}
#     hnrSequentialDict = {"filename": [], "speaker": [], "label": []}

#     seqDictList = [newSequentialDict, f0SequentialDict, mfccSequentialDict, amsSequentialDict, plpSequentialDict, hnrSequentialDict]

#     if prune == True:
#         meanDur = sum(GLOBALDICT["duration"])/len(GLOBALDICT["duration"])
#         sdDur = sd(GLOBALDICT["duration"], meanDur)

#         durUpperLim = meanDur + (2.5 * sdDur)
#         durLowerLim = 0.755 #hard-coded to ensure enough sequential samples.

#         global_df = pd.DataFrame(GLOBALDICT)
#         global_df = global_df[global_df.duration < durUpperLim]
#         global_df = global_df[global_df.duration >= durLowerLim]

#         f0Pruned = list()
#         mfccsPruned = list()
#         amsPruned = list()
#         plpPruned = list()
#         hnrPruned = list()

#         print("Pruning sequential data to match pruned global data")

#         for i, (contour, mfcc, ams, plp, hnr) in enumerate(zip(F0CONTOURS, MFCCS, AMSLIST, RASTAPLPLIST, HNR)):
#             if SEQUENTIALDICT["filename"][i] in global_df.filename.tolist():
#                 f0Pruned.append(contour)
#                 mfccsPruned.append(mfcc)
#                 amsPruned.append(ams)
#                 plpPruned.append(plp)
#                 hnrPruned.append(hnr)
#                 for dictionary in seqDictList:
#                     dictionary["filename"].append(SEQUENTIALDICT["filename"][i])
#                     dictionary["speaker"].append(SEQUENTIALDICT["filename"][i][0])
#                     dictionary["label"].append(SEQUENTIALDICT["label"][i])

#     else:
#         global_df = pd.DataFrame(GLOBALDICT)
#         f0Pruned = F0CONTOURS[:]
#         mfccsPruned = MFCCS[:]
#         amsPruned = AMSLIST[:]
#         plpPruned = RASTAPLPLIST[:]
#         hnrPruned = HNR[:]

#     print("Padding sequential data to uniform length")

#     longList = [np.max([len(contour) for contour in f0Pruned]), np.max([len(mfcc) for mfcc in mfccsPruned]), 
#                 np.max([np.shape(ams)[1] for ams in amsPruned]), np.max([np.shape(plp)[1] for plp in plpPruned]), 
#                 np.max([len(hnr) for hnr in hnrPruned])]
#     longest = np.max(longList)

#     #All f0 contours are 1xtime
#     #All MFCC grids are timex13
#     #All ams measures are 375xtime
#     #All plp measures are 9xtime
#     #All hnr measures are 1xtime
#     newMFCCsPruned, newAMSpruned, newPLPpruned = list(), list(), list()
#     for contour, mfcc, ams, plp, hnr in zip(f0Pruned, mfccsPruned, amsPruned, plpPruned, hnrPruned):
#         while len(contour) < longest:
#             contour.append(np.nan)

#         while len(hnr) < longest:
#             hnr.append(np.nan)

#         while len(mfcc) < longest:
#             padding = np.full((1,13), np.nan)
#             mfcc = np.append(mfcc, padding, axis=0)
#         newMFCCsPruned.append(mfcc)
        
#         while np.shape(ams)[1] < longest:
#             padding = np.full((375, 1), np.nan)
#             ams = np.append(ams, padding, axis=1)
#         newAMSpruned.append(ams)
        
#         while np.shape(plp)[1] < longest:
#             padding = np.full((9, 1), np.nan)
#             plp = np.append(plp, padding, axis=1)
#         newPLPpruned.append(plp)

#     #TODO change column names once input format for RNN is clearer
#     for i in range(longest):
#         f0colName = "frame_" + str(i) + "_f0"
#         newSequentialDict[f0colName] = [contour[i] for contour in f0Pruned]
#         f0SequentialDict[f0colName] = [contour[i] for contour in f0Pruned]

#         hnrcolName = "frame_" + str(i) + "_hnr"
#         newSequentialDict[hnrcolName] = [hnr[i] for hnr in hnrPruned]
#         hnrSequentialDict[hnrcolName] = [hnr[i] for hnr in hnrPruned]

#         ms = [mfcc[i] for mfcc in newMFCCsPruned]
#         for j in range(13):
#             colName = "frame_" + str(i) + "_mfcc_" + str(j)
#             newSequentialDict[colName] = [m[j] for m in ms]
#             mfccSequentialDict[colName] = [m[j] for m in ms]
#         for j in range(375):
#             colName = "frame_" + str(i) + "_ams_" + str(j)
#             allTams = [ams[j][i] for ams in newAMSpruned]
#             newSequentialDict[colName] = allTams
#             amsSequentialDict[colName] = allTams
#         for j in range(9):
#             colName = "frame_" + str(i) + "_plp_" + str(j)
#             allTplp = [plp[j][i] for plp in newPLPpruned]
#             newSequentialDict[colName] = allTplp
#             plpSequentialDict[colName] = allTplp

#     print("Saving global and sequential dataframes")

#     global_df.to_csv("{}/global_measures.csv".format(dirPath), index=False)

#     fileNames = ["all", "f0", "mfcc", "ams", "plp", "hnr"]

#     for d, f in zip(seqDictList, fileNames):
#         print(f)
#         sequential_df = pd.DataFrame(d)
#         sequential_df.to_csv("{}/{}_Seqmeasures.csv".format(dirPath, f), index=False)


#This creates long data - needed for GAM analysis
def makeLongDFs(wavPath, winSize, dirPath):

    for i, measure in enumerate([F0CONTOURS, MFCCS, AMSLIST, RASTAPLPLIST, HNR]):
        longDict = {'filename': [], 'speaker': [], 'label': [], 'time': []}

        if i == 0:
            #deal with f0
            #F0CONTOURS is a list of lists
            longDict['f0'] = []
            for j, c in enumerate(measure):
                for h, dataPoint in enumerate(c):
                    longDict['filename'].append(GLOBALDICT['filename'][j])
                    longDict['speaker'].append(GLOBALDICT['speaker'][j].lower())
                    longDict['label'].append(GLOBALDICT['label'][j])
                    longDict['time'].append((h+1)/len(c))
                    longDict['f0'].append(dataPoint)
        elif i == 1:
            #deal with mfccs
            #MFCCS is an array of size (num_frames, 13)
            longDict['mfccNum'] = []
            longDict['mfcc'] = []
            for j, c in enumerate(measure):
                for h in range(np.shape(c)[0]):
                    for m in range(13):
                        longDict['filename'].append(GLOBALDICT['filename'][j])
                        longDict['speaker'].append(GLOBALDICT['speaker'][j].lower())
                        longDict['label'].append(GLOBALDICT['label'][j])
                        longDict['time'].append((h+1)/np.shape(c)[0])
                        longDict['mfccNum'].append(m+1)
                        longDict['mfcc'].append(c[h][m])
        elif i == 2:
            #deal with ams
            #AMSLIST is an array of size (375, num_frames)
            longDict['amsNum'] = []
            longDict['ams'] = []
            for j, c in enumerate(measure):
                for h in range(np.shape(c)[1]):
                    for m in range(375):
                        longDict['filename'].append(GLOBALDICT['filename'][j])
                        longDict['speaker'].append(GLOBALDICT['speaker'][j].lower())
                        longDict['label'].append(GLOBALDICT['label'][j])
                        longDict['time'].append((h+1)/np.shape(c)[1])
                        longDict['amsNum'].append(m+1)
                        longDict['ams'].append(c[m][h])
        elif i == 3:
            #deal with plp
            #RASTAPLPLIST is an array of size (9, num_frames)
            longDict['plpNum'] = []
            longDict['plp'] = []
            for j, c in enumerate(measure):
                for h in range(np.shape(c)[1]):
                    for m in range(9):
                        longDict['filename'].append(GLOBALDICT['filename'][j])
                        longDict['speaker'].append(GLOBALDICT['speaker'][j].lower())
                        longDict['label'].append(GLOBALDICT['label'][j])
                        longDict['time'].append((h+1)/np.shape(c)[1])
                        longDict['plpNum'].append(m+1)
                        longDict['plp'].append(c[m][h])
        elif i == 4:
            #deal with hnr
            #HNR is a list of lists
            longDict['hnr'] = []
            for j, c in enumerate(measure):
                for h, dataPoint in enumerate(c):
                    longDict['filename'].append(GLOBALDICT['filename'][j])
                    longDict['speaker'].append(GLOBALDICT['speaker'][j].lower())
                    longDict['label'].append(GLOBALDICT['label'][j])
                    longDict['time'].append((h+1)/len(c))
                    longDict['hnr'].append(dataPoint)
        
        longdf = pd.DataFrame(longDict)
        fileNames = ["f0", "mfcc", "ams", "plp", "hnr"]
        longdf.to_csv("{}/{}_long.csv".format(dirPath, fileNames[i]), index=False)


def extractVectors(wav, speakers, wavPath, winSize, saveIndv=False):
    # f = open("../ReaperTxtFiles/{}_{}ms_ReaperF0Results/{}.wav.f0.p".format(wavPath, winSize, os.path.basename(wav).split(".")[0]), "r")
    # f0text = f.read()
    # f.close()
    # f0text = f0text.split()

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

        fileID = wavfile

        #Append to GLOBALDICT
        GLOBALDICT['filename'].append(fileID)
        GLOBALDICT['label'].append(irony)
        GLOBALDICT['speaker'].append(speaker.getSpeaker())
        GLOBALDICT['gender'].append(speaker.getGender())
        GLOBALDICT['duration'].append(dur)
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

        #Append hnr to HNR
        HNR.append(hnr)

        #Append ams to AMSLIST
        AMSLIST.append(ams)

        #Append plp to RASTAPLPLIST
        RASTAPLPLIST.append(plp) 

        ##This code saves out individual csv files for each sequential measure and for the global measure vector for each .wav file
        if saveIndv == True:



            f0 = pd.DataFrame(np.array(f0))
            f0.to_csv("../AcousticData/f0/{}.csv".format(fileID), index=False)

            mfccs = pd.DataFrame(mfccs)
            mfccs.to_csv("../AcousticData/mfccs/{}.csv".format(fileID), index=False)

            hnr = pd.DataFrame(hnr)
            hnr.to_csv("../AcousticData/hnr/{}.csv".format(fileID), index=False)

            smolDict = {'duration': [dur], 'f0globalMean': [meanF0], 'f0globalSD': [F0sd], 
                        'avgPauseLength': [apl], 'sound2silenceRatio': [s2s], 'totalPauses': [tp]}
            smolDict = pd.DataFrame(smolDict)
            smolDict.to_csv("../AcousticData/globalVector/{}.csv".format(fileID), index=False)


def makeSpeakerList(s):
    speakers = list()
    for speaker in s:
        speakers.append(Speaker(speaker, "../AcousticData/SpeakerMetaData/{}_f0.txt".format(speaker)))
    return speakers
  

def main(wavPath, speakerList, output, winSize="10"):

    global GLOBALDICT
    GLOBALDICT = {"filename": [], "label": [], "speaker": [], "gender": [], "duration": [], "f0globalMean": [], 
                  "f0globalSD": [], "avgPauseLength": [], "sound2silenceRatio": [], "totalPauses": []}

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

    if "long" in output:
        dirPath = "../AcousticData/{}_{}ms".format(wavPath, winSize)
        if not os.path.isdir(dirPath):
            os.mkdir(dirPath)
        makeLongDFs(wavPath, winSize, dirPath)
    
    # if "sequential" in output:
    #     dirPath = "../AcousticData/{}_{}ms".format(wavPath, winSize)
    #     if not os.path.isdir(dirPath):
    #         os.mkdir(dirPath)
    #     pruneAndSave(wavPath, winSize, dirPath, prune=prune)
    
    if ("global" in output) and ("sequential" not in output):
        global_df = pd.DataFrame(GLOBALDICT)
        global_df.to_csv("../AcousticData/{}_global_measures.csv".format(wavPath), index=False)

if __name__ == "__main__":
    t0 = time.time()
    # speakers = ["B", "G", "P", "R", "Y"]

    speakers = ["C", "D", "E", "S", "U"]
    # outputList = ['global', 'sequential', 'long', 'individual']
    outputList = ['long']
    main("Pruned2", speakers, outputList)
    t1 = time.time()
    print("All processes completed in {} minutes".format((t1-t0)/60))
