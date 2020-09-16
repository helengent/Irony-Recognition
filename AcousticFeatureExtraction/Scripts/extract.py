#!/usr/bin/env python3

import os
import numpy as np
import pandas as pd
from glob import glob
from speaker import Speaker
from extractor import Extractor

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
    mfccs = extractor.getMFCCs()
    f0 = extractor.getF0Contour()
    silence = extractor.findSilences()
    dur = extractor.dur
    hnr = extractor.getHNR()

    #Append to SEQUENTIALDICT
    GLOBALDICT['speaker'].append(speaker.getSpeaker())
    GLOBALDICT['gender'].append(speaker.getGender())
    GLOBALDICT['duration'].append(dur)
    GLOBALDICT['hnr'].append(hnr)
    for i, mfcc in enumerate(mfccs):
        colName = "mfcc" + str(i)
        GLOBALDICT[colName].append(mfcc)
    
    #Append f0 to F0CONTOURS
    F0CONTOURS.append(f0)


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
    GLOBALDICT = {"speaker": [], "gender": [], "duration": [], "hnr": []}
    for i in range(13):
        colName = "mfcc" + str(i)
        GLOBALDICT[colName] = []

    global SEQUENTIALDICT
    SEQUENTIALDICT = dict()

    global F0CONTOURS
    F0CONTOURS = list()

    speakers = makeSpeakerList()

    #wavs = glob('../../AudioData/SmolWaves/*/*/*.wav')
    wavs = glob('../../AudioData/TestWaves/*/*/*.wav')
    wavs.sort()

    for i, wav in enumerate(wavs):

        print("Working on file {} of {}".format(i, len(wavs)))
        extractVectors(wav, speakers)

    longest = np.max([len(contour) for contour in F0CONTOURS])
    for contour in F0CONTOURS:
        while len(contour) < longest:
            contour.append(-1)

    for i in range(longest):
        colName = "frame_" + str(i) + "_f0"
        SEQUENTIALDICT[colName] = [contour[i] for contour in F0CONTOURS]

    global_df = pd.DataFrame(GLOBALDICT)
    sequential_df = pd.DataFrame(SEQUENTIALDICT)

    print(1)

if __name__ == "__main__":
    main()