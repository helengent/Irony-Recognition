#!/usr/bin/env python3

import os
from glob import glob
from speaker import Speaker
from extractor import Extractor

def extractVector(wav, speakers):
    f = open("../ReaperF0Results/*/*/{}.f0.p".format(os.path.basename(wav).split(".")[0]), "r")
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
    speakers = makeSpeakerList()

    #wavs = glob('../../AudioData/SmolWaves/*/*/*.wav')
    wavs = glob('../../AudioData/TestWaves/*/*/*.wav')
    wavs.sort()

    for i, wav in enumerate(wavs):

        print("Working on file {} of {}".format(i, len(wavs)))
        extractVector(wav, speakers)

if __name__ == "__main__":
    main()