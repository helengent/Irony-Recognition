#!/usr/bin/env python3

from extractor import Extractor
from DEPRECATED_f0extractor import f0VecTime
from speaker import Speaker
from glob import glob
import pandas as pd
from matplotlib import pyplot as plt
import numpy as np
import sys

def plotIt(vec, othervec):
    plt.figure()
    plt.plot(vec)
    plt.plot(othervec)
    plt.show()

if __name__=="__main__":
     #initiate speakers list:
    B = Speaker("B", "../B.txt", gender="m")
    G = Speaker("G", "../G.txt", gender="f")
    P = Speaker("P", "../P.txt", gender="f")
    R = Speaker("R", "../R.txt", gender="nb")
    Y = Speaker("Y", "../Y.txt", gender="m")
    speakers = [B, G, P, R, Y]

    wavs = glob('../f0comparisonFile/duplicationWavs/*/*/*.wav')
    wavs.sort()
    f0Files = glob('../f0comparisonFile/duplicationWavs/*/*/*.f0.p')
    f0Files.sort()

    masterList = []
    #f0sequential = []

    Alldf = pd.DataFrame()
    Bdf = pd.DataFrame()
    Gdf = pd.DataFrame()
    Pdf = pd.DataFrame()
    Rdf = pd.DataFrame()
    Ydf = pd.DataFrame()

    for i in range(len(wavs)):

        print("Working on file {} of {}".format(i, len(wavs)))
        print(wavs[i])
        print(f0Files[i])

        #Get with Parseltongue-generated f0 contour
        wavfile = wavs[i]

        direc = wavfile[36]

        #set irony variable
        irony = wavfile[38]

        if irony == "i":
            k = wavfile[48].upper()
        else:
            k = wavfile[49].upper()
        print(k)

        #set speaker variable
        speaker = "NULL"
        for s in speakers:
            if k == s.getSpeaker():
                speaker = s

        extractor = Extractor(wavfile, speaker, irony)
        parseltongue = extractor.getF0Contour()
        parseltonguef0 = np.copy(parseltongue)
        for p, item in enumerate(parseltonguef0):
            if item == 0:
                parseltonguef0[p] = "NaN"

        #Get REaPER-generated f0 contour
        f = open(f0Files[i], "r")
        f0text = f.read()
        f.close()
        f0text = f0text.split()

        reaperf0 = f0VecTime(f0text, extractor.speaker.getLowerLimit(), extractor.speaker.getUpperLimit())
        for i, item in enumerate(reaperf0):
            if item == 0:
                reaperf0[i] = "NaN"

        for i in range(len(parseltonguef0)):
            if parseltonguef0[i] == 0:
                parseltonguef0[i] = "NaN"

        if len(reaperf0) != len(parseltonguef0):
            lengthList = [len(reaperf0), len(parseltonguef0)]
            targetLength = max(lengthList)
            while len(reaperf0) < targetLength:
                reaperf0.append("NaN")
            while len(parseltonguef0) < targetLength:
                parseltonguef0 = np.append(parseltonguef0, "NaN")

        else:
            targetLength = len(reaperf0)

        
        if extractor.irony == i:
            name = extractor.name[38:-4]
        else:
            name = extractor.name[41:-4]
        print(name)

        nameList = [name] * targetLength
        speakerList = [extractor.getSpeaker()] * targetLength
        ironyList = [extractor.irony] * targetLength

        p = {"Name": nameList, "Speaker": speakerList, "Irony": ironyList, "parseltonguef0": parseltonguef0, "reaperf0": reaperf0}
        
        #save everything out
        df = pd.DataFrame(p)

        if direc == "A":
            print("direc = " + direc)
            Alldf = Alldf.append(df, ignore_index=True)
        elif direc == "B":
            print("direc = " + direc)
            Bdf = Bdf.append(df, ignore_index=True)
        elif direc == "G":
            print("direc = " + direc)
            Gdf = Gdf.append(df, ignore_index=True)
        elif direc == "P":
            print("direc = " + direc)
            Pdf = Pdf.append(df, ignore_index=True)
        elif direc == "R":
            print("direc = " + direc)
            Rdf = Rdf.append(df, ignore_index=True)
        elif direc == "Y":
            print("direc = " + direc)
            Ydf = Ydf.append(df, ignore_index=True)
        else:
            print("Oh no... How did you do this?")


    Alldf.to_csv("../f0comparisonFile/duplicationAll.csv")
    Bdf.to_csv("../f0comparisonFile/duplicationB.csv")
    Gdf.to_csv("../f0comparisonFile/duplicationG.csv")
    Pdf.to_csv("../f0comparisonFile/duplicationP.csv")
    Rdf.to_csv("../f0comparisonFile/duplicationR.csv")
    Ydf.to_csv("../f0comparisonFile/duplicationY.csv")


