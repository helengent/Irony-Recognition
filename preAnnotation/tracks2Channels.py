#!/usr/bin/env python3

import numpy as np
from glob import glob
from lib.WAVReader import WAVReader as WR
from lib.WAVWriter import WAVWriter as WW

def tr2ch(fileList, fileName):

    trackList = list()

    for f in fileList:
        readr = WR(f)
        data = readr.getData()
        trackList.append(data)

    print(list(set([len(item) for item in trackList])))
    
    print(list(set([np.shape(item) for item in trackList])))

    newAR = np.zeros((len(trackList[0]), len(trackList)))

    for i, item in enumerate(trackList):
        newAR[:, i] = item[:, 0]

    print(np.shape(newAR))

    writr = WW("../../preAnnotationAudio/{}_multiTrack.wav".format(fileName), newAR, fs = readr.getSamplingRate(), bits = readr.getBitsPerSample())
    writr.write()


if __name__=="__main__":

    sourceDir = "../../preAnnotationAudio/sad_boyz_27"

    fileList = glob("{}/*".format(sourceDir))

    print(fileList)

    tr2ch(fileList, sourceDir.split("/")[-1])