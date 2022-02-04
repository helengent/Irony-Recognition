#/usr/bin/env python3

import os
import sys
import parselmouth
from glob import glob

sys.path.append(os.path.dirname(sys.path[1]))
from sd import sd


def upperLimit(vec):
    mean = sum(vec)/len(vec)
    sdev = sd(vec, mean)
    upper = mean + sdev * 2.5
    return upper


def lowerLimit(vec):
    mean = sum(vec)/len(vec)
    sdev = sd(vec, mean)
    lower = mean - sdev * 2.5
    return lower


def assembleVec(files):
    vec = []
    for thing in files:
        snd = parselmouth.Sound(thing)
        f0 = snd.to_pitch(time_step=0.005)
        fValues = f0.selected_array['frequency']
        for value in fValues:
            if value > 0:
                vec.append(value)
    return vec

def giveLowerLimit(vec):
    lowerLim = lowerLimit(vec)
    if lowerLim > 0:
        return lowerLim
    else:
        return 30


def giveUpperLimit(vec):
    upperLim = upperLimit(vec)
    return upperLim


def giveMean(vec):
    newVec = [item for item in vec if item != 0]
    if newVec == []:
        print("Empty vector!")
        return None
    else:
        mean = sum(newVec)/len(newVec)
        return mean


def giveSD(vec):
    mean = giveMean(vec)
    if mean == None:
        return None
    else:
        SD = sd(vec, mean)
        return SD


def main(wavPath, winSize, speakerList):
    f0Files = glob('../AudioData/Gated{}/*.wav'.format(wavPath))
    for speaker in speakerList:
        filesList = []
        indv = {}
        indv["speaker"] = speaker
        for item in f0Files:
            if os.path.basename(item).split("_")[1][0].upper() == speaker:
                filesList.append(item)
        vec = assembleVec(filesList)
        indv["upper"] = giveUpperLimit(vec)
        indv["lower"] = giveLowerLimit(vec)
        indv["mean"] = giveMean(vec)
        indv["sd"] = giveSD(vec)
        with open('../../Data/AcousticData/SpeakerMetaData/{}_f0.txt'.format(speaker), 'w') as f:
            for key in indv.keys():
                f.write("{}\t{}\n".format(key, str(indv[key])))
        f.close()
