#/usr/bin/env python3

import os
from sd import sd
#import parselmouth
from glob import glob

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
        f = open(thing, "r")
        text = f.read()
        f.close()
        text = text.split()
        for item in text:
            if item != "NaN":
                vec.append(float(item))
            else:
                vec.append(0.0)
    return vec

#New method to work with parseltongue and WAV files instead of REAPER text outputs
# def assembleVec(files):
#     vec = []
#     for thing in files:
#         snd = parselmouth.Sound(thing)
#         f0 = snd.to_pitch(time_step=0.005)
#         fValues = f0.selected_array['frequency']
#         for value in fValues:
#             if value > 0:
#                 vec.append(value)
#     return vec

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

if __name__=="__main__":
    f0Files = glob('../Pruned_10ms_ReaperF0Results/*.f0.p')
    speakerList = ["B", "G", "P", "R", "Y"]
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
        with open('../SpeakerF0Stats/{}.txt'.format(speaker), 'w') as f:
            for key in indv.keys():
                f.write("{}\t{}\n".format(key, str(indv[key])))
        f.close()
