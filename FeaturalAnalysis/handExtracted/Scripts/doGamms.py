#!/usr/bin/env python3

import numpy as np
import pandas as pd

def longify(df):
    newDict = {'filename': [], 'speaker': [], 'label': [], 'time': [], 'f0': [], 'mfcc0': [], 'mfcc1': [], 'mfcc2': [], 
                'mfcc3': [], 'mfcc4': [], 'mfcc5': [], 'mfcc6': [], 'mfcc7': [], 'mfcc8': [], 'mfcc9': [], 'mfcc10': [], 
                'mfcc11': [], 'mfcc12': []}
    for i, row in df.iterrows():
        time = 0
        n = 2
        while n < len(row):
            newDict['filename'].append(row[0])
            newDict['speaker'].append(row[0][0])
            newDict['label'].append(row[1])
            newDict['time'].append(time)

            newDict['f0'].append(row[n])
            n += 1
            newDict['mfcc0'].append(row[n])
            n += 1
            newDict['mfcc1'].append(row[n])
            n += 1
            newDict['mfcc2'].append(row[n])
            n += 1
            newDict['mfcc3'].append(row[n])
            n += 1
            newDict['mfcc4'].append(row[n])
            n += 1
            newDict['mfcc5'].append(row[n])
            n += 1
            newDict['mfcc6'].append(row[n])
            n += 1
            newDict['mfcc7'].append(row[n])
            n += 1
            newDict['mfcc8'].append(row[n])
            n += 1
            newDict['mfcc9'].append(row[n])
            n += 1
            newDict['mfcc10'].append(row[n])
            n += 1
            newDict['mfcc11'].append(row[n])
            n += 1
            newDict['mfcc12'].append(row[n])
            n += 1

            time += 1

    longdf = pd.DataFrame(newDict)

    longdf.to_csv("../Data/10ms_long_sequential.csv", index=False)

    return longdf

def balance(gloData):
    iData = gloData[gloData["label"]=="i"]
    nData = gloData[gloData["label"]=="n"]

    iMeanDur = sum(iData["duration"].tolist())/len(iData["duration"].tolist())
    iDurSD = np.std(iData["duration"].tolist())

    iDurLower = iMeanDur - (iDurSD * 2.5)
    iDurUpper = iMeanDur + (iDurSD * 2.5)

    nData = nData[nData["duration"] < iDurUpper]
    nData = nData[nData["duration"] > min(iData["duration"].tolist())]

    speakers = list(set(iData["speaker"].tolist()))
    minimum = np.min([iData[iData["speaker"]==s].shape[0] for s in speakers])

    iData = iData.groupby("speaker").sample(n=minimum, random_state=1)
    nData = nData.groupby("speaker").sample(n=minimum, random_state=1)

    gloBal = iData.append(nData)

    return gloBal

def interpolate(longData):
    
    fileNames = list(set(longData.filename.tolist()))
    longInterpolated = pd.DataFrame()

    for i, f in enumerate(fileNames):
        subset = longData[longData.filename == f]
        subset = subset.replace(0, np.nan)
        with open("../Data/TheBigDumb/f0/{}.txt".format(f), "w+") as w:
            w.writelines([str(p)+"\n" for p in subset.f0]) 
        with open("../Data/TheBigDumb/mfcc0/{}.txt".format(f), "w+") as w:
            w.writelines([str(p)+"\n" for p in subset.mfcc0])
        with open("../Data/TheBigDumb/mfcc1/{}.txt".format(f), "w+") as w:
            w.writelines([str(p)+"\n" for p in subset.mfcc1])
        with open("../Data/TheBigDumb/mfcc2/{}.txt".format(f), "w+") as w:
            w.writelines([str(p)+"\n" for p in subset.mfcc2])
        with open("../Data/TheBigDumb/mfcc3/{}.txt".format(f), "w+") as w:
            w.writelines([str(p)+"\n" for p in subset.mfcc3])
        with open("../Data/TheBigDumb/mfcc4/{}.txt".format(f), "w+") as w:
            w.writelines([str(p)+"\n" for p in subset.mfcc4])
        with open("../Data/TheBigDumb/mfcc5/{}.txt".format(f), "w+") as w:
            w.writelines([str(p)+"\n" for p in subset.mfcc5])
        with open("../Data/TheBigDumb/mfcc6/{}.txt".format(f), "w+") as w:
            w.writelines([str(p)+"\n" for p in subset.mfcc6])
        with open("../Data/TheBigDumb/mfcc7/{}.txt".format(f), "w+") as w:
            w.writelines([str(p)+"\n" for p in subset.mfcc7])
        with open("../Data/TheBigDumb/mfcc8/{}.txt".format(f), "w+") as w:
            w.writelines([str(p)+"\n" for p in subset.mfcc8])
        with open("../Data/TheBigDumb/mfcc9/{}.txt".format(f), "w+") as w:
            w.writelines([str(p)+"\n" for p in subset.mfcc9])
        with open("../Data/TheBigDumb/mfcc10/{}.txt".format(f), "w+") as w:
            w.writelines([str(p)+"\n" for p in subset.mfcc10])
        with open("../Data/TheBigDumb/mfcc11/{}.txt".format(f), "w+") as w:
            w.writelines([str(p)+"\n" for p in subset.mfcc11])
        with open("../Data/TheBigDumb/mfcc12/{}.txt".format(f), "w+") as w:
            w.writelines([str(p)+"\n" for p in subset.mfcc12])

        #linspace needs to be called for each filename since there's different lengths of data for each
        indices = np.linspace(0, subset.shape[0]-1, num=85)
        indices = [int(np.round(i)) for i in indices]

        newTime = np.linspace(0, 84, num=85)
        newTime = [int(t) for t in newTime]

        for j, i in enumerate(indices):
            if subset.f0.tolist()[i] == 0:
                peek = 1
                stop = False
                while (stop == False) and (peek < 6) and (i + peek < subset.shape[0]):
                    if subset.f0.tolist()[i+peek] != 0:
                        indices[j] = i+peek
                        stop = True
                    else:
                        peek += 1
        
        sbubset = subset.take(indices)
        sbubset.time = newTime
        sbubset = sbubset.replace(0, np.nan)

        sbubset = sbubset.interpolate()
        if sbubset.isnull().values.any():
            sbubset = sbubset.bfill()

        if sbubset.isnull().values.any():
            print("Okay. Go ahead and cry.")

        longInterpolated = longInterpolated.append(sbubset, ignore_index=True)
    
    return longInterpolated


def main(seqData, gloData):

    long_df = longify(seqData)
    # long_df = pd.read_csv("../Data/10ms_long_sequential.csv")
    gloBal = balance(gloData)

    longBal = long_df[long_df.filename.isin(gloBal.filename)]
    longBal = longBal.dropna()

    gloBal.to_csv("../Data/balancedGlobal.csv", index=False)
    longBal.to_csv("../Data/balancedLongSequential.csv", index=False)

    smoothed = interpolate(longBal)
    smoothed.to_csv("../Data/ready_for_gamms.csv", index=False)


if __name__=="__main__":
    globalDF = pd.read_csv("../Data/global_measures.csv")
    sequentialDF = pd.read_csv("../Data/10ms_sequential_measures.csv")
    main(sequentialDF, globalDF)