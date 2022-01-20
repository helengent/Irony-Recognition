#!/usr/bin/env python3

import sys
import numpy as np
import pandas as pd

sys.path.append("../../../AcousticFeatureExtraction")
from speaker import Speaker


def combine_formantTimes(df):

    colList = df.columns
    keepList = list()

    toAvg = dict()

    for c in colList:
        if "-" in c:
            phone = c.split("-")[0]
            formant = c.split("-")[1].split("_")[0]

            if phone not in toAvg:
                toAvg[phone] = dict()
            if formant not in toAvg[phone]:
                toAvg[phone][formant] = list()
            toAvg[phone][formant].append(c)
        else:
            keepList.append(c)

    newDF = df[keepList]
    for phone in toAvg.keys():
        for formant in toAvg[phone]:
            subset = df[toAvg[phone][formant]]
            newDF["{}_{}".format(phone, formant)] = subset.mean(axis = 1)

    return newDF


def normIt(value, segment, measure, speaker):

    formants = ["F0", "F1", "F2", "F3"]
    if measure in formants:
        avgs = list()
        for i in range(1, 6):
            avgs.append(float(speaker.seg_info[segment]["{}_{}".format(measure, i)]))
        avg = np.nanmean(avgs)
    else:
        avg = float(speaker.seg_info[segment][measure])
    newValue = (value - avg) /  avg
    return newValue


def normSegs(df):

    speakerList = list(set(df["speaker"].tolist()))
    speakers = dict()

    for speaker in speakerList:
        speakers[speaker] = Speaker(speaker, "/home/hmgent2/Data/AcousticData/SpeakerMetaData/{}_f0.txt".format(speaker.upper()), dur_txt="/home/hmgent2/Data/AcousticData/SpeakerMetaData/{}_avgDur.txt".format(speaker.upper()), seg_txt="/home/hmgent2/Data/AcousticData/SpeakerMetaData/{}_segmental.txt".format(speaker.upper()))

    for i, row in df.iterrows():
        counter = 0
        speaker = speakers[row["speaker"]]
        for j, item in row.iteritems():
            if counter > 20 and j != "Avg. Cons. Dur." and j != "Avg. Voca. Dur.":
                if "-" in j:
                    segment = j.split("-")[0]
                    measure = j.split("-")[1]
                else:
                    segment = j.split("_")[0]
                    measure = j.split("_")[1]
                row[j] = normIt(item, segment, measure, speaker)
            counter += 1
        df.loc[i] = row

    return df


def narrow(normed_df):

    colList = normed_df.columns
    sonorants = ['L', 'M', 'N', 'W', 'R', 'NG', 'Y']
    stops = ['K', 'P', 'T', 'D', 'B', 'G']
    fricatives = ['S', 'SH', 'Z', 'V', 'TH', 'F', 'DH', 'HH', 'CH', 'JH', 'ZH']

    stressedVowels = ['EY1', 'OW1', 'AH1', 'EH1', 'IH1', 'AA1', 'AY1', 'AE1', 'AO1', 'IY1', 'UH1', 
                        'OY1', 'UW1', 'AW1', 'ER1']
    unstressedVowels = ['EH2', 'AH0', 'OY2', 'ER0', 'AE2', 'IH0', 'IY2', 'IY0', 'OW2', 'IH2', 'EH0', 
                        'AO2', 'AA0', 'AA2', 'OW0', 'EY0', 'AE0', 'AW2', 'EY2', 'UW0', 'AH2', 'UW2', 
                        'AO0', 'AY2', 'UH2', 'AY0', 'ER2', 'OY0', 'UH0', 'AW0']

    subsets = [sonorants, stops, fricatives, stressedVowels, unstressedVowels]
    avgLists = {"sonorants": dict(), "stops": dict(), "fricatives": dict(), "stressedVowels": dict(), "unstressedVowels": dict()}

    keepList = list()

    for c in colList:
        phone = None
        if "-" in c:
            phone = c.split("-")[0]
            measure = c.split("-")[1]
        elif "_" in c:
            phone = c.split("_")[0]
            measure = c.split("_")[1]
        if phone:
            if phone in sonorants:
                if measure not in avgLists['sonorants']:
                    avgLists['sonorants'][measure] = list()
                avgLists['sonorants'][measure].append(c)
            if phone in stops:
                if measure not in avgLists['stops']:
                    avgLists['stops'][measure] = list()
                avgLists['stops'][measure].append(c)
            if phone in fricatives:
                if measure not in avgLists['fricatives']:
                    avgLists['fricatives'][measure] = list()
                avgLists['fricatives'][measure].append(c)
            if phone in stressedVowels:
                if measure not in avgLists['stressedVowels']:
                    avgLists['stressedVowels'][measure] = list()
                avgLists['stressedVowels'][measure].append(c)
            if phone in unstressedVowels:
                if measure not in avgLists['unstressedVowels']:
                    avgLists['unstressedVowels'][measure] = list()
                avgLists['unstressedVowels'][measure].append(c)
        else:
            keepList.append(c)
    
    newDF = normed_df[keepList]
    for key in avgLists.keys():
        for measure in avgLists[key]:
            subset = normed_df[avgLists[key][measure]]
            if measure == "VOT":
                measure = "duration"
            newDF["{}_{}".format(key, measure)] = subset.mean(axis = 1)

    return newDF


def main(df, prefix=""):

    df = pd.read_csv(df, index_col=0)
    df = df.reset_index()
    df = df.drop(columns=["index", "Sound Pressure Level", "Energy", "Root Mean Square", "ZCR"])

    #Features previously attested by other studies as being significantly different between ironic and non-ironic speech
    #   f0 - mean, range, sd (median wasn't included, but it'd be fine to do so)
    #   Energy - range and sd (mean energy can't really be compared across samples because of rms normalization and uncontrolled recording conditions)
    #   Timing - total utterance length, sound-to-silence ratio, total pauses, average pause length, syllables/second
    #   HNR - mean, range, sd

    #still need syllables per second
    prevAttested = df[["fileName", "speaker", "label", "f0globalMean", "f0globalRange", "f0globalSD", "f0globalMedian",
                        "rmsRange", "rmsSD", "hnrglobalMean", "hnrglobalRange", "hnrglobalSD", 
                        "duration", "sound2silenceRatio", "totalPauses", "Avg. Word Dur.", "Avg. Sil. Dur.", "SyllPerSecond"]]


    prevAttested.to_csv("../Data/{}prevAttested.csv".format(prefix), index=False)

    df = combine_formantTimes(df)

    df_normed = normSegs(df)

    df_normed.to_csv("../Data/{}all_Normed.csv".format(prefix), index=False)

    # df_normed = pd.read_csv("../Data/{}all_Normed.csv")

    df_narrowed = narrow(df_normed)

    df_narrowed.to_csv("../Data/{}all_narrowed.csv".format(prefix), index=False)


if __name__=="__main__":

    # input_df = "~/Data/AcousticData/text_feats/Pruned3_asr_text_feats.csv"
    input_df = "~/Data/AcousticData/text_feats/newTest_asr_text_feats.csv"
    prefix = "newTest_"

    main(input_df, prefix=prefix)