#!/usr/bin/env python3

import sys
import numpy as np
import pandas as pd
from glob import glob
from itertools import combinations
from krippendorf_alpha import krippendorff_alpha
from statsmodels.stats.inter_rater import fleiss_kappa


def compareEach(df, annotators):

    rulings = df.Ruling.tolist()
    helen = df.Hresponse.tolist()

    for annotator in annotators:
        tmpList1 = list()
        tmpList2 = list()
        ann = df[annotator].tolist()
        for a, r, h in zip(ann, rulings, helen):
            tmpList1.append(a == r)
            tmpList2.append(a == h)
        df["{}_group_consistency".format(annotator)] = tmpList1
        df["{}_H_consistency".format(annotator)] = tmpList2
    
    return df


def compareWithH(df):
    h = glob("../../ANH/*.wav")
    hDict = dict()

    for f in h:
        hDict[f.split("_")[1].split("-")[0]] = f.split("-")[1].split(".")[0]
    
    hList = list()
    matchList = list()
    for i, row in df.iterrows():
        hList.append(hDict[row[0].split("_")[1]])
        matchList.append(row[-2] == hDict[row[0].split("_")[1]])

    df["Hresponse"] = hList
    df["Hmatch"] = matchList

    return df


def getReadyForKrippendorf(df):
    #the krippendorf_alpha function requires data be a list of dicts where each dict corresponds to a rater and each dict item to a sample and rating
    outList = list()
    for i in range(df.shape[1] - 1):
        outList.append(dict())
    for i, row in df.iterrows():
        j = 0
        for col, item in row.iteritems():
            if j > 0:
                if item == "N":
                    outList[j - 1][row[0]] = 2
                elif item == "I":
                    outList[j - 1][row[0]] = 1
                elif item == "P":
                    outList[j - 1][row[0]] = 2
                else:
                    raise Exception
            j += 1
    return outList


def getAverages(df):

    avgs = list()
    rulings = list()

    for i, row in df.iterrows():
        r = list(row[1:])
        ir = r.count("I")
        ni = r.count("N")

        if ir + ni != len(r):
            print(row)

        if ir > ni:
            avgs.append(ir/len(r))
            rulings.append("I")
        elif ni > ir:
            avgs.append(ni/len(r))
            rulings.append("N")
        else:
            avgs.append(ir/len(r))
            rulings.append("TIE")

    df["Ruling"] = rulings
    df["IRR"] = avgs

    return df


def parseFileNames(directory):

    fileList = glob("{}/*/*.wav".format(directory))
    fileList = ['/'.join(item.split("/")[-2:]) for item in fileList]
    annotators = list(set([item.split('/')[0] for item in fileList]))
    unqFiles = list(set([item.split("/")[1].split("-")[0] for item in fileList]))
    # falseNegs = ["c51", "d135", "c49", "d142", "c139", "d140", "d138", "d50", "d141"]
    # newF = [f for f in unqFiles if f not in falseNegs]
    newF = unqFiles[:]

    dfDict = {"fileName": newF}

    for a in annotators:
        dfDict[a] = list()
        for f in newF:
            r = ''
            for fl in fileList:
                if fl[:-6] == "{}/{}".format(a, f):
                    r = fl[-5].upper()

            if r == '':
                print(1)

            dfDict[a].append(r)

    iList, nList = list(), list()
    for i, f in enumerate(newF):
        inQ = [fl[-5].upper() for fl in fileList if fl.split("/")[1][:-6] == f]
        N = inQ.count("N")
        # N += inQ.count("P")
        I = inQ.count("I")
        iList.append(I)
        nList.append(N)

    df = pd.DataFrame(dfDict)
    Fdf = np.zeros((len(newF), 2))
    Fdf[:, 0] = iList
    Fdf[:, 1] = nList
    return(df, annotators, Fdf)


if __name__=="__main__":

    sourceDir = "../../Short_files_for_norming"
    df, annotators, Fdf = parseFileNames(sourceDir)

    kList = getReadyForKrippendorf(df)
    kPairs = combinations(kList, 2)
    annPairs = combinations(annotators, 2)

    df = getAverages(df)

    print("Inter-Rater Reliability: {}%".format(np.round((sum(list(df['IRR']))/len(df) * 100), 2)))

    print("Krippendorf's alpha: {}".format(krippendorff_alpha(kList)))

    # print("Fleiss' Kappa: {}".format(fleiss_kappa(Fdf, method="fleiss")))

    df = compareWithH(df)

    print("Consistency with control responses: {}%".format(np.round((df["Hmatch"].tolist().count(True)/len(df) * 100), 2)))


    kScoreList = list()
    annPairList = list()
    ##Pairwise scores

    current = ["ANhd", "ANpd"]

    for k, a in zip(kPairs, annPairs):
        kScore = np.round(krippendorff_alpha(k), 2)
        kScoreList.append(kScore)
        annPairList.append(a)

        if a[0] in current or a[1] in current:
            print("Krippendorff's alpha for annotator pair {}:\t{}".format(a, kScore))

    print("\n")
    print("Total pairs of annotators:\t{}".format(len(kScoreList)))
    print("Total pairs with scores of at least 0.67:\t{}".format(len([item for item in kScoreList if item >= 0.67])))
    print("Average pairwise Krippendorff's alpha:\t{}".format(sum(kScoreList)/len(kScoreList)))
    print("Maximum pairwise Krippendorff's alpha:\t{}\t with annotators {}".format(np.max(kScoreList), annPairList[np.argmax(kScoreList)]))
    print("Minimum pairwise Krippendorff's alpha:\t{}\t with annotators {}".format(np.min(kScoreList), annPairList[np.argmin(kScoreList)]))

    #Save dataframe out without all the consistency booleans for each individual annotator
    df.to_csv("NormScores.csv", index=False)

    df = compareEach(df, annotators)

    print()

    for a in annotators:
        print("Rater: {}\t Group Consistency: {}%\t Control Consistency: {}%".format(a, np.round((df["{}_group_consistency".format(a)].tolist().count(True)/len(df)) * 100, 2), np.round((df["{}_H_consistency".format(a)].tolist().count(True)/len(df)) * 100, 2)))

    # df.to_csv("NormScores.csv", index=False)


