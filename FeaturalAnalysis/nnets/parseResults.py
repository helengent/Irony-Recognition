#!/usr/bin/env python3

import pandas as pd


outResults = {"inputID": [], "measures": [], 
                "speakerDependentPrecision_N": [], "speakerDependentPrecision_I": [], 
                "speakerDependentRecall_N": [], "speakerDependentRecall_I": [], 
                "speakerDependentF1_N": [], "speakerDependentF1_I": [], 
                "speakerDependentAccuracy": [], 
                "oldSplitPrecision_N": [], "oldSplitPrecision_I": [], 
                "oldSplitRecall_N": [], "oldSplitRecall_I": [], 
                "oldSplitF1_N": [], "oldSplitF1_I": [], 
                "oldSplitAccuracy": [],
                "newSplitPrecision_N": [], "newSplitPrecision_I": [], 
                "newSplitRecall_N": [], "newSplitRecall_I": [], 
                "newSplitF1_N": [], "newSplitF1_I": [], 
                "newSplitAccuracy": []}


inputTypes = [(False, "percentChunks", False), (False, "rawSequential", False), 
                    (False, "percentChunks", True),
                    ("ComParE", "percentChunks", False), ("PCs", "percentChunks", False), ("PCs_feats", "percentChunks", False),
                    ("ComParE", "percentChunks", True), ("PCs", "percentChunks", True), ("PCs_feats", "percentChunks", True)]

measureLists = [["f0", "hnr"], ["f0", "mfcc"], ["f0", "plp"], ["f0", "hnr", "mfcc"], ["f0", "hnr", "plp"], 
                    ["hnr"], ["hnr", "mfcc"], ["hnr", "plp"], ["hnr", "mfcc", "plp"],  
                    ["mfcc"], ["mfcc", "plp"],
                    ["plp"], ["f0"]]

speakerSplits = ["speakerDependent", "oldSplit", "newSplit"]

count = 0
for inputType in inputTypes:

    globAcoustic, seqAcoustic, text = inputType

    prefix = "_"

    if globAcoustic:
        prefix = prefix + "{}_".format(globAcoustic)
    if seqAcoustic:
        prefix = prefix + "{}_".format(seqAcoustic)
    if text:
        prefix = prefix + "text_"

    for measureList in measureLists:

        outResults["inputID"].append(prefix[1:-1])
        outResults["measures"].append(", ".join(measureList))

        for speakerSplit in speakerSplits:

            if "Dependent" in speakerSplit:
                s = "dependent"
            else:
                s = "independent"

            try:
                with open("Results/{}-{}/performanceReport_Pruned3_speaker-{}{}.txt".format(speakerSplit, "-".join(measureList), s, prefix)) as f:
                    resultsLines = f.readlines()

                for l in resultsLines[-4:]:

                    items = l.split()
                    metric = items[0][:-1]
                    
                    if len(items) >= 3:
                        outResults["{}{}_N".format(speakerSplit, metric)].append(float(items[1][1:]))
                        outResults["{}{}_I".format(speakerSplit, metric)].append(float(items[2][:-1]))
                    else:
                        outResults["{}{}".format(speakerSplit, metric)].append(float(items[1]))
            except FileNotFoundError:
                print("BAD")
                for metric in ["Precision", "Recall", "F1"]:
                    outResults["{}{}_N".format(speakerSplit, metric)].append("NA")
                    outResults["{}{}_I".format(speakerSplit, metric)].append("NA")
                outResults["{}Accuracy".format(speakerSplit)].append("NA")

            count += 1


outDF = pd.DataFrame(outResults)
outDF.to_csv("seqFeatureComparison_ResultsSummary.csv", index=False)
