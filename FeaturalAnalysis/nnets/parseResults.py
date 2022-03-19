#!/usr/bin/env python3

from glob import glob
import sys
from numpy.testing._private.utils import measure
import pandas as pd
import numpy as np

#Dictionary to house baseline input modality combo results
#First 3 columns are defined for evaluation metrics
#Each model will get a column named with its input combo joined by "-"
resultsDict = {"eval_1": ["Text", "Time-Series Acoustic", "Utterance-Level Acoustic", "Architecture", 
                    "Average Precision", "Average Precision", "Average Precision", 
                    "Average Precision", "Average Precision", "Average Precision", 
                    "Average Recall", "Average Recall", "Average Recall", 
                    "Average Recall", "Average Recall", "Average Recall", 
                    "Average F-measure", "Average F-measure", "Average F-measure", 
                    "Average F-measure", "Average F-measure", "Average F-measure",
                    "Average Accuracy", "Average Accuracy", "Average Accuracy",
                    "Average AUC", "Average AUC", "Average AUC",
                    "AUC std", "AUC std", "AUC std", 
                    "Average EER", "Average EER", "Average EER", 
                    "EER std", "EER std", "EER std"], 
                "eval_2": ["Text", "Time-Series Acoustic", "Utterance-Level Acoustic", "Architecture", 
                    "SD5", "SD5", "LOSO", "LOSO", "SIB", "SIB", 
                    "SD5", "SD5", "LOSO", "LOSO", "SIB", "SIB", 
                    "SD5", "SD5", "LOSO", "LOSO", "SIB", "SIB", 
                    "SD5", "LOSO", "SIB", "SD5", "LOSO", "SIB", 
                    "SD5", "LOSO", "SIB", "SD5", "LOSO", "SIB", "SD5", "LOSO", "SIB"], 
                "eval_3": ["Text", "Time-Series Acoustic", "Utterance-Level Acoustic", "Architecture", 
                    "N", "I", "N", "I", "N", "I", 
                    "N", "I", "N", "I", "N", "I",
                    "N", "I", "N", "I", "N", "I",
                    "", "", "", "", "", "", 
                    "", "", "", "", "", "", "", "", ""]}

#Dictionary to house sequential acoustic feature comparison results
#First 3 columns are defined for evaluation metrics
#Each model will get a column named with its input combo and measureList joined by "-"
seqResultsDict = {"eval_1": ["Text", "Time-Series Acoustic", "Utterance-Level Acoustic", "Time-Series Acoustic Features", 
                    "Average Precision", "Average Precision", "Average Precision", 
                    "Average Precision", "Average Precision", "Average Precision", 
                    "Average Recall", "Average Recall", "Average Recall", 
                    "Average Recall", "Average Recall", "Average Recall", 
                    "Average F-measure", "Average F-measure", "Average F-measure", 
                    "Average F-measure", "Average F-measure", "Average F-measure",
                    "Average Accuracy", "Average Accuracy", "Average Accuracy",
                    "Average AUC", "Average AUC", "Average AUC",
                    "AUC std", "AUC std", "AUC std", 
                    "Average EER", "Average EER", "Average EER", 
                    "EER std", "EER std", "EER std"], 
                "eval_2": ["Text", "Time-Series Acoustic", "Utterance-Level Acoustic", "Time-Series Acoustic Features", 
                    "SD5", "SD5", "LOSO", "LOSO", "SIB", "SIB", 
                    "SD5", "SD5", "LOSO", "LOSO", "SIB", "SIB", 
                    "SD5", "SD5", "LOSO", "LOSO", "SIB", "SIB", 
                    "SD5", "LOSO", "SIB", "SD5", "LOSO", "SIB", 
                    "SD5", "LOSO", "SIB", "SD5", "LOSO", "SIB", "SD5", "LOSO", "SIB"], 
                "eval_3": ["Text", "Time-Series Acoustic", "Utterance-Level Acoustic", "Time-Series Acoustic Features", 
                    "N", "I", "N", "I", "N", "I", 
                    "N", "I", "N", "I", "N", "I",
                    "N", "I", "N", "I", "N", "I",
                    "", "", "", "", "", "", 
                    "", "", "", "", "", "", "", "", ""]}

#Dictionary to house differences in sequential acoustic feature comparison results
#First 3 columns are defined for evaluation metrics
#Each model will get a column named with its input combo and measureList joined by "-"
#Values will be calculated as (metric - fiveResults[inputID][metric])
seqDiffsDict = {"eval_1": ["Text", "Time-Series Acoustic", "Utterance-Level Acoustic", "Time-Series Acoustic Features", 
                    "Average Precision", "Average Precision", "Average Precision", 
                    "Average Precision", "Average Precision", "Average Precision", 
                    "Average Recall", "Average Recall", "Average Recall", 
                    "Average Recall", "Average Recall", "Average Recall", 
                    "Average F-measure", "Average F-measure", "Average F-measure", 
                    "Average F-measure", "Average F-measure", "Average F-measure",
                    "Average Accuracy", "Average Accuracy", "Average Accuracy",
                    "Average AUC", "Average AUC", "Average AUC",
                    "AUC std", "AUC std", "AUC std", 
                    "Average EER", "Average EER", "Average EER", 
                    "EER std", "EER std", "EER std"], 
                "eval_2": ["Text", "Time-Series Acoustic", "Utterance-Level Acoustic", "Time-Series Acoustic Features", 
                    "SD5", "SD5", "LOSO", "LOSO", "SIB", "SIB", 
                    "SD5", "SD5", "LOSO", "LOSO", "SIB", "SIB", 
                    "SD5", "SD5", "LOSO", "LOSO", "SIB", "SIB", 
                    "SD5", "LOSO", "SIB", "SD5", "LOSO", "SIB", 
                    "SD5", "LOSO", "SIB", "SD5", "LOSO", "SIB", "SD5", "LOSO", "SIB"], 
                "eval_3": ["Text", "Time-Series Acoustic", "Utterance-Level Acoustic", "Time-Series Acoustic Features", 
                    "N", "I", "N", "I", "N", "I", 
                    "N", "I", "N", "I", "N", "I",
                    "N", "I", "N", "I", "N", "I",
                    "", "", "", "", "", "", 
                    "", "", "", "", "", "", "", "", ""]}

#Dictionary to house PC global acoustic feature comparison results
#First 3 columns are defined for evaluation metrics
#Each model will get a column named with its input combo joined by "-"
PCResultsDict = {"eval_1": ["Text", "Time-Series Acoustic", "Utterance-Level Acoustic", "Num PCs", 
                    "Average Precision", "Average Precision", "Average Precision", 
                    "Average Precision", "Average Precision", "Average Precision", 
                    "Average Recall", "Average Recall", "Average Recall", 
                    "Average Recall", "Average Recall", "Average Recall", 
                    "Average F-measure", "Average F-measure", "Average F-measure", 
                    "Average F-measure", "Average F-measure", "Average F-measure",
                    "Average Accuracy", "Average Accuracy", "Average Accuracy",
                    "Average AUC", "Average AUC", "Average AUC",
                    "AUC std", "AUC std", "AUC std", 
                    "Average EER", "Average EER", "Average EER", 
                    "EER std", "EER std", "EER std"], 
                "eval_2": ["Text", "Time-Series Acoustic", "Utterance-Level Acoustic", "Num PCs", 
                    "SD5", "SD5", "LOSO", "LOSO", "SIB", "SIB", 
                    "SD5", "SD5", "LOSO", "LOSO", "SIB", "SIB", 
                    "SD5", "SD5", "LOSO", "LOSO", "SIB", "SIB", 
                    "SD5", "LOSO", "SIB", "SD5", "LOSO", "SIB", 
                    "SD5", "LOSO", "SIB", "SD5", "LOSO", "SIB", "SD5", "LOSO", "SIB"], 
                "eval_3": ["Text", "Time-Series Acoustic", "Utterance-Level Acoustic", "Num PCs", 
                    "N", "I", "N", "I", "N", "I", 
                    "N", "I", "N", "I", "N", "I",
                    "N", "I", "N", "I", "N", "I",
                    "", "", "", "", "", "", 
                    "", "", "", "", "", "", "", "", ""]}

#Dictionary to house differences in sequential acoustic feature comparison results
#First 3 columns are defined for evaluation metrics
#Each model will get a column named with its input combo joined by "-"
#Values will be calculated as (metric - threeResults[inputID][metric])
PCDiffsDict = {"eval_1": ["Text", "Time-Series Acoustic", "Utterance-Level Acoustic", "Num PCs", 
                    "Average Precision", "Average Precision", "Average Precision", 
                    "Average Precision", "Average Precision", "Average Precision", 
                    "Average Recall", "Average Recall", "Average Recall", 
                    "Average Recall", "Average Recall", "Average Recall", 
                    "Average F-measure", "Average F-measure", "Average F-measure", 
                    "Average F-measure", "Average F-measure", "Average F-measure",
                    "Average Accuracy", "Average Accuracy", "Average Accuracy",
                    "Average AUC", "Average AUC", "Average AUC",
                    "AUC std", "AUC std", "AUC std", 
                    "Average EER", "Average EER", "Average EER", 
                    "EER std", "EER std", "EER std"], 
                "eval_2": ["Text", "Time-Series Acoustic", "Utterance-Level Acoustic", "Num PCs", 
                    "SD5", "SD5", "LOSO", "LOSO", "SIB", "SIB", 
                    "SD5", "SD5", "LOSO", "LOSO", "SIB", "SIB", 
                    "SD5", "SD5", "LOSO", "LOSO", "SIB", "SIB", 
                    "SD5", "LOSO", "SIB", "SD5", "LOSO", "SIB", 
                    "SD5", "LOSO", "SIB", "SD5", "LOSO", "SIB", "SD5", "LOSO", "SIB"], 
                "eval_3": ["Text", "Time-Series Acoustic", "Utterance-Level Acoustic", "Num PCs", 
                    "N", "I", "N", "I", "N", "I", 
                    "N", "I", "N", "I", "N", "I",
                    "N", "I", "N", "I", "N", "I",
                    "", "", "", "", "", "", 
                    "", "", "", "", "", "", "", "", ""]}


#Dictionary to store results of models with all 5 time-series acoustic measures for each combination of input modalities
fiveResults = {"inputID": [], 
                "speakerDependentPrecision_N": [], "speakerDependentPrecision_I": [], 
                "oldSplitPrecision_N": [], "oldSplitPrecision_I": [], 
                "newSplitPrecision_N": [], "newSplitPrecision_I": [], 
                "speakerDependentRecall_N": [], "speakerDependentRecall_I": [], 
                "oldSplitRecall_N": [], "oldSplitRecall_I": [],
                "newSplitRecall_N": [], "newSplitRecall_I": [], 
                "speakerDependentF1_N": [], "speakerDependentF1_I": [], 
                "oldSplitF1_N": [], "oldSplitF1_I": [],
                "newSplitF1_N": [], "newSplitF1_I": [], 
                "speakerDependentAccuracy": [], "oldSplitAccuracy": [], "newSplitAccuracy": [], 
                "speakerDependentAUC": [], "oldSplitAUC": [], "newSplitAUC": [], 
                "speakerDependentAUCstd": [], "oldSplitAUCstd": [], "newSplitAUCstd": [], 
                "speakerDependentEER": [], "oldSplitEER": [], "newSplitEER": [], 
                "speakerDependentEERstd": [], "oldSplitEERstd": [], "newSplitEERstd": []}

#Dictionary to store results of models with 3 PCs as global acoustic measures for each combination of input modalities
threeResults = {"inputID": [], 
                "speakerDependentPrecision_N": [], "speakerDependentPrecision_I": [], 
                "oldSplitPrecision_N": [], "oldSplitPrecision_I": [], 
                "newSplitPrecision_N": [], "newSplitPrecision_I": [], 
                "speakerDependentRecall_N": [], "speakerDependentRecall_I": [], 
                "oldSplitRecall_N": [], "oldSplitRecall_I": [],
                "newSplitRecall_N": [], "newSplitRecall_I": [], 
                "speakerDependentF1_N": [], "speakerDependentF1_I": [], 
                "oldSplitF1_N": [], "oldSplitF1_I": [],
                "newSplitF1_N": [], "newSplitF1_I": [], 
                "speakerDependentAccuracy": [], "oldSplitAccuracy": [], "newSplitAccuracy": [], 
                "speakerDependentAUC": [], "oldSplitAUC": [], "newSplitAUC": [], 
                "speakerDependentAUCstd": [], "oldSplitAUCstd": [], "newSplitAUCstd": [], 
                "speakerDependentEER": [], "oldSplitEER": [], "newSplitEER": [], 
                "speakerDependentEERstd": [], "oldSplitEERstd": [], "newSplitEERstd": []}


# inputTypes = [(False, False, True), (False, "percentChunks", False), 
#                 ("ComParE", False, False), ("PCs", False, False), ("PCs_feats", False, False),
#                 (False, "percentChunks", True),
#                 ("ComParE", False, True), ("PCs", False, True), ("PCs_feats", False, True),
#                 ("ComParE", "percentChunks", False), ("PCs", "percentChunks", False), ("PCs_feats", "percentChunks", False),
#                 ("ComParE", "percentChunks", True), ("PCs", "percentChunks", True), ("PCs_feats", "percentChunks", True),
#                 ("2PCs", False, False), ("2PCs_feats", False, False), 
#                 ("6PCs", False, False), ("6PCs_feats", False, False), 
#                 ("30PCs", False, False), ("30PCs_feats", False, False), 
#                 ("2PCs", "percentChunks", True), ("2PCs_feats", "percentChunks", True), 
#                 ("6PCs", "percentChunks", True), ("6PCs_feats", "percentChunks", True), 
#                 ("30PCs", "percentChunks", True), ("30PCs_feats", "percentChunks", True)]

inputTypes = [("rawGlobal", False, False), 
                    ("rawGlobal", False, True), ("rawGlobal", "percentChunks", False), 
                    ("rawGlobal", "percentChunks", True)]

measureLists = [["f0", "hnr", "mfcc", "plp"], 
                # ["f0", "hnr", "mfcc"], ["f0", "hnr", "plp"], 
                # ["f0", "mfcc", "plp"], ["hnr", "mfcc", "plp"],
                # ["f0", "hnr"], ["f0", "mfcc"], ["f0", "plp"],
                # ["hnr", "mfcc"], ["hnr", "plp"], ["mfcc", "plp"],
                # ["f0"], ["hnr"], ["mfcc"], ["plp"]
                ]  


speakerSplits = ["speakerDependent", "oldSplit", "newSplit"]

for i, measureList in enumerate(measureLists):
    
    for inputType in inputTypes:

        globAcoustic, seqAcoustic, text = inputType

        prefix = "_"

        if globAcoustic:
            prefix = prefix + "{}_".format(globAcoustic)
        if seqAcoustic:
            prefix = prefix + "{}_".format(seqAcoustic)
        if text:
            prefix = prefix + "text_"

        intermediateResults = dict()

        for speakerSplit in speakerSplits:

            if "Dependent" in speakerSplit:
                s = "dependent"
            else:
                s = "independent"

            subDir = "{}-{}".format(speakerSplit, "-".join(measureList))

            try:
                with open("Results/{}/performanceReport_Pruned3_speaker-{}{}.txt".format(subDir, s, prefix)) as f:
                    resultsLines = f.readlines()

                for l in resultsLines[-8:]:

                    items = l.split()
                    metric = items[0][:-1]

                    if len(items) >= 3:
                        intermediateResults["{}{}_N".format(speakerSplit, metric)] = float(items[1][1:])
                        intermediateResults["{}{}_I".format(speakerSplit, metric)] = float(items[2][:-1])
                    else:
                        intermediateResults["{}{}".format(speakerSplit, metric)] = float(items[1])
                    
            except FileNotFoundError:
                print("BAD")
                for metric in ["Precision", "Recall", "F1"]:
                    intermediateResults["{}{}_N".format(speakerSplit, metric)] = "NA"
                    intermediateResults["{}{}_I".format(speakerSplit, metric)] = "NA"
                for metric in ["Accuracy", "AUC", "AUCstd", "EER", "EERstd"]:
                    intermediateResults["{}{}".format(speakerSplit, metric)] = "NA"

        #Once finished with all 3 speaker splits for a given inputType + measureList
        t, s, g = "", "", ""
        if text:
            t = "Text"
        if seqAcoustic:
            s = "10% Chunks"
        if globAcoustic:
            if globAcoustic == "PCs_feats":
                g = "PCs and top feats"
            else:
                g = globAcoustic

        orderedMetrics = [intermediateResults["speakerDependentPrecision_N"], intermediateResults["speakerDependentPrecision_I"], 
            intermediateResults["oldSplitPrecision_N"], intermediateResults["oldSplitPrecision_I"], 
            intermediateResults["newSplitPrecision_N"], intermediateResults["newSplitPrecision_I"], 
            intermediateResults["speakerDependentRecall_N"], intermediateResults["speakerDependentRecall_I"], 
            intermediateResults["oldSplitRecall_N"], intermediateResults["oldSplitRecall_I"], 
            intermediateResults["newSplitRecall_N"], intermediateResults["newSplitRecall_I"], 
            intermediateResults["speakerDependentF1_N"], intermediateResults["speakerDependentF1_I"], 
            intermediateResults["oldSplitF1_N"], intermediateResults["oldSplitF1_I"], 
            intermediateResults["newSplitF1_N"], intermediateResults["newSplitF1_I"], 
            intermediateResults["speakerDependentAccuracy"], intermediateResults["oldSplitAccuracy"], intermediateResults["newSplitAccuracy"], 
            intermediateResults["speakerDependentAUC"], intermediateResults["oldSplitAUC"], intermediateResults["newSplitAUC"],
            intermediateResults["speakerDependentAUCstd"], intermediateResults["oldSplitAUCstd"], intermediateResults["newSplitAUCstd"],
            intermediateResults["speakerDependentEER"], intermediateResults["oldSplitEER"], intermediateResults["newSplitEER"],
            intermediateResults["speakerDependentEERstd"], intermediateResults["oldSplitEERstd"], intermediateResults["newSplitEERstd"]]

        if len(measureList) == 4 and globAcoustic not in ["2PCs", "2PCs_feats", "6PCs", "6PCs_feats", "30PCs", "30PCs_feats"]:
            # Results for models with all 5 time-series acoustic features will finish first
            # Only these results will go in resultsDict, fiveResults, and threeResults

            if text and not seqAcoustic and not globAcoustic:
                arch = "CNN"
            elif not text and seqAcoustic and not globAcoustic:
                arch = "LSTM"
            elif not text and not seqAcoustic and globAcoustic:
                arch = "FFNN"
            elif text and seqAcoustic and not globAcoustic:
                arch = "CNN + LSTM"
            elif text and not seqAcoustic and globAcoustic:
                arch = "CNN + FFNN"
            elif not text and seqAcoustic and globAcoustic:
                arch = "LSTM + FFNN"
            elif text and seqAcoustic and globAcoustic:
                arch = "CNN + LSTM + FFNN"
            else:
                print("OH NO")
            
            resultsDict["{}-{}".format("-".join([str(item) for item in inputType]), "-".join([str(item) for item in measureList]))] = [t, s, g, arch]
            resultsDict["{}-{}".format("-".join([str(item) for item in inputType]), "-".join([str(item) for item in measureList]))].extend(orderedMetrics)

            fiveResults["inputID"].append("-".join([str(item) for item in inputType]))
            for m, key in zip(orderedMetrics, list(fiveResults.keys())[1:]):
                fiveResults[key].append(m)

            threeResults["inputID"].append("-".join([str(item) for item in inputType]))
            for m, key in zip(orderedMetrics, list(threeResults.keys())[1:]):
                threeResults[key].append(m)

        elif len(measureList) == 5 and globAcoustic in ["2PCs", "2PCs_feats", "6PCs", "6PCs_feats", "30PCs", "30PCs_feats"]:
            #PCResultsDict
            PCnum = globAcoustic[0]
            if "feats" in globAcoustic:
                g = "PCs and top feats"
            else:
                g = "PCs"
            PCResultsDict["-".join([str(item) for item in inputType])] = [t, s, g, PCnum]
            PCResultsDict["-".join([str(item) for item in inputType])].extend(orderedMetrics)

            #PCDiffsDict
            metDiffs = list()
            if "feats" in globAcoustic:
                generalIn = ("PCs_feats", seqAcoustic, text)
            else:
                generalIn = ("PCs", seqAcoustic, text)

            ind = threeResults["inputID"].index("-".join([str(item) for item in generalIn]))
            for m, key in zip(orderedMetrics, list(threeResults.keys())[1:]):
                try:
                    metDiffs.append(np.round(m - threeResults[key][ind], 3))
                except:
                    metDiffs.append("NA")
            PCDiffsDict["-".join([str(item) for item in inputType])] = [t, s, g, PCnum]
            PCDiffsDict["-".join([str(item) for item in inputType])].extend(metDiffs)

        else:
            #seqResultsDict
            seqFeats = ", ".join(measureList)
            seqResultsDict["{}-{}".format("-".join([str(item) for item in inputType]), "-".join(measureList))] = [t, s, g, seqFeats]
            seqResultsDict["{}-{}".format("-".join([str(item) for item in inputType]), "-".join(measureList))].extend(orderedMetrics)
            
            #seqDiffsDict
            metDiffs = list()
            ind = fiveResults["inputID"].index("-".join([str(item) for item in inputType]))
            for m, key in zip(orderedMetrics, list(fiveResults.keys())[1:]):
                try:
                    metDiffs.append(np.round(m - fiveResults[key][ind], 3))
                except:
                    metDiffs.append("NA")
            seqDiffsDict["{}-{}".format("-".join([str(item) for item in inputType]), "-".join(measureList))] = [t, s, g, seqFeats]
            seqDiffsDict["{}-{}".format("-".join([str(item) for item in inputType]), "-".join(measureList))].extend(metDiffs)


#ResultsDict
outResults = pd.DataFrame(resultsDict)
outResults.to_csv("ResultsTables/modalityCompare.csv", index=False)

#seqResultsDict
seqResults = pd.DataFrame(seqResultsDict)
seqResults.to_csv("ResultsTables/seqResults.csv", index=False)

#seqDiffsDict
seqDiffs = pd.DataFrame(seqDiffsDict)
seqDiffs.to_csv("ResultsTables/seqDiffs.csv", index=False)

#PCResultsDict
PCResults = pd.DataFrame(PCResultsDict)
PCResults.to_csv("ResultsTables/PCResults.csv", index=False)

#PCDiffsDict
PCDiffs = pd.DataFrame(PCDiffsDict)
PCDiffs.to_csv("ResultsTables/PCDiffs.csv", index=False)
