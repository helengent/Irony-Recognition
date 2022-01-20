#!/usr/bin/env python3

import os
import sys
import keras
import pickle
import numpy as np
from numpy.testing._private.utils import measure
import pandas as pd
from glob import glob
from keras import layers
from keras import models
from matplotlib import pyplot as plt
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.preprocessing.text import Tokenizer
from sklearn.metrics import precision_recall_fscore_support, confusion_matrix, accuracy_score, roc_auc_score, roc_curve, ConfusionMatrixDisplay
from sklearn.model_selection import train_test_split, StratifiedKFold


sys.path.append("../../Models")
from LSTM_acousticOnly import acousticOnlyLSTM
from FeedForward import FeedForwardNN
from textOnly import textOnlyNN
from LSTM_withText import acousticTextLSTM
from LSTM_CNN_withText import acousticTextLSTM_CNN
from globAcousticCNN import globAcousticCNN
from LSTM_FFNN_CNN_withText import acousticTextLSTM_CNN_FFNN
from FFNN_CNN_withText import acousticTextCNN_FFNN
from LSTM_FFNN import acousticLSTM_FFNN

sys.path.append("../../AcousticFeatureExtraction")
from speaker import Speaker
from sd import sd


class ModelTester:

    def __init__(self, inputType, dataPath, counter, measureList, modelPath, speakerSplit="independent", splitName="newSplit", fileMod="Pruned3"):
        self.glob_acoustic, self.seq_acoustic, self.text = inputType 
        self.dataPath = dataPath
        self.speakerSplit = speakerSplit
        self.splitName = splitName
        self.counter = counter
        self.measureList = measureList
        self.fileMod = fileMod

        self.prefix = "speaker-{}_".format(self.speakerSplit)
        if self.glob_acoustic:
            self.prefix = self.prefix + "{}_".format(inputType[0])
        if self.seq_acoustic:
            self.prefix = self.prefix + "{}_".format(inputType[1])
        if self.text:
            self.prefix = self.prefix + "text_"

        self.model_path = modelPath
        self.model = models.load_model(self.model_path)

        self.model_inputs = self.loadData()


    def performanceReport(self):

        with open("Inference/performanceReport_{}_{}{}.txt".format(self.fileMod, self.prefix, self.counter), "w") as f:

            f.write("Test performance\n")
            f.write("Precision:\t{}\n".format(self.test_performance[0][0]))
            f.write("Recall:\t{}\n".format(self.test_performance[0][1]))
            f.write("F1:\t{}\n".format(self.test_performance[0][2]))
            f.write("Accuracy:\t{}\n".format(self.test_performance[1]))


    def getROCstats(self, trueLabs, Ipreds):

        fpr, tpr, thresholds = roc_curve(trueLabs, Ipreds, pos_label=1)
        auc = roc_auc_score(trueLabs, Ipreds)

        fnr = 1 - tpr
        eer_threshold = thresholds[np.nanargmin(np.absolute((fnr - fpr)))]
        eer = fpr[np.nanargmin(np.absolute((fnr - fpr)))]

        return (fpr, tpr, thresholds, auc, eer)


    def ROCcurve(self):

        mean_fpr = np.linspace(0, 1, 100)
        fig, ax = plt.subplots()

        fpr, tpr, thresholds, auc, eer = self.rocStats
        tpr = np.interp(mean_fpr, fpr, tpr)
        tpr[0] = 0.0

        ax.plot([0, 1], [0, 1], linestyle='--', lw=2, color="r", label="Chance", alpha=0.8)
        ax.plot(mean_fpr, tpr, color="b", 
                label = "ROC AUC: {}\nEER: {}".format(np.round(auc, 2), np.round(eer, 2)), 
                lw=2, alpha=0.8)

        ax.set(xlim=[-0.05, 1.05], ylim=[-0.05, 1.05], title="ROC Curve")
        ax.legend(loc="lower right")
        plt.savefig("Inference/ROC_{}{}.png".format(self.prefix, self.counter))


    def confusionMatrix(self, predictions):

        cm = confusion_matrix(self.labels, predictions, normalize="true")
        test_cm = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["N", "I"])
        # fig = plt.figure()
        # plt.matshow(cm)
        # plt.title('Confusion Matrix for Best Model')
        # plt.colorbar()
        test_cm.plot(cmap=plt.cm.Blues)
        plt.ylabel('True Label')
        plt.xlabel('Predicated Label')
        plt.savefig("Inference/CM_{}{}.png".format(self.prefix, self.counter))


    def outputPredictions(self, predictions):

        transformedPreds, transformedLabs = list(), list()
        for item, lab in zip(predictions, self.labels):
            if item == 1.0:
                transformedPreds.append("I")
            elif item == 0.0:
                transformedPreds.append("N")
            else:
                print("Oh no!")
                raise Exception

            if lab == 1.0:
                transformedLabs.append("I")
            elif lab == 0.0:
                transformedLabs.append("N")
            else:
                print("Oh no!")
                raise Exception

        self.meta["prediction"] = transformedPreds
        self.meta["match"] = [item1 == item2 for item1, item2 in zip(transformedLabs, transformedPreds)]

        self.meta.to_csv("Inference/predictions_{}{}".format(self.prefix, self.counter), index = False)


    def loadData(self):

        inputs = list()

        self.meta = pd.read_csv("{}/speaker-{}_test_{}.meta".format(self.dataPath, self.speakerSplit, self.counter))
        textList = [open("/home/hmgent2/Data/TextData/{}_asr/{}.txt".format(self.fileMod, f)).read().strip("\n") for f in self.meta["fileName"].tolist()]
        self.meta["text"] = textList

        if self.text:
            self.text_test = np.load("{}/Text/{}/speaker-{}_test-{}_tokenized.npy".format(self.dataPath, self.splitName, self.speakerSplit, self.counter))
            self.labels = np.load("{}/Text/{}/speaker-{}_test-{}_labels.npy".format(self.dataPath, self.splitName, self.speakerSplit, self.counter))
            inputs.append(self.text_test)

        if self.seq_acoustic:
            self.seq_acoustic_test = np.load("{}/{}/{}-{}/speaker-{}_test-{}_acoustic.npy".format(self.dataPath, self.seq_acoustic, self.splitName, "-".join(self.measureList), self.speakerSplit, self.counter))
            self.labels = np.load("{}/{}/{}-{}/speaker-{}_test-{}_labels.npy".format(self.dataPath, self.seq_acoustic, self.splitName, "-".join(self.measureList), self.speakerSplit, self.counter))
            inputs.append(self.seq_acoustic_test)

        if self.glob_acoustic:
            self.glob_acoustic_test = np.load("{}/{}/{}-{}/speaker-{}_test-{}_acoustic.npy".format(self.dataPath, self.glob_acoustic, self.splitName, "-".join(self.measureList), self.speakerSplit, self.counter))
            self.labels = np.load("{}/{}/{}-{}/speaker-{}_test-{}_labels.npy".format(self.dataPath, self.glob_acoustic, self.splitName, "-".join(self.measureList), self.speakerSplit, self.counter))
            inputs.append(self.glob_acoustic_test)

        return inputs


    def testModel(self):

        test_preds = self.model.predict(self.model_inputs)

        test_stats = precision_recall_fscore_support(self.labels, test_preds.argmax(axis=1))
        test_accuracy = accuracy_score(self.labels, test_preds.argmax(axis=1))

        self.test_performance = (test_stats, test_accuracy)

        self.rocStats = self.getROCstats(self.labels, test_preds[:, 1])

        self.performanceReport()
        self.ROCcurve()
        self.confusionMatrix(test_preds.argmax(axis=1))
        self.outputPredictions(test_preds.argmax(axis=1))


if __name__=="__main__":

    # fileMod = "newTest"
    # dataPath = "/home/hmgent2/Data/newTest_ModelInputs"

    fileMod = "Pruned3"
    dataPath = "/home/hmgent2/Data/ModelInputs"
    speakerSplits = ["dependent", "independent"]

    #Make list of tuples with combinations of input types
    # glob_acoustic = [False, "PCs", "rawGlobal"]
    # seq_acoustic = [False, "percentChunks", "rawSequential"]
    # text = [False, True]

    # inputTypes = [(False, False, True), (False, "percentChunks", False), (False, "rawSequential", False), 
    #                 ("ComParE", False, False), ("PCs", False, False), ("PCs_feats", False, False), ("rawGlobal", False, False),
    #                 (False, "percentChunks", True), (False, "rawSequential", True), 
    #                 ("ComParE", False, True), ("PCs", False, True), ("PCs_feats", False, True), ("rawGlobal", False, True), 
    #                 ("ComParE", "percentChunks", False), ("PCs", "percentChunks", False), ("PCs_feats", "percentChunks", False),("rawGlobal", "percentChunks", False), 
    #                 ("ComParE", "rawSequential", False), ("PCs", "rawSequential", False), ("PCs_feats", "rawSequential", False),("rawGlobal", "rawSequential", False), 
    #                 ("ComParE", "percentChunks", True), ("PCs", "percentChunks", True), ("PCs_feats", "percentChunks", True),("rawGlobal", "percentChunks", True),
    #                 ("ComParE", "rawSequential", False), ("PCs", "rawSequential", True), ("PCs_feats", "rawSequential", True),("rawGlobal", "rawSequential", True)]

    inputTypes = [("PCs", "percentChunks", True), ("2PCs_feats", "percentChunks", True)]
    measureLists = [["f0", "hnr", "mfcc"], ["f0", "mfcc", "plp"]]

    for inputType, speakerSplit, measureList in zip(inputTypes, speakerSplits, measureLists):

        if speakerSplit == "dependent":
            counter = 3
            splitName = "speakerDependent"
            modelPath = "Checkpoints/speakerDependent-f0-hnr-mfcc/speaker-dependent_PCs_percentChunks_text_checkpoints.ckpt_{}".format(counter)
        elif speakerSplit == "independent":
            #This means speaker c was left out of the training data
            counter = "c"
            splitName = "newSplit"
            modelPath = "Checkpoints/newSplit-f0-mfcc-plp/speaker-independent_2PCs_feats_percentChunks_text_checkpoints.ckpt_{}".format(counter)

        m = ModelTester(inputType, dataPath, counter, measureList, modelPath, speakerSplit=speakerSplit, splitName=splitName, fileMod = fileMod)
        m.testModel()
