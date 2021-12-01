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
from sklearn.metrics import precision_recall_fscore_support, confusion_matrix, accuracy_score, roc_auc_score, roc_curve
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


class ModelTrainer:

    def __init__(self, fileMod, fileList, speakerList, inputType, dataPath, speakerSplit="independent", f0Normed=False, percentage=10, measureList=None, frameMax=200):
        self.fileMod = fileMod
        self.fileList = fileList
        self.speakerList = speakerList
        self.glob_acoustic, self.seq_acoustic, self.text = inputType 
        self.dataPath = dataPath
        self.speakerSplit = speakerSplit
        self.f0Normed = f0Normed
        self.percentage = percentage
        self.measureList = measureList
        self.frameMax = frameMax

        if self.glob_acoustic:
            if self.glob_acoustic == "ComParE":
                self.glob_file = pd.read_csv("{}/{}/all_inputs.csv".format(self.dataPath, self.glob_acoustic), index_col=0)
            else:
                self.glob_file = pd.read_csv("{}/{}/all_inputs.csv".format(self.dataPath, self.glob_acoustic))


        self.prefix = "speaker-{}_".format(self.speakerSplit)
        if self.glob_acoustic:
            self.prefix = self.prefix + "{}-globAcoustic_".format(inputType[0])
        if self.seq_acoustic:
            self.prefix = self.prefix + "{}-seqAcoustic_".format(inputType[1])
        if self.text:
            self.prefix = self.prefix + "text_"

        self.csv_path = "Checkpoints/{}checkpoints.csv".format(self.prefix)
        self.checkpoint_path = "Checkpoints/{}checkpoints.ckpt".format(self.prefix)
        
        if self.speakerSplit == "independent":
            counters = self.speakerList
        else:
            counters = range(5)
        self.plot_paths = {counter: "Plots/{}_{}.png".format(self.prefix, counter) for counter in counters}

        self.train_performance_list, self.test_performance_list, self.rocStats = list(), list(), list()
        self.class_weights = {0.0: 1.0, 1.0: 1.0}
    
        self.train_list, self.dev_list, self.test_list = self.trainTestSplit()
        self.glob_scaler, self.seq_scaler = self.createScalers()
        self.tokenizer = self.makeTokenizer()

        self.prepareData()


    #Split data into training, dev, and test sets using either LOSO or 5-fold cross-validation
    def trainTestSplit(self):
        train_list, dev_list, test_list = list(), list(), list()

        if self.speakerSplit == "independent":
            for speaker in self.speakerList:

                leftOutList = [item for item in self.fileList if item.split("_")[1][0] != speaker]
                tr, d = train_test_split(leftOutList, test_size=0.1, shuffle=True, stratify=[item[-1] for item in leftOutList], random_state=6)
                t = [item for item in self.fileList if item.split("_")[1][0] == speaker]

                tr_labs = [item[-1] for item in tr]
                d_labs = [item[-1] for item in d]
                t_labs = [item[-1] for item in t]

                train_list.append((tr, tr_labs, speaker))
                dev_list.append((d, d_labs, speaker))
                test_list.append((t, t_labs, speaker))
        
        else:
            #Do 5-fold cross validation
            lab_list = np.array([item[-1] for item in self.fileList])
            skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=6)
            count = 0
            for train_index, test_index in skf.split(self.fileList, lab_list):
                tr = np.array(self.fileList)[train_index]   
                t = np.array(self.fileList)[test_index]
            
                tr, d = train_test_split(tr, test_size=0.1, shuffle=True, stratify=lab_list[train_index], random_state=6)
                
                tr_labs = [item[-1] for item in tr]
                d_labs = [item[-1] for item in d]
                t_labs = lab_list[test_index]

                train_list.append((tr, tr_labs, count))
                dev_list.append((d, d_labs, count))
                test_list.append((t, t_labs, count))
                count += 1
        
        return train_list, dev_list, test_list


    def Hz2Mels(self, value):
        return (1/np.log(2)) * (np.log(1 + (value/1000))) * 1000


    def normF0(self, f0, speaker, normType="m"):
        #possible normTypes: "m", "z", "d"
        f0_mean = speaker.getSpeakerMeanF0()
        f0_sd = speaker.getSpeakerSDF0()
        if normType == "m":
            normedVec = [(self.Hz2Mels(value) - f0_mean)/f0_mean for value in f0]
        elif normType == "z":
            normedVec = [(self.Hz2Mels(value) - f0_mean)/f0_sd for value in f0]
        elif normType == "d":
            normedVec = [self.Hz2Mels(value) - f0_mean for value in f0]
        else:
            raise ValueError("Invalid normType")
        return np.array(normedVec)


    def scaleIt(self, bigList):

        chunkSize = np.shape(bigList[0])[0]
        biggestFriend = np.zeros((len(bigList), chunkSize, np.shape(bigList[0])[1]))
        longLad = list()

        for utt in bigList:
            for chunk in utt:
                longLad.append(chunk)
        
        longLad = np.array(longLad)
        scaled = self.seq_scaler.transform(longLad)

        start = 0
        stop = chunkSize
        for i in range(len(bigList)):
            small = scaled[start:stop]
            biggestFriend[i, :, :] = small
            start += chunkSize
            stop += chunkSize

        return biggestFriend


    def makeScaler(self, nList):
        longLad = list()
        
        for utt in nList:
            for chunk in utt:
                longLad.append(chunk)
        
        longLad = np.array(longLad)
        scaler = StandardScaler()
        scaler.fit(longLad)
        
        return scaler


    def chunkStats(self, x):
        if 100 % self.percentage != 0:
            raise ValueError("percentage must be an integer that 100 is divisible by")
        stats = list()
        numChunks = int(100/self.percentage)
        chunkSize = int(len(x)/numChunks)
        start = 0

        for n in range(numChunks):
            chunk = x[start:start+chunkSize]
            chunkMean = np.mean(chunk)
            chunkSD = sd(chunk, chunkMean)
            stats.append([chunkMean, chunkSD])
            start += chunkSize
        
        return stats


    #   Assembles a numpy array with all measures from all files of shape (n, x, y) where
    #       n = the length of fileList
    #       x = frame_max (the number of frames padded/truncated to per file)
    #       y = the number of acoustic measures (possibly multiple per item in measureList e.g. ams=375)
    #   the speaker variable is the speaker left OUT of the training and dev data
    def assembleArray(self, smallList):

        unscaled = list()

        for fileName in smallList:

            speaker = fileName.split("_")[1][0]

            s = Speaker(speaker, "/home/hmgent2/Data/AcousticData/SpeakerMetaData/{}_f0.txt".format(speaker.upper()), "/home/hmgent2/Data/AcousticData/SpeakerMetaData/{}_avgDur.txt".format(speaker.upper()))

            toStack = list()

            if "f0" in self.measureList:
                f0 = pd.read_csv("/home/hmgent2/Data/AcousticData/f0/{}.csv".format(fileName))["0"].tolist()
                if f0Normed:
                    f0 = self.normF0(f0, s, normType=f0Normed)
                else:
                    f0 = self.Hz2Mels(np.array(f0))
                if self.seq_acoustic == "percentChunks":
                    f0 = self.chunkStats(f0)
                elif self.seq_acoustic == "rawSequential":
                    while len(f0) < self.frameMax:
                        f0 = np.append(f0, 0)
                    if len(f0) > self.frameMax:
                        f0 = f0[:self.frameMax]
                    f0 = f0.reshape((-1, 1))
                toStack.append(f0)

            if "hnr" in self.measureList:
                hnr = np.array(pd.read_csv("/home/hmgent2/Data/AcousticData/hnr/{}.csv".format(fileName))["0"].tolist())
                if self.seq_acoustic == "percentChunks":
                    hnr = self.chunkStats(hnr)
                elif self.seq_acoustic == "rawSequential":
                    while len(hnr) < self.frameMax:
                        hnr = np.append(hnr, 0)
                    if len(hnr) > self.frameMax:
                        hnr = hnr[:self.frameMax]
                    hnr = hnr.reshape((-1, 1))
                toStack.append(hnr)

            if "mfcc" in self.measureList:
                for i in range(13):
                    mfcc = np.array(pd.read_csv("/home/hmgent2/Data/AcousticData/mfcc/{}.csv".format(fileName))[str(i)].tolist())
                    if self.seq_acoustic == "percentChunks":
                        mfcc = self.chunkStats(mfcc)
                    elif self.seq_acoustic == "rawSequential":
                        while len(mfcc) < self.frameMax:
                            mfcc = np.append(mfcc, 0)
                        if len(mfcc) > self.frameMax:
                            mfcc = mfcc[:self.frameMax]
                        mfcc = mfcc.reshape((-1, 1))
                    toStack.append(mfcc)

            if "plp" in self.measureList:
                for i in range(9):
                    plp = np.array(pd.read_csv("/home/hmgent2/Data/AcousticData/plp/{}.csv".format(fileName), header=None)[i].tolist())
                    if self.seq_acoustic == "percentChunks":
                        plp = self.chunkStats(plp)
                    elif self.seq_acoustic == "rawSequential":
                        while len(plp) < self.frameMax:
                            plp = np.append(plp, 0)
                        if len(plp) > self.frameMax:
                            plp = plp[:self.frameMax]
                        plp = plp.reshape((-1, 1))
                    toStack.append(plp)

            if "ams" in self.measureList:
                for i in range(375):
                    ams = np.array(pd.read_csv("/home/hmgent2/Data/AcousticData/ams/{}.csv".format(fileName), header=None)[i].tolist())
                    if self.seq_acoustic == "percentChunks":
                        ams = self.chunkStats(ams)
                    elif self.seq_acoustic == "rawSequential":
                        while len(ams) < self.frameMax:
                            ams = np.append(ams, 0)
                        if len(ams) > self.frameMax:
                            ams = ams[:self.frameMax]
                        ams = ams.reshape((-1, 1))
                    toStack.append(ams)

            assert len(list(set([len(item) for item in toStack]))) == 1

            wholeFriend = np.hstack(toStack)

            unscaled.append(wholeFriend)

        return unscaled


    def makeTokenizer(self):

        textList = list()

        for f in self.fileList:
            text = open("/home/hmgent2/Data/TextData/{}_asr/{}.txt".format(self.fileMod, f)).read()
            textList.append(text)

        tokenizer = Tokenizer(oov_token="UNK")
        tokenizer.fit_on_texts(textList)
        vocab_size = len(tokenizer.word_index) + 1
        print("VOCABULARY SIZE: {}".format(vocab_size))
        return tokenizer


    #Fit appropriate scaler(s) from non-ironic data
    def createScalers(self):
        #Create scaler for sequential data
        if self.seq_acoustic:
            if not os.path.exists("{}/{}/scaler.pkl".format(self.dataPath, self.seq_acoustic)):
                ndata = [item for item in self.fileList if item[-1] == "N"]
                ndata = self.assembleArray(ndata)
                seq_scaler= self.makeScaler(ndata)
                with open("{}/{}/scaler.pkl".format(dataPath, self.seq_acoustic), "wb") as f:
                    pickle.dump(seq_scaler, f)
            else:
                with open("{}/{}/scaler.pkl".format(dataPath, self.seq_acoustic), "rb") as f:
                    seq_scaler = pickle.load(f)
        else:
            seq_scaler = None

        #create scaler for global data
        if self.glob_acoustic and "PCs" not in self.glob_acoustic:
            nData = self.glob_file[self.glob_file["label"] == "N"]
            nData.pop("fileName")
            nData.pop("speaker")
            nData.pop("label")
            glob_scaler = StandardScaler()
            glob_scaler.fit(nData)
        else:
            glob_scaler = None

        return glob_scaler, seq_scaler


    def performanceReport(self):

        with open("Results/performanceReport_{}_{}.txt".format(self.fileMod, self.prefix), "w") as f:

            test_precisions, test_recalls, test_f1s, test_accuracies = list(), list(), list(), list()

            if self.speakerSplit == "independent":
                for speaker, test_stats in zip(self.speakerList, self.test_performance_list):

                    f.write("Speaker {} left out\n\n".format(speaker))

                    f.write("Confusion Matrix\n\n")

                    f.write("{}\n\n".format(str(test_stats[2])))

                    f.write("Test performance\n")
                    f.write("Precision:\t{}\n".format(test_stats[0][0]))
                    f.write("Recall:\t{}\n".format(test_stats[0][1]))
                    f.write("F1:\t{}\n".format(test_stats[0][2]))
                    f.write("Accuracy:\t{}\n\n".format(test_stats[1]))

                    test_precisions.append(test_stats[0][0])
                    test_recalls.append(test_stats[0][1])
                    test_f1s.append(test_stats[0][2])
                    test_accuracies.append(test_stats[1])

                f.write("Cross-Speaker Average Performance\n\n")
                

                f.write("Test performance\n")
                f.write("Precision:\t{}\n".format(np.mean(test_precisions, axis=0)))
                f.write("Recall:\t{}\n".format(np.mean(test_recalls, axis=0)))
                f.write("F1:\t{}\n".format(np.mean(test_f1s, axis=0)))
                f.write("Accuracy:\t{}\n".format(np.mean(test_accuracies, axis=0)))

            else:
                for test_stats, counter in zip(self.test_performance_list, range(len(self.test_performance_list))):
                    f.write("Model {}\n\n".format(counter))

                    f.write("Confusion Matrix\n\n")

                    f.write("{}\n\n".format(str(test_stats[2])))

                    f.write("Test performance\n")
                    f.write("Precision:\t{}\n".format(test_stats[0][0]))
                    f.write("Recall:\t{}\n".format(test_stats[0][1]))
                    f.write("F1:\t{}\n".format(test_stats[0][2]))
                    f.write("Accuracy:\t{}\n\n".format(test_stats[1]))

                    test_precisions.append(test_stats[0][0])
                    test_recalls.append(test_stats[0][1])
                    test_f1s.append(test_stats[0][2])
                    test_accuracies.append(test_stats[1])

                f.write("Cross-Validated Average Performance\n\n")

                f.write("Test performance\n")
                f.write("Precision:\t{}\n".format(np.mean(test_precisions, axis=0)))
                f.write("Recall:\t{}\n".format(np.mean(test_recalls, axis=0)))
                f.write("F1:\t{}\n".format(np.mean(test_f1s, axis=0)))
                f.write("Accuracy:\t{}\n".format(np.mean(test_accuracies, axis=0)))


    def getROCstats(self, trueLabs, Ipreds):

        fpr, tpr, thresholds = roc_curve(trueLabs, Ipreds, pos_label=1)
        auc = roc_auc_score(trueLabs, Ipreds)

        return (fpr, tpr, thresholds, auc)


    def ROCcurve(self):

        tprs = list()
        aucs = list()

        mean_fpr = np.linspace(0, 1, 100)
        fig, ax = plt.subplots()

        for stats in self.rocStats:
            fpr, tpr, thresholds, auc = stats
            interp_tpr = np.interp(mean_fpr, fpr, tpr)
            interp_tpr[0] = 0.0
            tprs.append(interp_tpr)
            aucs.append(auc)

        ax.plot([0, 1], [0, 1], linestyle='--', lw=2, color="r", label="Chance", alpha=0.8)
        mean_tpr = np.mean(tprs, axis=0)
        mean_tpr[-1] = 1
        mean_auc = np.mean(aucs)
        std_auc = np.std(aucs)
        ax.plot(mean_fpr, mean_tpr, color="b", 
                label = "Mean ROC (AUC: {} std: {}".format(np.round(mean_auc, 2), np.round(std_auc, 2)), 
                lw=2, alpha=0.8)

        std_tpr = np.std(tprs, axis=0)
        tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
        tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
        ax.fill_between(mean_fpr, tprs_lower, tprs_upper, color="grey", alpha=0.2)

        ax.set(xlim=[-0.05, 1.05], ylim=[-0.05, 1.05], title="ROC Curve")
        ax.legend(loc="lower right")
        plt.savefig("Plots/ROC_{}.png".format(self.prefix))


    def transformLabs(self, x_list):
        new_list = list()
        for x in x_list:
            if x.upper() == "I":
                new_list.append(1.0)
            elif x.upper() == "N":
                new_list.append(0.0)
            else:
                print("Invalid label")
                raise Exception
        return new_list


    def prepareData(self):

        for train, dev, test in zip(self.train_list, self.dev_list, self.test_list):
        
            X_train, y_train, counter = train
            X_dev, y_dev, counter = dev
            X_test, y_test, counter = test

            if self.seq_acoustic:

                if not os.path.exists("{}/{}/speaker-{}_train-{}_acoustic.npy".format(self.dataPath, self.seq_acoustic, self.speakerSplit, counter)):
                    X_train = self.assembleArray(X_train)
                    X_train = self.scaleIt(X_train)
                    with open("{}/{}/speaker-{}_train-{}_acoustic.npy".format(self.dataPath, self.seq_acoustic, self.speakerSplit, counter), "wb") as f:
                        np.save(f, X_train)

                if not os.path.exists("{}/{}/speaker-{}_train-{}_labels.npy".format(self.dataPath, self.seq_acoustic, self.speakerSplit, counter)):
                    y_train = np.array(self.transformLabs(y_train))
                    with open("{}/{}/speaker-{}_train-{}_labels.npy".format(self.dataPath, self.seq_acoustic, self.speakerSplit, counter), "wb") as f:
                        np.save(f, y_train)

                if not os.path.exists("{}/{}/speaker-{}_dev-{}_acoustic.npy".format(self.dataPath, self.seq_acoustic, self.speakerSplit, counter)):
                    X_dev= self.assembleArray(X_dev)
                    X_dev = self.scaleIt(X_dev)
                    with open("{}/{}/speaker-{}_dev-{}_acoustic.npy".format(self.dataPath, self.seq_acoustic, self.speakerSplit, counter), "wb") as f:
                        np.save(f, X_dev)

                if not os.path.exists("{}/{}/speaker-{}_dev-{}_labels.npy".format(self.dataPath, self.seq_acoustic, self.speakerSplit, counter)):
                    y_dev = np.array(self.transformLabs(y_dev))
                    with open("{}/{}/speaker-{}_dev-{}_labels.npy".format(self.dataPath, self.seq_acoustic, self.speakerSplit, counter), "wb") as f:
                        np.save(f, y_dev)

                if not os.path.exists("{}/{}/speaker-{}_test-{}_acoustic.npy".format(self.dataPath, self.seq_acoustic, self.speakerSplit, counter)):
                    X_test = self.assembleArray(X_test)
                    X_test = self.scaleIt(X_test)
                    with open("{}/{}/speaker-{}_test-{}_acoustic.npy".format(self.dataPath, self.seq_acoustic, self.speakerSplit, counter), "wb") as f:
                        np.save(f, X_test)

                if not os.path.exists("{}/{}/speaker-{}_test-{}_labels.npy".format(self.dataPath, self.seq_acoustic, self.speakerSplit, counter)):
                    y_test = np.array(self.transformLabs(y_test))
                    with open("{}/{}/speaker-{}_test-{}_labels.npy".format(self.dataPath, self.seq_acoustic, self.speakerSplit, counter), "wb") as f:
                        np.save(f, y_test)


            if self.glob_acoustic:

                if not os.path.exists("{}/{}/speaker-{}_train-{}_acoustic.npy".format(self.dataPath, self.glob_acoustic, self.speakerSplit, counter)):
                    desiredIndices = list()
                    for i, item in enumerate(self.glob_file["fileName"].tolist()): 
                        if item in X_train: 
                            desiredIndices.append(i)
                    X_train = self.glob_file.iloc[desiredIndices]
                    X_train.pop("fileName")
                    X_train.pop("speaker")
                    X_train.pop("label")
                    
                    if "PCs" not in self.glob_acoustic:
                        X_train = self.glob_scaler.transform(X_train)
                    X_train = np.array(X_train)
                    with open("{}/{}/speaker-{}_train-{}_acoustic.npy".format(self.dataPath, self.glob_acoustic, self.speakerSplit, counter), "wb") as f:
                        np.save(f, X_train)

                if not os.path.exists("{}/{}/speaker-{}_train-{}_labels.npy".format(self.dataPath, self.glob_acoustic, self.speakerSplit, counter)):
                    y_train = np.array(self.transformLabs(y_train))
                    with open("{}/{}/speaker-{}_train-{}_labels.npy".format(self.dataPath, self.glob_acoustic, self.speakerSplit, counter), "wb") as f:
                        np.save(f, y_train)

                if not os.path.exists("{}/{}/speaker-{}_dev-{}_acoustic.npy".format(self.dataPath, self.glob_acoustic, self.speakerSplit, counter)):
                    desiredIndices = list()
                    for i, item in enumerate(self.glob_file["fileName"].tolist()): 
                        if item in X_dev: 
                            desiredIndices.append(i)
                    X_dev = self.glob_file.iloc[desiredIndices]
                    X_dev.pop("fileName")
                    X_dev.pop("speaker")
                    X_dev.pop("label")

                    if "PCs" not in self.glob_acoustic:
                        X_dev = self.glob_scaler.transform(X_dev)
                    X_dev = np.array(X_dev)
                    with open("{}/{}/speaker-{}_dev-{}_acoustic.npy".format(self.dataPath, self.glob_acoustic, self.speakerSplit, counter), "wb") as f:
                        np.save(f, X_dev)

                if not os.path.exists("{}/{}/speaker-{}_dev-{}_labels.npy".format(self.dataPath, self.glob_acoustic, self.speakerSplit, counter)):
                    y_dev = np.array(self.transformLabs(y_dev))
                    with open("{}/{}/speaker-{}_dev-{}_labels.npy".format(self.dataPath, self.glob_acoustic, self.speakerSplit, counter), "wb") as f:
                        np.save(f, y_dev)

                if not os.path.exists("{}/{}/speaker-{}_test-{}_acoustic.npy".format(self.dataPath, self.glob_acoustic, self.speakerSplit, counter)):
                    desiredIndices = list()
                    for i, item in enumerate(self.glob_file["fileName"].tolist()): 
                        if item in X_test: 
                            desiredIndices.append(i)
                    X_test = self.glob_file.iloc[desiredIndices]                    
                    X_test.pop("fileName")
                    X_test.pop("speaker")
                    X_test.pop("label")

                    if "PCs" not in self.glob_acoustic:
                        X_test = self.glob_scaler.transform(X_test)
                    X_test = np.array(X_test)
                    with open("{}/{}/speaker-{}_test-{}_acoustic.npy".format(self.dataPath, self.glob_acoustic, self.speakerSplit, counter), "wb") as f:
                        np.save(f, X_test)

                if not os.path.exists("{}/{}/speaker-{}_test-{}_labels.npy".format(self.dataPath, self.glob_acoustic, self.speakerSplit, counter)):
                    y_test = np.array(self.transformLabs(y_test))
                    with open("{}/{}/speaker-{}_test-{}_labels.npy".format(self.dataPath, self.glob_acoustic, self.speakerSplit, counter), "wb") as f:
                        np.save(f, y_test)


            if self.text:

                if not os.path.exists("{}/Text/speaker-{}_train-{}_tokenized.npy".format(self.dataPath, self.speakerSplit, counter)):
                    textList = list()
                    for f in X_train:
                        text = open("/home/hmgent2/Data/TextData/{}_asr/{}.txt".format(self.fileMod, f)).read()
                        textList.append(text)
                    sequences = self.tokenizer.texts_to_sequences(textList)
                    X_train = sequence.pad_sequences(sequences, padding="post", truncating="post", maxlen=25) #maxlen chosen as 95th percentile of sentence lengths
                    with open("{}/Text/speaker-{}_train-{}_tokenized.npy".format(self.dataPath, self.speakerSplit, counter), "wb") as f:
                        np.save(f, X_train)

                if not os.path.exists("{}/Text/speaker-{}_train-{}_labels.npy".format(self.dataPath, self.speakerSplit, counter)):
                    y_train = np.array(self.transformLabs(y_train))
                    with open("{}/Text/speaker-{}_train-{}_labels.npy".format(self.dataPath, self.speakerSplit, counter), "wb") as f:
                        np.save(f, y_train)

                if not os.path.exists("{}/Text/speaker-{}_dev-{}_tokenized.npy".format(self.dataPath, self.speakerSplit, counter)):
                    textList = list()
                    for f in X_dev:
                        text = open("/home/hmgent2/Data/TextData/{}_asr/{}.txt".format(self.fileMod, f)).read()
                        textList.append(text)
                    sequences = self.tokenizer.texts_to_sequences(textList)
                    X_dev = sequence.pad_sequences(sequences, padding="post", truncating="post", maxlen=25) #maxlen chosen as 95th percentile of sentence lengths
                    with open("{}/Text/speaker-{}_dev-{}_tokenized.npy".format(self.dataPath, self.speakerSplit, counter), "wb") as f:
                        np.save(f, X_dev)

                if not os.path.exists("{}/Text/speaker-{}_dev-{}_labels.npy".format(self.dataPath, self.speakerSplit, counter)):
                    y_dev = np.array(self.transformLabs(y_dev))
                    with open("{}/Text/speaker-{}_dev-{}_labels.npy".format(self.dataPath, self.speakerSplit, counter), "wb") as f:
                        np.save(f, y_dev)

                if not os.path.exists("{}/Text/speaker-{}_test-{}_tokenized.npy".format(self.dataPath, self.speakerSplit, counter)):
                    textList = list()
                    for f in X_test:
                        text = open("/home/hmgent2/Data/TextData/{}_asr/{}.txt".format(self.fileMod, f)).read()
                        textList.append(text)
                    sequences = self.tokenizer.texts_to_sequences(textList)
                    X_test = sequence.pad_sequences(sequences, padding="post", truncating="post", maxlen=25) #maxlen chosen as 95th percentile of sentence lengths
                    with open("{}/Text/speaker-{}_test-{}_tokenized.npy".format(self.dataPath, self.speakerSplit, counter), "wb") as f:
                        np.save(f, X_test)

                if not os.path.exists("{}/Text/speaker-{}_test-{}_labels.npy".format(self.dataPath, self.speakerSplit, counter)):
                    y_test = np.array(self.transformLabs(y_test))
                    with open("{}/Text/speaker-{}_test-{}_labels.npy".format(self.dataPath, self.speakerSplit, counter), "wb") as f:
                        np.save(f, y_test)


    def trainModel(self):

        for train, dev, test in zip(self.train_list, self.dev_list, self.test_list):
        
            X_train, y_train, counter = train
            X_dev, y_dev, counter = dev
            X_test, y_test, counter = test

            if self.seq_acoustic:

                seq_acoustic_train_data = np.load("{}/{}/speaker-{}_train-{}_acoustic.npy".format(self.dataPath, self.seq_acoustic, self.speakerSplit, counter))
                train_labs = np.load("{}/{}/speaker-{}_train-{}_labels.npy".format(self.dataPath, self.seq_acoustic, self.speakerSplit, counter))

                seq_acoustic_dev_data = np.load("{}/{}/speaker-{}_dev-{}_acoustic.npy".format(self.dataPath, self.seq_acoustic, self.speakerSplit, counter))
                dev_labs = np.load("{}/{}/speaker-{}_dev-{}_labels.npy".format(self.dataPath, self.seq_acoustic, self.speakerSplit, counter))

                seq_acoustic_test_data = np.load("{}/{}/speaker-{}_test-{}_acoustic.npy".format(self.dataPath, self.seq_acoustic, self.speakerSplit, counter))
                test_labs = np.load("{}/{}/speaker-{}_test-{}_labels.npy".format(self.dataPath, self.seq_acoustic, self.speakerSplit, counter))

            if self.glob_acoustic:

                glob_acoustic_train_data = np.load("{}/{}/speaker-{}_train-{}_acoustic.npy".format(self.dataPath, self.glob_acoustic, self.speakerSplit, counter))
                train_labs = np.load("{}/{}/speaker-{}_train-{}_labels.npy".format(self.dataPath, self.glob_acoustic, self.speakerSplit, counter))

                glob_acoustic_dev_data = np.load("{}/{}/speaker-{}_dev-{}_acoustic.npy".format(self.dataPath, self.glob_acoustic, self.speakerSplit, counter))
                dev_labs = np.load("{}/{}/speaker-{}_dev-{}_labels.npy".format(self.dataPath, self.glob_acoustic, self.speakerSplit, counter))

                glob_acoustic_test_data = np.load("{}/{}/speaker-{}_test-{}_acoustic.npy".format(self.dataPath, self.glob_acoustic, self.speakerSplit, counter))
                test_labs = np.load("{}/{}/speaker-{}_test-{}_labels.npy".format(self.dataPath, self.glob_acoustic, self.speakerSplit, counter))
            
            if self.text:
                text_train_data = np.load("{}/Text/speaker-{}_train-{}_tokenized.npy".format(self.dataPath, self.speakerSplit, counter))
                train_labs = np.load("{}/Text/speaker-{}_train-{}_labels.npy".format(self.dataPath, self.speakerSplit, counter))

                text_dev_data = np.load("{}/Text/speaker-{}_dev-{}_tokenized.npy".format(self.dataPath, self.speakerSplit, counter))
                dev_labs = np.load("{}/Text/speaker-{}_dev-{}_labels.npy".format(self.dataPath, self.speakerSplit, counter))

                text_test_data = np.load("{}/Text/speaker-{}_test-{}_tokenized.npy".format(self.dataPath, self.speakerSplit, counter))
                test_labs = np.load("{}/Text/speaker-{}_test-{}_labels.npy".format(self.dataPath, self.speakerSplit, counter))


            if self.seq_acoustic and not self.glob_acoustic and not self.text:
                self.model = acousticOnlyLSTM(seq_acoustic_train_data, seq_acoustic_dev_data, seq_acoustic_test_data, 
                                                train_labs, dev_labs, test_labs, "{}_{}".format(self.csv_path, counter), 
                                                "{}_{}".format(self.checkpoint_path, counter), self.plot_paths[counter], 
                                                self.class_weights)

            elif self.glob_acoustic and not self.seq_acoustic and not self.text:
                self.model = FeedForwardNN(glob_acoustic_train_data, glob_acoustic_dev_data, glob_acoustic_test_data, 
                                            train_labs, dev_labs, test_labs, "{}_{}".format(self.csv_path, counter), 
                                            "{}_{}".format(self.checkpoint_path, counter), self.plot_paths[counter], 
                                            self.class_weights)

            elif self.text and not self.seq_acoustic and not self.glob_acoustic:
                self.model = textOnlyNN(text_train_data, text_dev_data, text_test_data, 
                                        train_labs, dev_labs, test_labs, self.tokenizer, 
                                        "{}_{}".format(self.csv_path, counter), 
                                        "{}_{}".format(self.checkpoint_path, counter), 
                                        self.plot_paths[counter], self.class_weights)

            elif self.text and self.seq_acoustic and not self.glob_acoustic:
                # self.model = acousticTextLSTM(seq_acoustic_train_data, seq_acoustic_dev_data, seq_acoustic_test_data, 
                #                                 text_train_data, text_dev_data, text_test_data, 
                #                                 train_labs, dev_labs, test_labs, self.tokenizer, 
                #                                 "{}_{}".format(self.csv_path, counter), 
                #                                 "{}_{}".format(self.checkpoint_path, counter), 
                #                                 self.plot_paths[counter], self.class_weights)
                self.model = acousticTextLSTM_CNN(seq_acoustic_train_data, seq_acoustic_dev_data, seq_acoustic_test_data, 
                                                text_train_data, text_dev_data, text_test_data, 
                                                train_labs, dev_labs, test_labs, self.tokenizer, 
                                                "{}_{}".format(self.csv_path, counter), 
                                                "{}_{}".format(self.checkpoint_path, counter), 
                                                self.plot_paths[counter], self.class_weights)

            elif self.text and self.glob_acoustic and not self.seq_acoustic:
                self.model = acousticTextCNN_FFNN(glob_acoustic_train_data, glob_acoustic_dev_data, glob_acoustic_test_data, 
                                                text_train_data, text_dev_data, text_test_data, 
                                                train_labs, dev_labs, test_labs, self.tokenizer,
                                                "{}_{}".format(self.csv_path, counter), 
                                                "{}_{}".format(self.checkpoint_path, counter), 
                                                self.plot_paths[counter], self.class_weights)
            
            elif self.seq_acoustic and self.glob_acoustic and not self.text:
                self.model = acousticLSTM_FFNN(seq_acoustic_train_data, seq_acoustic_dev_data, seq_acoustic_test_data, 
                                                glob_acoustic_train_data, glob_acoustic_dev_data, glob_acoustic_test_data, 
                                                train_labs, dev_labs, test_labs,
                                                "{}_{}".format(self.csv_path, counter), 
                                                "{}_{}".format(self.checkpoint_path, counter), 
                                                self.plot_paths[counter], self.class_weights)
           
            elif self.seq_acoustic and self.glob_acoustic and self.text:
                self.model = acousticTextLSTM_CNN_FFNN(seq_acoustic_train_data, seq_acoustic_dev_data, seq_acoustic_test_data, 
                                                        glob_acoustic_train_data, glob_acoustic_dev_data, glob_acoustic_test_data, 
                                                        text_train_data, text_dev_data, text_test_data,
                                                        train_labs, dev_labs, test_labs, self.tokenizer, 
                                                        "{}_{}".format(self.csv_path, counter), 
                                                        "{}_{}".format(self.checkpoint_path, counter), 
                                                        self.plot_paths[counter], self.class_weights)

            else:
                print("Bad combination of model inputs... This shouldn't have been possible")

            self.model.train()
            print(self.model.model.summary())

            train_preds, test_preds = self.model.test()

            test_stats = precision_recall_fscore_support(test_labs, test_preds.argmax(axis=1))
            test_accuracy = accuracy_score(test_labs, test_preds.argmax(axis=1))
            test_cm = confusion_matrix(test_labs, test_preds.argmax(axis=1))

            self.test_performance_list.append((test_stats, test_accuracy, test_cm))

            rocStats = self.getROCstats(test_labs, test_preds[:, 1])
            self.rocStats.append(rocStats)

        self.performanceReport()
        self.ROCcurve()



if __name__=="__main__":

    fileMod = "Pruned3"
    fileList = glob("../../AudioData/Gated{}/*.wav".format(fileMod))
    baseList = np.array([item.split("/")[-1][:-4] for item in fileList])
    speakerList = list(set([item.split("_")[1][0] for item in baseList]))
    dataPath = "/home/hmgent2/Data/ModelInputs"
    speakerSplits = ["dependent", "independent"]
    # speakerSplits = ["independent"]

    #Make list of tuples with combinations of input types
    # glob_acoustic = [False, "PCs", "rawGlobal"]
    # seq_acoustic = [False, "percentChunks", "rawSequential"]
    # text = [False, True]

    # badList = [("rawGlobal", False, False), (False, "rawSequential", True)]

    # inputTypes = [(False, False, True), (False, "percentChunks", False), (False, "rawSequential", False), 
    #                 ("ComParE", False, False), ("PCs", False, False), ("PCs_feats", False, False), 
    #                 (False, "percentChunks", True)]

    inputTypes = [("ComParE", False, True), ("PCs", False, True), ("PCs_feats", False, True), ("rawGlobal", False, True), 
                    ("ComParE", "percentChunks", False), ("PCs", "percentChunks", False), ("PCs_feats", "percentChunks", False),("rawGlobal", "percentChunks", False), 
                    ("ComParE", "rawSequential", False), ("PCs", "rawSequential", False), ("PCs_feats", "rawSequential", False),("rawGlobal", "rawSequential", False), 
                    ("ComParE", "percentChunks", True), ("PCs", "percentChunks", True), ("PCs_feats", "percentChunks", True),("rawGlobal", "percentChunks", True),
                    ("ComParE", "rawSequential", False), ("PCs", "rawSequential", True), ("PCs_feats", "rawSequential", True),("rawGlobal", "rawSequential", True)]

    measureList = ["f0", "hnr", "mfcc", "plp"]
    f0Normed=False
    percentage=10

    for speakerSplit in speakerSplits:
        for inputType in inputTypes:

            try:
                m = ModelTrainer(fileMod, baseList, speakerList, inputType, dataPath, speakerSplit=speakerSplit, f0Normed=f0Normed, percentage=percentage, measureList = measureList)

                m.trainModel()
            except Exception as e:
                with open("bad.txt", "a+") as f:
                    f.write("{}\t{}\n{}\n\n".format(inputType, speakerSplit, e))