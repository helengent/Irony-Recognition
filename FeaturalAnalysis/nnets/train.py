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
from LSTM_FFNN_CNN_withText_Attention import acousticTextLSTM_CNN_FFNN_Attention
from LSTM_FFNN_LSTM_withText import acousticTextLSTM_LSTM_FFNN

sys.path.append("../../AcousticFeatureExtraction")
from speaker import Speaker
from sd import sd


class ModelTrainer:

    def __init__(self, fileMod, fileList, speakerList, inputType, dataPath, subDir, speakerSplit="independent", f0Normed=False, percentage=10, measureList=None, frameMax=200, use_attention = False):
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
        self.subDir = subDir
        self.use_attention = use_attention

        if self.glob_acoustic:
            if self.glob_acoustic == "ComParE":
                self.glob_file = pd.read_csv("{}/{}/all_inputs.csv".format(self.dataPath, self.glob_acoustic), index_col=0)
            else:
                self.glob_file = pd.read_csv("{}/{}/all_inputs.csv".format(self.dataPath, self.glob_acoustic))

        self.prefix = "speaker-{}_".format(self.speakerSplit)
        if self.glob_acoustic:
            self.prefix = self.prefix + "{}_".format(inputType[0])
        if self.seq_acoustic:
            self.prefix = self.prefix + "{}_".format(inputType[1])
        if self.text:
            self.prefix = self.prefix + "text_"
        if self.use_attention:
            self.prefix = self.prefix + "ATTENTION_"

        self.csv_path = "Checkpoints/{}/{}checkpoints.csv".format(self.subDir, self.prefix)
        self.checkpoint_path = "Checkpoints/{}/{}checkpoints.ckpt".format(self.subDir, self.prefix)
        
        if "independent" in self.speakerSplit:
            counters = self.speakerList
        else:
            counters = range(5)
        self.plot_paths = {counter: "Plots/{}/{}_{}.png".format(self.subDir, self.prefix, counter) for counter in counters}

        self.train_performance_list, self.test_performance_list, self.rocStats = list(), list(), list()
        self.class_weights = {0.0: 1.0, 1.0: 1.0}
    
        self.train_list, self.dev_list, self.test_list = self.trainTestSplit()
        self.glob_scaler, self.seq_scaler, self.glob_imputer, self.seq_imputer = self.createScalers()
        self.tokenizer = self.makeTokenizer()

        self.prepareData()


    #Split data into training, dev, and test sets using either LOSO or 5-fold cross-validation
    def trainTestSplit(self):
        train_list, dev_list, test_list = list(), list(), list()

        if self.speakerSplit == "independent":
            for speaker in self.speakerList:

                leftOutList = [item for item in self.fileList if item.split("_")[1][0] not in speaker]
                tr, d = train_test_split(leftOutList, test_size=0.1, shuffle=True, stratify=[item[-1] for item in leftOutList], random_state=6)
                t = [item for item in self.fileList if item.split("_")[1][0] in speaker]

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
        if np.isnan(np.sum(longLad)):
            longLad = self.seq_imputer.transform(longLad)
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


    def makeImputer(self, nList):
        longLad = list()
        
        for utt in nList:
            for chunk in utt:
                longLad.append(chunk)
        
        longLad = np.array(longLad)
        imputer = SimpleImputer()
        imputer.fit(longLad)
        
        return imputer


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
            if not os.path.exists("{}/{}/{}/scaler.pkl".format(self.dataPath, self.seq_acoustic, self.subDir)):
                ndata = [item for item in self.fileList if item[-1] == "N"]
                ndata = self.assembleArray(ndata)
                seq_scaler= self.makeScaler(ndata)
                with open("{}/{}/{}/scaler.pkl".format(dataPath, self.seq_acoustic, self.subDir), "wb") as f:
                    pickle.dump(seq_scaler, f)
            else:
                with open("{}/{}/{}/scaler.pkl".format(dataPath, self.seq_acoustic, self.subDir), "rb") as f:
                    seq_scaler = pickle.load(f)

            if not os.path.exists("{}/{}/{}/imputer.pkl".format(self.dataPath, self.seq_acoustic, self.subDir)):
                ndata = [item for item in self.fileList if item[-1] == "N"]
                ndata = self.assembleArray(ndata)
                seq_imputer = self.makeImputer(ndata)
                with open("{}/{}/{}/imputer.pkl".format(dataPath, self.seq_acoustic, self.subDir), "wb") as f:
                    pickle.dump(seq_imputer, f)
            else:
                with open("{}/{}/{}/imputer.pkl".format(dataPath, self.seq_acoustic, self.subDir), "rb") as f:
                    seq_imputer = pickle.load(f)
        else:
            seq_scaler = None
            seq_imputer = None

        #create scaler for global data
        if self.glob_acoustic and "PCs" not in self.glob_acoustic:
            nData = self.glob_file[self.glob_file["label"] == "N"]
            nData.pop("fileName")
            nData.pop("speaker")
            nData.pop("label")
            glob_scaler = StandardScaler()
            glob_imputer = SimpleImputer()
            glob_scaler.fit(nData)
            glob_imputer.fit(nData)
        else:
            glob_scaler = None
            glob_imputer = None

        return glob_scaler, seq_scaler, glob_imputer, seq_imputer


    def performanceReport(self):

        with open("Results/{}/performanceReport_{}_{}.txt".format(self.subDir, self.fileMod, self.prefix), "w") as f:

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

        fnr = 1 - tpr
        eer_threshold = thresholds[np.nanargmin(np.absolute((fnr - fpr)))]
        eer = fpr[np.nanargmin(np.absolute((fnr - fpr)))]

        return (fpr, tpr, thresholds, auc, eer)


    def ROCcurve(self):

        tprs = list()
        aucs = list()
        eers = list()

        mean_fpr = np.linspace(0, 1, 100)
        fig, ax = plt.subplots()

        for stats in self.rocStats:
            fpr, tpr, thresholds, auc, eer = stats
            interp_tpr = np.interp(mean_fpr, fpr, tpr)
            interp_tpr[0] = 0.0
            tprs.append(interp_tpr)
            aucs.append(auc)
            eers.append(eer)

        ax.plot([0, 1], [0, 1], linestyle='--', lw=2, color="r", label="Chance", alpha=0.8)
        mean_tpr = np.mean(tprs, axis=0)
        mean_tpr[-1] = 1
        mean_auc = np.mean(aucs)
        std_auc = np.std(aucs)
        mean_eer = np.mean(eers)
        std_eer = np.std(eers)
        ax.plot(mean_fpr, mean_tpr, color="b", 
                label = "Mean ROC AUC: {} AUC std: {}\nEER: {} EER std: {}".format(np.round(mean_auc, 2), np.round(std_auc, 2), np.round(mean_eer, 2), np.round(std_eer, 2)), 
                lw=2, alpha=0.8)

        std_tpr = np.std(tprs, axis=0)
        tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
        tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
        ax.fill_between(mean_fpr, tprs_lower, tprs_upper, color="grey", alpha=0.2)

        ax.set(xlim=[-0.05, 1.05], ylim=[-0.05, 1.05], title="ROC Curve")
        ax.legend(loc="lower right")
        plt.savefig("Plots/{}/ROC_{}.png".format(self.subDir, self.prefix))


    def transformLabs(self, x_list):
        new_list = list()
        for x in x_list:
            if type(x) == str or type(x) == np.str_:
                if x.upper() == "I":
                    new_list.append(1.0)
                elif x.upper() == "N":
                    new_list.append(0.0)
                else:
                    print("Invalid label")
                    raise Exception
            else:
                new_list.append(x)
        return new_list


    def prepareData(self):

        for train, dev, test in zip(self.train_list, self.dev_list, self.test_list):
        
            X_train, y_train, counter = train
            X_dev, y_dev, counter = dev
            X_test, y_test, counter = test

            if self.seq_acoustic:

                if not os.path.exists("{}/{}/{}/speaker-{}_train-{}_acoustic.npy".format(self.dataPath, self.seq_acoustic, self.subDir, self.speakerSplit, counter)):
                    X_train = self.assembleArray(X_train)
                    X_train = self.scaleIt(X_train)
                    with open("{}/{}/{}/speaker-{}_train-{}_acoustic.npy".format(self.dataPath, self.seq_acoustic, self.subDir, self.speakerSplit, counter), "wb") as f:
                        np.save(f, X_train)

                if not os.path.exists("{}/{}/{}/speaker-{}_train-{}_labels.npy".format(self.dataPath, self.seq_acoustic, self.subDir, self.speakerSplit, counter)):
                    # if type(y_train[0]) == str:
                    y_train = np.array(self.transformLabs(y_train))
                    with open("{}/{}/{}/speaker-{}_train-{}_labels.npy".format(self.dataPath, self.seq_acoustic, self.subDir, self.speakerSplit, counter), "wb") as f:
                        np.save(f, y_train)

                if not os.path.exists("{}/{}/{}/speaker-{}_dev-{}_acoustic.npy".format(self.dataPath, self.seq_acoustic, self.subDir, self.speakerSplit, counter)):
                    X_dev= self.assembleArray(X_dev)
                    X_dev = self.scaleIt(X_dev)
                    with open("{}/{}/{}/speaker-{}_dev-{}_acoustic.npy".format(self.dataPath, self.seq_acoustic, self.subDir, self.speakerSplit, counter), "wb") as f:
                        np.save(f, X_dev)

                if not os.path.exists("{}/{}/{}/speaker-{}_dev-{}_labels.npy".format(self.dataPath, self.seq_acoustic, self.subDir, self.speakerSplit, counter)):
                    # if type(y_dev[0]) == str:
                    y_dev = np.array(self.transformLabs(y_dev))
                    with open("{}/{}/{}/speaker-{}_dev-{}_labels.npy".format(self.dataPath, self.seq_acoustic, self.subDir, self.speakerSplit, counter), "wb") as f:
                        np.save(f, y_dev)

                if not os.path.exists("{}/{}/{}/speaker-{}_test-{}_acoustic.npy".format(self.dataPath, self.seq_acoustic, self.subDir, self.speakerSplit, counter)):
                    X_test = self.assembleArray(X_test)
                    X_test = self.scaleIt(X_test)
                    with open("{}/{}/{}/speaker-{}_test-{}_acoustic.npy".format(self.dataPath, self.seq_acoustic, self.subDir, self.speakerSplit, counter), "wb") as f:
                        np.save(f, X_test)

                if not os.path.exists("{}/{}/{}/speaker-{}_test-{}_labels.npy".format(self.dataPath, self.seq_acoustic, self.subDir, self.speakerSplit, counter)):
                    # if type(y_test[0]) == str:
                    y_test = np.array(self.transformLabs(y_test))
                    with open("{}/{}/{}/speaker-{}_test-{}_labels.npy".format(self.dataPath, self.seq_acoustic, self.subDir, self.speakerSplit, counter), "wb") as f:
                        np.save(f, y_test)

            X_train, y_train, counter = train
            X_dev, y_dev, counter = dev
            X_test, y_test, counter = test

            if self.glob_acoustic:

                if not os.path.exists("{}/{}/{}/speaker-{}_train-{}_acoustic.npy".format(self.dataPath, self.glob_acoustic, self.subDir, self.speakerSplit, counter)):
                    newX = pd.DataFrame()
                    for name in X_train:
                        subset = self.glob_file[self.glob_file["fileName"] == name]
                        newX = newX.append(subset)
                    X_train = newX
                    train_meta = pd.DataFrame()
                    train_meta["fileName"] = X_train.pop("fileName")
                    train_meta["speaker"] = X_train.pop("speaker")
                    train_meta["label"] = X_train.pop("label")
                    train_meta.to_csv("{}/speaker-{}_train_{}.meta".format(self.dataPath, self.speakerSplit, counter))
                    
                    if "PCs" not in self.glob_acoustic:
                        # if True in np.isnan(np.sum(X_train)).tolist():
                        #     X_train = self.glob_imputer.transform(X_train)
                        X_train = self.glob_scaler.transform(X_train)
                    X_train = np.array(X_train)
                    with open("{}/{}/{}/speaker-{}_train-{}_acoustic.npy".format(self.dataPath, self.glob_acoustic, self.subDir, self.speakerSplit, counter), "wb") as f:
                        np.save(f, X_train)

                if not os.path.exists("{}/{}/{}/speaker-{}_train-{}_labels.npy".format(self.dataPath, self.glob_acoustic, self.subDir, self.speakerSplit, counter)):
                    # if type(y_train[0]) == str:
                    y_train = np.array(self.transformLabs(y_train))
                    with open("{}/{}/{}/speaker-{}_train-{}_labels.npy".format(self.dataPath, self.glob_acoustic, self.subDir, self.speakerSplit, counter), "wb") as f:
                        np.save(f, y_train)

                if not os.path.exists("{}/{}/{}/speaker-{}_dev-{}_acoustic.npy".format(self.dataPath, self.glob_acoustic, self.subDir, self.speakerSplit, counter)):
                    newX = pd.DataFrame()
                    for name in X_dev:
                        subset = self.glob_file[self.glob_file["fileName"] == name]
                        newX = newX.append(subset)
                    X_dev = newX
                    dev_meta = pd.DataFrame()
                    dev_meta["fileName"] = X_dev.pop("fileName")
                    dev_meta["speaker"] = X_dev.pop("speaker")
                    dev_meta["label"] = X_dev.pop("label")
                    dev_meta.to_csv("{}/speaker-{}_dev_{}.meta".format(self.dataPath, self.speakerSplit, counter))

                    if "PCs" not in self.glob_acoustic:
                        # if np.isnan(np.sum(X_dev)):
                        #     X_train = self.glob_imputer.transform(X_dev)
                        X_dev = self.glob_scaler.transform(X_dev)
                    X_dev = np.array(X_dev)
                    with open("{}/{}/{}/speaker-{}_dev-{}_acoustic.npy".format(self.dataPath, self.glob_acoustic, self.subDir, self.speakerSplit, counter), "wb") as f:
                        np.save(f, X_dev)

                if not os.path.exists("{}/{}/{}/speaker-{}_dev-{}_labels.npy".format(self.dataPath, self.glob_acoustic, self.subDir, self.speakerSplit, counter)):
                    # if type(y_dev[0]) == str:
                    y_dev = np.array(self.transformLabs(y_dev))
                    with open("{}/{}/{}/speaker-{}_dev-{}_labels.npy".format(self.dataPath, self.glob_acoustic, self.subDir, self.speakerSplit, counter), "wb") as f:
                        np.save(f, y_dev)

                if not os.path.exists("{}/{}/{}/speaker-{}_test-{}_acoustic.npy".format(self.dataPath, self.glob_acoustic, self.subDir, self.speakerSplit, counter)): 
                    newX = pd.DataFrame()
                    for name in X_test:
                        subset = self.glob_file[self.glob_file["fileName"] == name]
                        newX = newX.append(subset)
                    X_test = newX    
                    test_meta = pd.DataFrame()
                    test_meta["fileName"] = X_test.pop("fileName")
                    test_meta["speaker"] = X_test.pop("speaker")
                    test_meta["label"] = X_test.pop("label")
                    test_meta.to_csv("{}/speaker-{}_test_{}.meta".format(self.dataPath, self.speakerSplit, counter))

                    if "PCs" not in self.glob_acoustic:
                        # if np.isnan(np.sum(X_test)):
                        #     X_train = self.glob_imputer.transform(X_test)
                        X_test = self.glob_scaler.transform(X_test)
                    X_test = np.array(X_test)
                    with open("{}/{}/{}/speaker-{}_test-{}_acoustic.npy".format(self.dataPath, self.glob_acoustic, self.subDir, self.speakerSplit, counter), "wb") as f:
                        np.save(f, X_test)

                if not os.path.exists("{}/{}/{}/speaker-{}_test-{}_labels.npy".format(self.dataPath, self.glob_acoustic, self.subDir, self.speakerSplit, counter)):
                    # if type(y_test[0]) == str:
                    y_test = np.array(self.transformLabs(y_test))
                    with open("{}/{}/{}/speaker-{}_test-{}_labels.npy".format(self.dataPath, self.glob_acoustic, self.subDir, self.speakerSplit, counter), "wb") as f:
                        np.save(f, y_test)

            X_train, y_train, counter = train
            X_dev, y_dev, counter = dev
            X_test, y_test, counter = test

            if self.text:

                if not os.path.exists("{}/Text/newSplit/speaker-{}_train-{}_tokenized.npy".format(self.dataPath, self.speakerSplit, counter)):
                    textList = list()
                    for f in X_train:
                        text = open("/home/hmgent2/Data/TextData/{}_asr/{}.txt".format(self.fileMod, f)).read()
                        textList.append(text)
                    sequences = self.tokenizer.texts_to_sequences(textList)
                    X_train = sequence.pad_sequences(sequences, padding="post", truncating="post", maxlen=25) #maxlen chosen as 95th percentile of sentence lengths
                    with open("{}/Text/newSplit/speaker-{}_train-{}_tokenized.npy".format(self.dataPath, self.speakerSplit, counter), "wb") as f:
                        np.save(f, X_train)

                if not os.path.exists("{}/newSplit/Text/speaker-{}_train-{}_labels.npy".format(self.dataPath, self.speakerSplit, counter)):
                    # if type(y_train[0]) == str:
                    y_train = np.array(self.transformLabs(y_train))
                    with open("{}/Text/newSplit/speaker-{}_train-{}_labels.npy".format(self.dataPath, self.speakerSplit, counter), "wb") as f:
                        np.save(f, y_train)

                if not os.path.exists("{}/Text/newSplit/speaker-{}_dev-{}_tokenized.npy".format(self.dataPath, self.speakerSplit, counter)):
                    textList = list()
                    for f in X_dev:
                        text = open("/home/hmgent2/Data/TextData/{}_asr/{}.txt".format(self.fileMod, f)).read()
                        textList.append(text)
                    sequences = self.tokenizer.texts_to_sequences(textList)
                    X_dev = sequence.pad_sequences(sequences, padding="post", truncating="post", maxlen=25) #maxlen chosen as 95th percentile of sentence lengths
                    with open("{}/Text/newSplit/speaker-{}_dev-{}_tokenized.npy".format(self.dataPath, self.speakerSplit, counter), "wb") as f:
                        np.save(f, X_dev)

                if not os.path.exists("{}/newSplit/Text/speaker-{}_dev-{}_labels.npy".format(self.dataPath, self.speakerSplit, counter)):
                    # if type(y_dev[0]) == str:
                    y_dev = np.array(self.transformLabs(y_dev))
                    with open("{}/Text/newSplit/speaker-{}_dev-{}_labels.npy".format(self.dataPath, self.speakerSplit, counter), "wb") as f:
                        np.save(f, y_dev)

                if not os.path.exists("{}/Text/newSplit/speaker-{}_test-{}_tokenized.npy".format(self.dataPath, self.speakerSplit, counter)):
                    textList = list()
                    for f in X_test:
                        text = open("/home/hmgent2/Data/TextData/{}_asr/{}.txt".format(self.fileMod, f)).read()
                        textList.append(text)
                    sequences = self.tokenizer.texts_to_sequences(textList)
                    X_test = sequence.pad_sequences(sequences, padding="post", truncating="post", maxlen=25) #maxlen chosen as 95th percentile of sentence lengths
                    with open("{}/Text/newSplit/speaker-{}_test-{}_tokenized.npy".format(self.dataPath, self.speakerSplit, counter), "wb") as f:
                        np.save(f, X_test)

                if not os.path.exists("{}/Text/newSplit/speaker-{}_test-{}_labels.npy".format(self.dataPath, self.speakerSplit, counter)):
                    # if type(y_test[0]) == str:
                    y_test = np.array(self.transformLabs(y_test))
                    with open("{}/Text/newSplit/speaker-{}_test-{}_labels.npy".format(self.dataPath, self.speakerSplit, counter), "wb") as f:
                        np.save(f, y_test)


    def trainModel(self):

        for train, dev, test in zip(self.train_list, self.dev_list, self.test_list):
        
            X_train, y_train, counter = train
            X_dev, y_dev, counter = dev
            X_test, y_test, counter = test

            if self.seq_acoustic:

                seq_acoustic_train_data = np.load("{}/{}/{}/speaker-{}_train-{}_acoustic.npy".format(self.dataPath, self.seq_acoustic, self.subDir, self.speakerSplit, counter))
                train_labs = np.load("{}/{}/{}/speaker-{}_train-{}_labels.npy".format(self.dataPath, self.seq_acoustic, self.subDir, self.speakerSplit, counter))

                seq_acoustic_dev_data = np.load("{}/{}/{}/speaker-{}_dev-{}_acoustic.npy".format(self.dataPath, self.seq_acoustic, self.subDir, self.speakerSplit, counter))
                dev_labs = np.load("{}/{}/{}/speaker-{}_dev-{}_labels.npy".format(self.dataPath, self.seq_acoustic, self.subDir, self.speakerSplit, counter))

                seq_acoustic_test_data = np.load("{}/{}/{}/speaker-{}_test-{}_acoustic.npy".format(self.dataPath, self.seq_acoustic, self.subDir, self.speakerSplit, counter))
                test_labs = np.load("{}/{}/{}/speaker-{}_test-{}_labels.npy".format(self.dataPath, self.seq_acoustic, self.subDir, self.speakerSplit, counter))

            if self.glob_acoustic:

                glob_acoustic_train_data = np.load("{}/{}/{}/speaker-{}_train-{}_acoustic.npy".format(self.dataPath, self.glob_acoustic, self.subDir, self.speakerSplit, counter))
                train_labs = np.load("{}/{}/{}/speaker-{}_train-{}_labels.npy".format(self.dataPath, self.glob_acoustic, self.subDir, self.speakerSplit, counter))

                glob_acoustic_dev_data = np.load("{}/{}/{}/speaker-{}_dev-{}_acoustic.npy".format(self.dataPath, self.glob_acoustic, self.subDir, self.speakerSplit, counter))
                dev_labs = np.load("{}/{}/{}/speaker-{}_dev-{}_labels.npy".format(self.dataPath, self.glob_acoustic, self.subDir, self.speakerSplit, counter))

                glob_acoustic_test_data = np.load("{}/{}/{}/speaker-{}_test-{}_acoustic.npy".format(self.dataPath, self.glob_acoustic, self.subDir, self.speakerSplit, counter))
                test_labs = np.load("{}/{}/{}/speaker-{}_test-{}_labels.npy".format(self.dataPath, self.glob_acoustic, self.subDir, self.speakerSplit, counter))
            
            if self.text:
                text_train_data = np.load("{}/Text/newSplit/speaker-{}_train-{}_tokenized.npy".format(self.dataPath, self.speakerSplit, counter))
                train_labs = np.load("{}/Text/newSplit/speaker-{}_train-{}_labels.npy".format(self.dataPath, self.speakerSplit, counter))

                text_dev_data = np.load("{}/Text/newSplit/speaker-{}_dev-{}_tokenized.npy".format(self.dataPath, self.speakerSplit, counter))
                dev_labs = np.load("{}/Text/newSplit/speaker-{}_dev-{}_labels.npy".format(self.dataPath, self.speakerSplit, counter))

                text_test_data = np.load("{}/Text/newSplit/speaker-{}_test-{}_tokenized.npy".format(self.dataPath, self.speakerSplit, counter))
                test_labs = np.load("{}/Text/newSplit/speaker-{}_test-{}_labels.npy".format(self.dataPath, self.speakerSplit, counter))

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
                if self.use_attention:
                    self.model = acousticTextLSTM_CNN_FFNN_Attention(seq_acoustic_train_data, seq_acoustic_dev_data, seq_acoustic_test_data, 
                                        glob_acoustic_train_data, glob_acoustic_dev_data, glob_acoustic_test_data, 
                                        text_train_data, text_dev_data, text_test_data,
                                        train_labs, dev_labs, test_labs, self.tokenizer, 
                                        "{}_{}".format(self.csv_path, counter), 
                                        "{}_{}".format(self.checkpoint_path, counter), 
                                        self.plot_paths[counter], self.class_weights)

                else:
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

            test_preds = self.model.test()

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

    # speakerList = ["c", "d", "ejou", "fhkqst"]
    speakerList = list(set([item.split("_")[1][0] for item in baseList]))

    dataPath = "/home/hmgent2/Data/ModelInputs"
    speakerSplits = ["dependent", "independent", "independent_2"] #"dependent", "independent", 
    att = False
    # speakerSplits = ["independent"]

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


    # inputTypes = [(False, "percentChunks", False), ("PCs", "percentChunks", True)]
    inputTypes = [("PCs", "percentChunks", True)]

    # measureLists = [["f0", "hnr", "mfcc", "plp"]]
    # measureLists = [["f0", "hnr"], ["f0", "mfcc"], ["f0", "plp"], ["f0", "hnr", "mfcc"], ["f0", "hnr", "plp"], 
    #                 ["hnr"], ["hnr", "mfcc"], ["hnr", "plp"], ["hnr", "mfcc", "plp"],  
    #                 ["mfcc"], ["mfcc", "plp"],
    #                 ["plp"]]
    measureLists = [["f0", "hnr", "mfcc"]]

    f0Normed=False
    percentage=10

    for measureList in measureLists:
            
        for inputType in inputTypes:
            for speakerSplit in speakerSplits:

                if speakerSplit == "dependent":
                    subDir = "speakerDependent-{}".format("-".join(measureList))
                    # subDir = "speakerDependent"
                elif speakerSplit == "independent":
                    subDir = "oldSplit-{}".format("-".join(measureList))
                    # subDir = "oldSplit"
                elif speakerSplit == "independent_2":
                    subDir = "newSplit-{}".format("-".join(measureList))
                    speakerList = ["c", "d", "ejou", "fhkqst"]
                    speakerSplit = "independent"

                if not os.path.isdir("Checkpoints/{}".format(subDir)):
                    for parent in ["Results", "Checkpoints", "Plots"]:
                        os.mkdir("{}/{}".format(parent, subDir))
                    # for parent in ["ComParE", "PCs", "PCs_feats", "percentChunks", "rawGlobal", "rawSequential"]:
                    for parent in ["PCs", "percentChunks"]:
                        os.mkdir("{}/{}/{}".format(dataPath, parent, subDir))

                try:
                    m = ModelTrainer(fileMod, baseList, speakerList, inputType, dataPath, subDir, speakerSplit=speakerSplit, f0Normed=f0Normed, percentage=percentage, measureList = measureList, use_attention = att)

                    m.trainModel()

                except Exception as e:
                    with open("bad.txt", "a+") as f:
                        f.write("{}\t{}\n{}\n\n".format(inputType, speakerSplit, e))