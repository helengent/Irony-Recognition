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
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, StratifiedKFold


from sklearn import svm
from sklearn.metrics import precision_recall_fscore_support
from sklearn.neighbors import KNeighborsClassifier
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.neural_network import MLPClassifier


sys.path.append("../../Models")
from LSTM_acousticOnly import acousticOnlyLSTM
from FeedForward import FeedForwardNN

sys.path.append("../../AcousticFeatureExtraction")
from speaker import Speaker
from sd import sd


class ModelTrainer:

    def __init__(self, fileMod, fileList, speakerList, inputType, dataPath, speakerSplit="independent"):
        self.fileMod = fileMod
        self.fileList = fileList
        self.speakerList = speakerList
        self.glob_acoustic = inputType 
        self.dataPath = dataPath
        self.speakerSplit = speakerSplit

        if self.glob_acoustic:
            if self.glob_acoustic == "ComParE":
                self.glob_file = pd.read_csv("{}/{}/all_inputs.csv".format(self.dataPath, self.glob_acoustic), index_col=0)
            else:
                self.glob_file = pd.read_csv("{}/{}/all_inputs.csv".format(self.dataPath, self.glob_acoustic))


        self.prefix = "speaker-{}_".format(self.speakerSplit)
        if self.glob_acoustic:
            self.prefix = self.prefix + "{}-globAcoustic_".format(inputType[0])

        self.modelList = [svm.SVC(kernel="rbf"), 
                                svm.SVC(kernel="poly"), 
                                svm.SVR(kernel="rbf"), 
                                KNeighborsClassifier(3), GaussianProcessClassifier(1.0 * RBF(1.0)),
                                DecisionTreeClassifier(max_depth=5), 
                                RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1),
                                MLPClassifier(alpha=1, max_iter=1000),
                                AdaBoostClassifier(),
                                GaussianNB(),
                                QuadraticDiscriminantAnalysis()]
        self.modelNameList = ["rbf SVC", "poly SVC", "rbf SVR", "kNeighbors", "Gaussian Process", 
                                "Decision Tree", "Random Forest", "MLP", "AdaBoost", "Naive Bayes", "QDA"]
        
        if self.speakerSplit == "independent":
            counters = self.speakerList
        else:
            counters = range(5)
        self.plot_paths = {counter: "Plots/{}_{}.png".format(self.prefix, counter) for counter in counters}

        self.test_performance_list = list()
    
        self.train_list, self.dev_list, self.test_list = self.trainTestSplit()
        self.glob_scaler = self.createScaler()

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


    #Fit appropriate scaler(s) from non-ironic data
    def createScaler(self):

        #create scaler for global data
        if "PCs" not in self.glob_acoustic:
            nData = self.glob_file[self.glob_file["label"] == "N"]
            nData.pop("fileName")
            nData.pop("speaker")
            nData.pop("label")
            glob_scaler = StandardScaler()
            glob_scaler.fit(nData)
        else:
            glob_scaler = None

        return glob_scaler


    def performanceReport(self):

        test_precisions, test_recalls, test_f1s, test_scores = dict(), dict(), dict(), dict()

        for test_stats, counter in zip(self.test_performance_list, range(len(self.test_performance_list))):

            name = test_stats[2]
            score = test_stats[1]

            if name not in test_precisions:
                test_precisions[name] = list()
                test_recalls[name] = list()
                test_f1s[name] = list()
                test_scores[name] = list()
            test_precisions[name].append(test_stats[0][0])
            test_recalls[name].append(test_stats[0][1])
            test_f1s[name].append(test_stats[0][2])
            test_scores[name].append(score)

        avgDict = {"Classifier Name": [], "Average Precision": [], "Average Recall": [], "Average F1": [], "Average Accuracy": []}

        for n in self.modelNameList:
            avgDict["Classifier Name"].append(n)
            avgDict["Average Precision"].append(np.mean(test_precisions[n]))
            avgDict["Average Recall"].append(np.mean(test_recalls[n]))
            avgDict["Average F1"].append(np.mean(test_f1s[n]))
            avgDict["Average Accuracy"].append(np.mean(test_scores[n]))

        outDF = pd.DataFrame(avgDict)
        outDF.to_csv("Results/performanceReport_{}_{}.txt".format(self.fileMod, self.prefix))


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

            if not os.path.exists("{}/{}/speaker-{}_train-{}_acoustic.npy".format(self.dataPath, self.glob_acoustic, self.speakerSplit, counter)):
                newX = pd.DataFrame()
                for name in X_train:
                    subset = self.glob_file[self.glob_file["fileName"] == name]
                    newX = newX.append(subset)
                X_train = newX
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
                newX = pd.DataFrame()
                for name in X_dev:
                    subset = self.glob_file[self.glob_file["fileName"] == name]
                    newX = newX.append(subset)
                X_dev = newX
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
                newX = pd.DataFrame()
                for name in X_test:
                    subset = self.glob_file[self.glob_file["fileName"] == name]
                    newX = newX.append(subset)
                X_test = newX                  
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


    def trainModel(self):

        for train, dev, test in zip(self.train_list, self.dev_list, self.test_list):
        
            X_train, y_train, counter = train
            X_dev, y_dev, counter = dev
            X_test, y_test, counter = test

            train_data = np.load("{}/{}/speaker-{}_train-{}_acoustic.npy".format(self.dataPath, self.glob_acoustic, self.speakerSplit, counter))
            train_labs = np.load("{}/{}/speaker-{}_train-{}_labels.npy".format(self.dataPath, self.glob_acoustic, self.speakerSplit, counter))

            dev_data = np.load("{}/{}/speaker-{}_dev-{}_acoustic.npy".format(self.dataPath, self.glob_acoustic, self.speakerSplit, counter))
            dev_labs = np.load("{}/{}/speaker-{}_dev-{}_labels.npy".format(self.dataPath, self.glob_acoustic, self.speakerSplit, counter))

            test_data = np.load("{}/{}/speaker-{}_test-{}_acoustic.npy".format(self.dataPath, self.glob_acoustic, self.speakerSplit, counter))
            test_labs = np.load("{}/{}/speaker-{}_test-{}_labels.npy".format(self.dataPath, self.glob_acoustic, self.speakerSplit, counter))
                
            for model, name in zip(self.modelList, self.modelNameList):

                train_data = np.concatenate((train_data, dev_data), axis=0)
                train_labs = np.concatenate((train_labs, dev_labs), axis=0)

                model.fit(train_data, train_labs)

                score = model.score(test_data, test_labs)

                preds = model.predict(test_data)
                preds = np.round(preds)

                test_performance = precision_recall_fscore_support(test_labs, preds)

                self.test_performance_list.append((test_performance, score, name))

        self.performanceReport()



if __name__=="__main__":

    fileMod = "Pruned3"
    fileList = glob("../../AudioData/Gated{}/*.wav".format(fileMod))
    baseList = np.array([item.split("/")[-1][:-4] for item in fileList])
    speakerList = list(set([item.split("_")[1][0] for item in baseList]))
    dataPath = "/home/hmgent2/Data/ModelInputs"
    speakerSplits = ["dependent", "independent"]

    inputTypes = ["PCs_feats", "PCs", "ComParE", "rawGlobal"]

    for speakerSplit in speakerSplits:
        for inputType in inputTypes:

            try:
                m = ModelTrainer(fileMod, baseList, speakerList, inputType, dataPath, speakerSplit=speakerSplit)

                m.trainModel()
            except:
                with Exception as e:
                    print(e)
