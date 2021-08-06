#!/usr/bin/env python3

import os
import sys
import keras
import numpy as np
import pandas as pd
from keras import layers
from keras import models

sys.path.append(os.path.join('/'.join(sys.path[1].split("/")[:-3]), 'Models'))
from LSTM_acousticOnly import acousticOnlyLSTM


def performanceReport(train_stats_all, test_stats_all, speakerList, fileMod, speakerSplit="independent"):

    with open("{}_performanceReport_speaker-{}.txt".format(fileMod, speakerSplit), "w") as f:

        train_precisions, train_recalls, train_f1s = list(), list(), list()
        test_precisions, test_recalls, test_f1s = list(), list(), list()

        if speakerSplit == "independent":
            for speaker, train_stats, test_stats in zip(speakerList, train_stats_all, test_stats_all):

                f.write("Speaker {} left out\n\n".format(speaker))

                f.write("Training performance\n")
                f.write("Precision:\t{}\n".format(train_stats[0]))
                f.write("Recall:\t{}\n".format(train_stats[1]))
                f.write("F1:\t{}\n\n".format(train_stats[2]))

                train_precisions.append(train_stats[0])
                train_recalls.append(train_stats[1])
                train_f1s.append(train_stats[2])

                f.write("Test performance\n")
                f.write("Precision:\t{}\n".format(test_stats[0]))
                f.write("Recall:\t{}\n".format(test_stats[1]))
                f.write("F1:\t{}\n\n".format(test_stats[2]))

                test_precisions.append(test_stats[0])
                test_recalls.append(test_stats[1])
                test_f1s.append(test_stats[2])

            f.write("Cross-Speaker Average Performance\n\n")
            
            f.write("Train performance\n")
            f.write("Precision:\t{}\n".format(np.mean(train_precisions, axis=0)))
            f.write("Recall:\t{}\n".format(np.mean(train_recalls, axis=0)))
            f.write("F1:\t{}\n\n".format(np.mean(train_f1s, axis=0)))

            f.write("Test performance\n")
            f.write("Precision:\t{}\n".format(np.mean(test_precisions, axis=0)))
            f.write("Recall:\t{}\n".format(np.mean(test_recalls, axis=0)))
            f.write("F1:\t{}\n".format(np.mean(test_f1s, axis=0)))

        else:
            for train_stats, test_stats in zip(train_stats_all, test_stats_all):
                f.write("Training performance\n")
                f.write("Precision:\t{}\n".format(train_stats[0]))
                f.write("Recall:\t{}\n".format(train_stats[1]))
                f.write("F1:\t{}\n\n".format(train_stats[2]))

                train_precisions.append(train_stats[0])
                train_recalls.append(train_stats[1])
                train_f1s.append(train_stats[2])

                f.write("Test performance\n")
                f.write("Precision:\t{}\n".format(test_stats[0]))
                f.write("Recall:\t{}\n".format(test_stats[1]))
                f.write("F1:\t{}\n\n".format(test_stats[2]))


def transformLabs(x):
    if x.upper() == "I":
        return 0.0
    elif x.upper() == "N":
        return 1.0
    else:
        print("Invalid label")
        raise Exception


def main(fileMod, speakerList, csv_path, checkpoint_path, speakerSplit="independent"):

    train_performance_list, test_performance_list = list(), list()

    class_weights = {0.0: 1.0, 1.0: 1.0}
    
    if speakerSplit == "independent":
        for speaker in speakerList:

            #Load data with this speaker as the test data only
            X_train = np.load("../Data/{}_{}LeftOut_train_acoustic.npy".format(fileMod, speaker))
            X_dev = np.load("../Data/{}_{}LeftOut_dev_acoustic.npy".format(fileMod, speaker))
            X_test = np.load("../Data/{}_{}LeftOut_test_acoustic.npy".format(fileMod, speaker))

            y_train = np.load("../Data/{}_{}LeftOut_train_labels.npy".format(fileMod, speaker))
            y_dev = np.load("../Data/{}_{}LeftOut_dev_labels.npy".format(fileMod, speaker))
            y_test = np.load("../Data/{}_{}LeftOut_test_labels.npy".format(fileMod, speaker))

            y_train = np.array([transformLabs(x) for x in y_train])
            y_dev = np.array([transformLabs(x) for x in y_dev])
            y_test = np.array([transformLabs(x) for x in y_test])

            lstm = acousticOnlyLSTM(X_train, X_dev, X_test, y_train, y_dev, y_test, csv_path, checkpoint_path, class_weights, heirarchical=True)

            lstm.train()

            train_stats, test_stats = lstm.test()

            train_performance_list.append(train_stats)
            test_performance_list.append(test_stats)

    else:

        #Load data
        X_train = np.load("../Data/{}_train_acoustic.npy".format(fileMod))
        X_dev = np.load("../Data/{}_dev_acoustic.npy".format(fileMod))
        X_test = np.load("../Data/{}_test_acoustic.npy".format(fileMod))

        y_train = np.load("../Data/{}_train_labels.npy".format(fileMod))
        y_dev = np.load("../Data/{}_dev_labels.npy".format(fileMod))
        y_test = np.load("../Data/{}_test_labels.npy".format(fileMod))

        y_train = np.array([transformLabs(x) for x in y_train])
        y_dev = np.array([transformLabs(x) for x in y_dev])
        y_test = np.array([transformLabs(x) for x in y_test])

        lstm = acousticOnlyLSTM(X_train, X_dev, X_test, y_train, y_dev, y_test, csv_path, checkpoint_path, class_weights, heirarchical=True)

        lstm.train()

        train_stats, test_stats = lstm.test()

        train_performance_list.append(train_stats)
        test_performance_list.append(test_stats)


    performanceReport(train_performance_list, test_performance_list, speakerList, fileMod, speakerSplit=speakerSplit)


if __name__=="__main__":

    csv_path = "../Checkpoints/LSTM_checkpoints.csv"
    checkpoint_path = "../Checkpoints/LSTM_checkpoints.ckpt"

    fileMod = "Pruned2"
    speakerList = ["e", "j", "o", "s", "u", "c", "d"]

    main(fileMod, speakerList, csv_path, checkpoint_path)#, speakerSplit="dependent")