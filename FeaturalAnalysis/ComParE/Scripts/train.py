#!/usr/bin/env python3

import os
import sys
import keras
import numpy as np
import pandas as pd
from keras import layers
from keras import models
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split

sys.path.append(os.path.join('/'.join(sys.path[1].split("/")[:-3]), 'Models'))
from FeedForward import FeedForwardNN


def performanceReport(train_stats, test_stats, filemod):

    with open("{}_performanceReport.txt".format(fileMod), "w") as f:

        f.write("Training performance\n")
        f.write("Precision:\t{}\n".format(train_stats[0]))
        f.write("Recall:\t{}\n".format(train_stats[1]))
        f.write("F1:\t{}\n\n".format(train_stats[2]))

        f.write("Test performance\n")
        f.write("Precision:\t{}\n".format(test_stats[0]))
        f.write("Recall:\t{}\n".format(test_stats[1]))
        f.write("F1:\t{}\n\n".format(test_stats[2]))


def plotIt(history, modelName):
    acc = history.history['acc']
    val_acc = history.history['val_acc']

    loss = history.history['loss']
    val_loss = history.history['val_loss']



def main(df, csv_path, checkpoint_path, fileMod):
    labs = np.array(df.pop("label"))
    newdf = np.array(df)

    X_train, X_test, y_train, y_test = train_test_split(newdf, labs, test_size=0.1, random_state=6)
    X_train, X_dev, y_train, y_dev = train_test_split(X_train, y_train, test_size=0.1, random_state=6)

    y_train = y_train.reshape((-1, 1))
    y_dev = y_dev.reshape((-1, 1))
    y_test = y_test.reshape((-1, 1))

    class_weights = {0.0: 1.0, 1.0: 50.0}
    
    ffnn = FeedForwardNN(X_train, X_dev, X_test, y_train, y_dev, y_test, csv_path, checkpoint_path, class_weights)

    ffnn.train()

    train_stats, test_stats = ffnn.test()

    performanceReport(train_stats, test_stats, fileMod)


if __name__ == "__main__":

    fileMod = "Pruned2"

    with open("../../../AcousticData/ComParE/baseline_consolidated_{}.pkl".format(fileMod), "rb") as p:
        bigDF = pd.read_pickle(p)

    csv_path = "../Checkpoints/ComParE_checkpoints_{}.csv".format(fileMod)
    checkpoint_path = "../Checkpoints/ComParE_checkpoints_{}.ckpt".format(fileMod)

    main(bigDF, csv_path, checkpoint_path, fileMod)