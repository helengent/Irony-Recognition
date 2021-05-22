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


def plotIt(history, modelName):
    acc = history.history['binary_accuracy']
    val_acc = history.history['val_binary_accuracy']

    loss = history.history['loss']
    val_loss = history.history['val_loss']



def main(df, csv_path, checkpoint_path):
    labs = np.array(df.pop("label"))
    newdf = np.array(df)

    X_train, X_test, y_train, y_test = train_test_split(newdf, labs, test_size=0.1, random_state=6)
    X_train, X_dev, y_train, y_dev = train_test_split(X_train, y_train, test_size=0.1, random_state=6)

    y_train = y_train.reshape((-1, 1))
    y_dev = y_dev.reshape((-1, 1))
    y_test = y_test.reshape((-1, 1))
    
    ffnn = FeedForwardNN(X_train, X_dev, X_test, y_train, y_dev, y_test, csv_path, checkpoint_path)

    ffnn.train()


if __name__ == "__main__":
    with open("../Data/baseline_consolidated.pkl", "rb") as p:
        bigDF = pd.read_pickle(p)

    csv_path = "../Checkpoints/ComParE_checkpoints.csv"
    checkpoint_path = "../Checkpoints/ComParE_checkpoints.ckpt"

    main(bigDF, csv_path, checkpoint_path)