#!/usr/bin/env python3

import os
from tensorflow import keras
import numpy as np
import pandas as pd
from tensorflow.keras import layers
from tensorflow.keras import models
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_recall_fscore_support

import keras_tuner as kt  
from keras_self_attention import SeqSelfAttention


def transformLabs(x):
    if x.upper() == "I":
        return 0.0
    elif x.upper() == "N":
        return 1.0
    else:
        print("Invalid label")
        raise Exception


def buildModel_FFNN(hp):

    model = models.Sequential()

    for l in range(hp.Int("num_dense", 1, 10)):

        model.add(layers.Dense(hp.Int("dense{}_size".format(l), 4, 64, step=2), activation='relu'))

        if hp.Boolean("dropout{}".format(l)):
            model.add(layers.Dropout(hp.Float("dropout{}_size".format(l), min_value=0.0, max_value=0.9, step=0.05)))

    model.add(layers.Dense(2, activation=hp.Choice('activation', ['softmax', 'sigmoid'])))

    model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    return model    


def buildModel_LSTM(hp):

    model = models.Sequential()

    model.add(layers.LSTM(hp.Int('lstm_size', 4, 64, step=2), activation="relu", dropout=hp.Float('dropout', min_value=0.0, max_value=0.9, step=0.05), recurrent_dropout=hp.Float('recurrent_dropout', min_value=0.0, max_value=0.9, step=0.05), return_sequences=True))

    model.add(SeqSelfAttention(attention_width=hp.Int('attention_width', 5, 25, step=5), attention_activation=hp.Choice('attention_activation', ['sigmoid', 'relu', 'tanh']), name="Attention"))
    model.add(layers.LSTM(1))

    model.add(layers.Dense(2, activation=hp.Choice('activation', ['softmax', 'sigmoid'])))

    model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy']) 
    
    return model


def main(fileMod, dataPath, modelType, dataType="timeSeries"):

    if dataType == "timeSeries":
        #Load data
        X_train = np.load("{}/{}_train_acoustic.npy".format(dataPath, fileMod))
        X_dev = np.load("{}/{}_dev_acoustic.npy".format(dataPath, fileMod))
        X_test = np.load("{}/{}_test_acoustic.npy".format(dataPath, fileMod))

        y_train = np.load("{}/{}_train_labels.npy".format(dataPath, fileMod))
        y_dev = np.load("{}/{}_dev_labels.npy".format(dataPath, fileMod))
        y_test = np.load("{}/{}_test_labels.npy".format(dataPath, fileMod))

        y_train = np.array([transformLabs(x) for x in y_train])
        y_dev = np.array([transformLabs(x) for x in y_dev])
        y_test = np.array([transformLabs(x) for x in y_test])

    elif dataType == "uttLevel":
        with open("{}/baseline_consolidated_{}.pkl".format(dataPath, fileMod), "rb") as p:
            df = pd.read_pickle(p)

        labs = np.array(df.pop("label"))
        newdf = np.array(df)

        X_train, X_test, y_train, y_test = train_test_split(newdf, labs, test_size=0.1, random_state=6)
        X_train, X_dev, y_train, y_dev = train_test_split(X_train, y_train, test_size=0.1, random_state=6)

        y_train = y_train.reshape((-1, 1))
        y_dev = y_dev.reshape((-1, 1))
        y_test = y_test.reshape((-1, 1))

    if modelType == "lstm":
        tuner = kt.RandomSearch(buildModel_LSTM, objective='val_accuracy', max_trials=50, directory="Results", project_name="percentChunks_withAttention")
    elif modelType == "ffnn":
        tuner = kt.RandomSearch(buildModel_FFNN, objective='val_accuracy', max_trials=50, directory="Results", project_name="baseline_ComParE")

    
    tuner.search(X_train, y_train, epochs=100, validation_data=(X_dev, y_dev))

    tuner.results_summary()


if __name__=="__main__":

    fileMod = "Pruned2"

    dataPath = "../percentChunks/Data"
    modelType = "lstm"
    dataType = "timeSeries"

    # dataPath = "../../../Data/AcousticData/ComParE"
    # modelType = "ffnn"
    # dataType = "uttLevel"

    main(fileMod, dataPath, modelType, dataType=dataType)