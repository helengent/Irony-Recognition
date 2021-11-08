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

    for l in range(6):

        model.add(layers.Dense(hp.Int("dense{}_size".format(l), 4, 64, step=2), activation='relu'))
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
        X_train = np.load("/home/hmgent2/Data/ModelInputs/PCs/speaker-dependent_train-0_acoustic.npy")
        y_train = np.load("/home/hmgent2/Data/ModelInputs/PCs/speaker-dependent_train-0_labels.npy")

        X_dev = np.load("/home/hmgent2/Data/ModelInputs/PCs/speaker-dependent_dev-0_acoustic.npy")
        y_dev = np.load("/home/hmgent2/Data/ModelInputs/PCs/speaker-dependent_dev-0_labels.npy")

        X_test = np.load("/home/hmgent2/Data/ModelInputs/PCs/speaker-dependent_test-0_acoustic.npy")
        y_test = np.load("/home/hmgent2/Data/ModelInputs/PCs/speaker-dependent_test-0_labels.npy")


    if modelType == "lstm":
        tuner = kt.RandomSearch(buildModel_LSTM, objective='val_accuracy', max_trials=50, directory="Results", project_name="percentChunks_withAttention")
    elif modelType == "ffnn":
        tuner = kt.RandomSearch(buildModel_FFNN, objective='val_accuracy', max_trials=50, directory="Results", project_name="PCs")


    tuner.search(X_train, y_train, epochs=100, validation_data=(X_dev, y_dev))

    tuner.results_summary()


if __name__=="__main__":

    fileMod = "Pruned3"

    # dataPath = "../percentChunks/Data"
    # modelType = "lstm"
    # dataType = "timeSeries"

    dataPath = "../../../Data/AcousticData/PCs"
    modelType = "ffnn"
    dataType = "uttLevel"

    main(fileMod, dataPath, modelType, dataType=dataType)