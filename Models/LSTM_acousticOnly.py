#!/usr/bin/env python3

import os
import keras
import numpy as np
import pandas as pd
from keras import layers
from keras import models
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_recall_fscore_support



#This is designed to handle sequential data
#Unimodal 1D input - Probably not great for things like mfccs, ams, rastaplp
class acousticOnlyLSTM():


    def __init__(self, X_train, X_dev, X_test, y_train, y_dev, y_test, csv_path, checkpoint_path, class_weights, speaker=None):

        # Load in training, dev, and test data
        self.train_in = X_train
        self.dev_in = X_dev
        self.test_in = X_test

        # Load in training, dev, and test labels
        self.train_out = y_train
        self.dev_out = y_dev
        self.test_out = y_test

        self.csv_path = csv_path
        self.checkpoint_path = checkpoint_path
        self.class_weights = class_weights

        if speaker:
            self.plotName = "Speaker_{}_leftOut".format(speaker)
        else:
            self.plotName = "SpeakerDependent"

        self.model = models.Sequential()

        self.model.add(layers.LSTM(24, activation="relu", dropout=0.05, recurrent_dropout=0.2))
        self.model.add(layers.Dense(2, activation='sigmoid'))

        self.model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])


    def plotHist(self):

        trainAcc = self.history.history['accuracy']
        testAcc = self.history.history['val_accuracy']
        trainLoss = self.history.history['loss']
        testLoss = self.history.history['val_loss']

        epochs = range(1, len(trainAcc) + 1)

        plt.figure()
        plt.subplot(1, 2, 1)
        plt.plot(epochs, trainAcc, 'g', label="Train Accuracy")
        plt.plot(epochs, testAcc, 'r', label="Dev Accuracy")
        plt.title("Accuracy")
        plt.legend()

        plt.subplot(1, 2, 2)
        plt.plot(epochs, trainLoss, 'g', label="Train Loss")
        plt.plot(epochs, testLoss, 'r', label="Dev Loss")
        plt.title("Loss")
        plt.legend()

        plt.savefig("{}.png".format(self.plotName))


    def train(self):
        # Stop early if training loss rises
        es = keras.callbacks.EarlyStopping(monitor='loss', patience=3)

        # CSV Logger for checking later
        csv = keras.callbacks.CSVLogger(filename = self.csv_path, separator = ',', append = False)

        # Save the model weights 
        cp = keras.callbacks.ModelCheckpoint(filepath = self.checkpoint_path, verbose=1, monitor = "val_loss", save_best_only = True, mode = "min")

        # Fit the model
        self.history = self.model.fit(self.train_in, self.train_out, epochs = 150, batch_size = 64, validation_data = (self.dev_in, self.dev_out), callbacks=[es, csv, cp], class_weight=self.class_weights)
        self.plotHist()


    def test(self):

        train_preds = self.model.predict(self.train_in).argmax(axis=1)
        test_preds = self.model.predict(self.test_in).argmax(axis=1)

        train_performance = precision_recall_fscore_support(self.train_out, train_preds)
        test_performance = precision_recall_fscore_support(self.test_out, test_preds)

        return(train_performance, test_performance)            