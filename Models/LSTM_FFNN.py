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

from gensim.models import KeyedVectors


class acousticLSTM_FFNN():


    def __init__(self, seq_acoustic_train, seq_acoustic_dev, seq_acoustic_test, glob_acoustic_train, glob_acoustic_dev, glob_acoustic_test, y_train, y_dev, y_test, csv_path, checkpoint_path, plot_path, class_weights):

        # Load in training, dev, and test data
        self.train_input = [seq_acoustic_train, glob_acoustic_train]
        self.dev_input = [seq_acoustic_dev, glob_acoustic_dev]
        self.test_input = [seq_acoustic_test, glob_acoustic_test]

        # Load in training, dev, and test labels
        self.train_out = y_train
        self.dev_out = y_dev
        self.test_out = y_test

        self.csv_path = csv_path
        self.checkpoint_path = checkpoint_path
        self.class_weights = class_weights
        self.plotName = plot_path

        seq_acoustic = layers.Input(shape=(seq_acoustic_train.shape[1], seq_acoustic_train.shape[2]))
        glob_acoustic = layers.Input(shape=glob_acoustic_train.shape[-1])

        lstm = layers.LSTM(300, activation='relu', dropout=0.05, recurrent_dropout=0.2)(seq_acoustic)
        lstm = layers.Dense(300, activation='relu')(lstm)

        ffnn = layers.Dense(300)(glob_acoustic)

        fused = layers.Concatenate()([lstm, ffnn])
        
        dropout = layers.Dropout(0.5)(fused)
        x = keras.Model(inputs=[seq_acoustic, glob_acoustic], outputs=dropout)
            
        z = layers.Dense(16, activation='relu')(x.output)
        z = layers.Dense(2, activation='sigmoid')(z)

        self.model = keras.Model(inputs=x.input, outputs=z)
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

        plt.savefig(self.plotName)


    def train(self):
        # Stop early if training loss rises
        es = keras.callbacks.EarlyStopping(monitor='loss', patience=3)

        # CSV Logger for checking later
        csv = keras.callbacks.CSVLogger(filename = self.csv_path, separator = ',', append = False)

        # Save the model weights 
        cp = keras.callbacks.ModelCheckpoint(filepath = self.checkpoint_path, verbose=1, monitor = "val_loss", save_best_only = True, mode = "min")

        # Fit the model
        self.history = self.model.fit(self.train_input, self.train_out, epochs = 150, batch_size = 64, validation_data = (self.dev_input, self.dev_out), callbacks=[es, csv, cp], class_weight=self.class_weights)
        self.plotHist()


    def test(self):

        train_preds = self.model.predict(self.train_input)
        test_preds = self.model.predict(self.test_input)

        return(train_preds, test_preds)            