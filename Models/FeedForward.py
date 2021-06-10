#!/usr/bin/env python3

import keras
import numpy as np
import pandas as pd
from keras import layers
from keras import models
from sklearn.metrics import precision_recall_fscore_support


#This is designed to handle ComParE-style data where the input is non-sequential
class FeedForwardNN():


    def __init__(self, X_train, X_dev, X_test, y_train, y_dev, y_test, csv_path, checkpoint_path, class_weights):

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

        input_dim = np.shape(self.train_in)[-1]

        self.model = models.Sequential()
        self.model.add(layers.Dense(12, input_dim=input_dim, activation='relu'))
        self.model.add(layers.Dropout(0.5))
        self.model.add(layers.Dense(8, activation='relu'))
        self.model.add(layers.Dropout(0.5))
        self.model.add(layers.Dense(4, activation='relu'))
        self.model.add(layers.Dropout(0.5))
        self.model.add(layers.Dense(2, activation='softmax'))

        self.model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])


    def train(self):
        # Stop early if training loss rises
        es = keras.callbacks.EarlyStopping(monitor='loss', patience=3)

        # CSV Logger for checking later
        csv = keras.callbacks.CSVLogger(filename = self.csv_path, separator = ',', append = False)

        # Save the model weights 
        cp = keras.callbacks.ModelCheckpoint(filepath = self.checkpoint_path, verbose=1, monitor = "val_loss", save_best_only = True, mode = "min")

        # Fit the model
        self.history = self.model.fit(self.train_in, self.train_out, epochs = 150, batch_size = 64, validation_data = (self.dev_in, self.dev_out), callbacks=[es, csv, cp], class_weight = self.class_weights)


    def test(self):

        train_preds = self.model.predict(self.train_in).argmax(axis = 1)
        test_preds = self.model.predict(self.test_in).argmax(axis = 1)

        train_performance = precision_recall_fscore_support(self.train_out, train_preds)
        test_performance = precision_recall_fscore_support(self.test_out, test_preds)

        return(train_performance, test_performance)


            