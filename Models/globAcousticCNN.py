#!/usr/bin/env python3

import tensorflow as tf
import keras
import numpy as np
import pandas as pd
from keras import layers
from keras import models
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_fscore_support
from gensim.models import KeyedVectors


#This is designed to handle ComParE-style data where the input is non-sequential
class globAcousticCNN():


    def __init__(self, X_train, X_dev, X_test, y_train, y_dev, y_test, csv_path, checkpoint_path, plot_path, class_weights):

        # Load in training, dev, and test data
        self.train_in = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
        self.dev_in = X_dev.reshape((X_dev.shape[0], X_dev.shape[1], 1))
        self.test_in = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))

        # Load in training, dev, and test labels
        self.train_out = y_train
        self.dev_out = y_dev
        self.test_out = y_test

        self.csv_path = csv_path
        self.checkpoint_path = checkpoint_path
        self.class_weights = class_weights
        self.plotName = plot_path

        input_dim = np.shape(self.train_in)[-2]

        acoustic_in = layers.Input(shape=(input_dim, 1))

        acousticConvs = list()
        for pooling_size in [3, 4, 5]:
            conv = layers.Conv1D(128, pooling_size, activation='relu')(acoustic_in)
            conv = layers.GlobalMaxPooling1D()(conv)
            acousticConvs.append(conv)

        acoustic = layers.Concatenate()(acousticConvs)
        acoustic = layers.Dense(300, activation='relu')(acoustic)

        dropout = layers.Dropout(0.5)(acoustic)
        x = keras.Model(inputs=[acoustic_in], outputs=dropout)
            
        z = layers.Dense(16, activation='relu')(x.output)
        z = layers.Dense(2, activation='sigmoid')(z)

        self.model = keras.Model(inputs=x.input, outputs=z)
        self.model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

        # self.model = models.Sequential()
        # self.model.add(layers.Embedding(input_dim=embeddings.shape[0], output_dim=embeddings.shape[1], input_length=input_dim, weights=[embeddings], trainable=False))
        
        # self.model.add(layers.Conv1D(128, 3, activation='relu'))
        # self.model.add(layers.GlobalAveragePooling1D())
        
        # self.model.add(layers.Dropout(0.5))

        # self.model.add(layers.Dense(16, activation='relu'))
        # self.model.add(layers.Dense(2, activation='softmax'))

        # self.model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
        # self.model.compile(loss=keras.losses.BinaryCrossentropy(from_logits=True), optimizer='adam', metrics=['binary_accuracy'])


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
        self.history = self.model.fit(self.train_in, self.train_out, epochs = 150, batch_size = 64, validation_data = (self.dev_in, self.dev_out), callbacks=[es, csv, cp], class_weight = self.class_weights)
        self.plotHist()


    def test(self):

        train_preds = self.model.predict(self.train_in)
        test_preds = self.model.predict(self.test_in)

        return(train_preds, test_preds)


            