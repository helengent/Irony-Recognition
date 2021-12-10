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


#This is designed to handle sequential data
#Unimodal 1D input - Probably not great for things like mfccs, ams, rastaplp
class acousticTextLSTM():


    def __init__(self, acoustic_train, acoustic_dev, acoustic_test, text_train, text_dev, text_test, y_train, y_dev, y_test, tokenizer, csv_path, checkpoint_path, plot_path, class_weights):

        # Load in training, dev, and test data
        self.train_input = [text_train, acoustic_train]
        self.dev_input = [text_dev, acoustic_dev]
        self.test_input = [text_test, acoustic_test]

        # Load in training, dev, and test labels
        self.train_out = y_train
        self.dev_out = y_dev
        self.test_out = y_test

        self.csv_path = csv_path
        self.checkpoint_path = checkpoint_path
        self.class_weights = class_weights
        self.plotName = plot_path

        input_dim = np.shape(text_train)[-1]

        #Load pre-trained embeddings (thanks Google)
        w2v = KeyedVectors.load_word2vec_format('/home/hmgent2/Data/GoogleNews-vectors-negative300.bin', binary=True)

        embeddings = np.zeros((len(tokenizer.word_index)+1, 300))
        for word, i in tokenizer.word_index.items():
            if word in w2v:
                embeddings[i] = w2v[word]

        #input layers
        text = layers.Input(shape=(input_dim))
        acoustic = layers.Input(shape=(acoustic_train.shape[1], acoustic_train.shape[2]))

        embedded = layers.Embedding(input_dim=embeddings.shape[0], output_dim=embeddings.shape[1], input_length=input_dim, weights=[embeddings], trainable=False)(text)
        embedded = layers.GlobalMaxPooling1D()(embedded)
        embedded = layers.Dense(300, activation='relu')(embedded)

        lstm = layers.LSTM(300, activation='relu', dropout=0.05, recurrent_dropout=0.2)(acoustic)
        lstm = layers.Dense(300, activation='relu')(lstm)

        fused = layers.Concatenate()([embedded, lstm])
        
        dropout = layers.Dropout(0.5)(fused)
        x = keras.Model(inputs=[text,acoustic], outputs=dropout)
            
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


    def test(self, inputs = None):

        if inputs == None:
            test_preds = self.model.predict(self.test_input)
        else:
            test_preds = self.model.predict(inputs)

        return test_preds         