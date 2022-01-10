#!/usr/bin/env python3

import os
from glob import glob
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
from gensim.models import KeyedVectors
from tensorflow.keras.preprocessing.text import Tokenizer


def makeTokenizer(fileMod, fileList):

        textList = list()

        for f in fileList:
            text = open("/home/hmgent2/Data/TextData/{}_asr/{}.txt".format(fileMod, f)).read()
            textList.append(text)

        tokenizer = Tokenizer(oov_token="UNK")
        tokenizer.fit_on_texts(textList)
        vocab_size = len(tokenizer.word_index) + 1
        print("VOCABULARY SIZE: {}".format(vocab_size))
        return tokenizer


def buildModel_LSTM_FFNN_CNN_withText(hp):

    #Load pre-trained embeddings (thanks Google)
    w2v = KeyedVectors.load_word2vec_format('/home/hmgent2/Data/GoogleNews-vectors-negative300.bin', binary=True)

    embeddings = np.zeros((len(TOKENIZER.word_index)+1, 300))
    for word, i in TOKENIZER.word_index.items():
        if word in w2v:
            embeddings[i] = w2v[word]

    #input layers (shapes based on knowledge of input shapes)
    text = layers.Input(shape=(25))
    seq_acoustic = layers.Input(shape=(10, 48))
    glob_acoustic = layers.Input(shape=17)

    embedded = layers.Embedding(input_dim=embeddings.shape[0], output_dim=embeddings.shape[1], input_length=25, weights=[embeddings], trainable=False)(text)

    embeddingConvs = list()
    pooling_size = 2
    for p_size in range(hp.Int("conv_layers", 1, 5, step=1)):
        pooling_size += 1
        conv = layers.Conv1D(128, pooling_size, activation='relu')(embedded)
        conv = layers.GlobalMaxPooling1D()(conv)
        embeddingConvs.append(conv)

    uniSize = hp.Int("concat_size", 16, 400, step=4)

    embedded = layers.Concatenate()(embeddingConvs)
    embedded = layers.Dense(uniSize, activation='relu')(embedded)

    lstm = layers.LSTM(hp.Int("lstm_size", 16, 400, step=4), activation='relu', dropout=hp.Float("lstm_dropout", min_value=0.0, max_value=0.9, step=0.05), recurrent_dropout=hp.Float("recurrent_dropout", min_value=0.0, max_value=0.5, step=0.05))(seq_acoustic)
    lstm = layers.Dense(uniSize, activation='relu')(lstm)

    ffnn = layers.Dense(uniSize)(glob_acoustic)

    fused = layers.Concatenate()([embedded, lstm, ffnn])
    
    dropout = layers.Dropout(hp.Float("dropout", min_value=0.0, max_value=0.9, step=0.05))(fused)
    x = keras.Model(inputs=[text, seq_acoustic, glob_acoustic], outputs=dropout)
        
    z = layers.Dense(hp.Int("z1_size", 4, 32, step=2), activation='relu')(x.output)
    z = layers.Dense(2, activation=hp.Choice("outActivation", ["sigmoid", "softmax"]))(z)

    model = keras.Model(inputs=x.input, outputs=z)
    model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    
    return model


def main(fileMod, dataPath, inputType, measureList):

    global_acoustic, sequential_acoustic, text = inputType

    X_train, X_dev, X_test = list(), list(), list()

    #Create tokenizer
    fileList = glob("../../AudioData/Gated{}/*.wav".format(fileMod))
    baseList = np.array([item.split("/")[-1][:-4] for item in fileList])
    global TOKENIZER
    TOKENIZER = makeTokenizer(fileMod, baseList)

    #Load data
    if text:
        text_train = np.load("{}/Text/speakerDependent/speaker-dependent_train-0_tokenized.npy".format(dataPath))
        text_dev = np.load("{}/Text/speakerDependent/speaker-dependent_dev-0_tokenized.npy".format(dataPath))
        text_test = np.load("{}/Text/speakerDependent/speaker-dependent_test-0_tokenized.npy".format(dataPath))

        X_train.append(text_train)
        X_dev.append(text_dev)
        X_test.append(text_test)

        y_train = np.load("{}/Text/speakerDependent/speaker-dependent_train-0_labels.npy".format(dataPath))
        y_dev = np.load("{}/Text/speakerDependent/speaker-dependent_dev-0_labels.npy".format(dataPath))
        y_test = np.load("{}/Text/speakerDependent/speaker-dependent_test-0_labels.npy".format(dataPath))

    if sequential_acoustic:
        seq_train = np.load("{}/{}/speakerDependent/speaker-dependent_train-0_acoustic.npy".format(dataPath, sequential_acoustic))
        seq_dev = np.load("{}/{}/speakerDependent/speaker-dependent_dev-0_acoustic.npy".format(dataPath, sequential_acoustic))
        seq_test = np.load("{}/{}/speakerDependent/speaker-dependent_test-0_acoustic.npy".format(dataPath, sequential_acoustic))

        X_train.append(seq_train)
        X_dev.append(seq_dev)
        X_test.append(seq_test)

        y_train = np.load("{}/{}/speakerDependent/speaker-dependent_train-0_labels.npy".format(dataPath, sequential_acoustic))
        y_dev = np.load("{}/{}/speakerDependent/speaker-dependent_dev-0_labels.npy".format(dataPath, sequential_acoustic))
        y_test = np.load("{}/{}/speakerDependent/speaker-dependent_test-0_labels.npy".format(dataPath, sequential_acoustic))

    if global_acoustic:
        glob_train = np.load("{}/{}/speakerDependent/speaker-dependent_train-0_acoustic.npy".format(dataPath, global_acoustic))
        glob_dev = np.load("{}/{}/speakerDependent/speaker-dependent_dev-0_acoustic.npy".format(dataPath, global_acoustic))
        glob_test = np.load("{}/{}/speakerDependent/speaker-dependent_test-0_acoustic.npy".format(dataPath, global_acoustic))

        X_train.append(glob_train)
        X_dev.append(glob_dev)
        X_test.append(glob_test)

        y_train = np.load("{}/{}/speakerDependent/speaker-dependent_train-0_labels.npy".format(dataPath, global_acoustic))
        y_dev = np.load("{}/{}/speakerDependent/speaker-dependent_dev-0_labels.npy".format(dataPath, global_acoustic))
        y_test = np.load("{}/{}/speakerDependent/speaker-dependent_test-0_labels.npy".format(dataPath, global_acoustic))


    if sequential_acoustic and global_acoustic and text:
        tuner = kt.RandomSearch(buildModel_LSTM_FFNN_CNN_withText, objective='val_accuracy', max_trials=500, directory="Results", project_name="trimodal")

    tuner.search(X_train, y_train, epochs=100, validation_data=(X_dev, y_dev))

    tuner.results_summary()


if __name__=="__main__":

    fileMod = "Pruned3"

    dataPath = "/home/hmgent2/Data/ModelInputs/"
    inputType = ("PCs_feats", "percentChunks", True)
    measureList = ["f0", "hnr", "mfcc", "plp"]

    main(fileMod, dataPath, inputType, measureList)