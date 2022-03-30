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
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.preprocessing.text import Tokenizer

import keras_tuner as kt  
from gensim.models import KeyedVectors
from tensorflow.keras.preprocessing.text import Tokenizer
import subprocess
import sys
import pickle

from sklearn.model_selection import train_test_split, StratifiedKFold
sys.path.append("../../AcousticFeatureExtraction")
from speaker import Speaker
from sd import sd



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


def buildModel_LSTM_FFNN_CNN_withText_speakerDep(hp):

    #Load pre-trained embeddings (thanks Google)
    w2v = KeyedVectors.load_word2vec_format('/home/hmgent2/Data/GoogleNews-vectors-negative300.bin', binary=True)

    embeddings = np.zeros((len(TOKENIZER.word_index)+1, 300))
    for word, i in TOKENIZER.word_index.items():
        if word in w2v:
            embeddings[i] = w2v[word]

    #input layers (shapes based on knowledge of input shapes)
    text = layers.Input(shape=(25))
    seq_acoustic = layers.Input(shape=(10, 30))
    glob_acoustic = layers.Input(shape=3)

    embedded = layers.Embedding(input_dim=embeddings.shape[0], output_dim=embeddings.shape[1], input_length=25, weights=[embeddings], trainable=False)(text)

    embeddingConvs = list()
    for p_size in range(hp.Int("conv_layers", 1, 5, step=1)):
        conv = layers.Conv1D(hp.Int("conv{}_size".format(p_size), 16, 264, step=8), hp.Int("pooling{}_size".format(p_size), 3, 10, step=1), activation='relu')(embedded)
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

    model.compile(loss='sparse_categorical_crossentropy', optimizer=hp.Choice("optimizer", ['adam', 'sgd']), metrics=['accuracy'])
    
    return model


def buildModel_LSTM_FFNN_CNN_withText_speakerInd(hp):

    #Load pre-trained embeddings (thanks Google)
    w2v = KeyedVectors.load_word2vec_format('/home/hmgent2/Data/GoogleNews-vectors-negative300.bin', binary=True)

    embeddings = np.zeros((len(TOKENIZER.word_index)+1, 300))
    for word, i in TOKENIZER.word_index.items():
        if word in w2v:
            embeddings[i] = w2v[word]

    #input layers (shapes based on knowledge of input shapes)
    text = layers.Input(shape=(25))
    seq_acoustic = layers.Input(shape=(10, 46))
    glob_acoustic = layers.Input(shape=11)

    embedded = layers.Embedding(input_dim=embeddings.shape[0], output_dim=embeddings.shape[1], input_length=25, weights=[embeddings], trainable=False)(text)

    embeddingConvs = list()
    for p_size in range(hp.Int("conv_layers", 1, 5, step=1)):
        conv = layers.Conv1D(hp.Int("conv{}_size".format(p_size), 16, 264, step=8), hp.Int("pooling{}_size".format(p_size), 3, 10, step=1), activation='relu')(embedded)
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

    model.compile(loss='sparse_categorical_crossentropy', optimizer=hp.Choice("optimizer", ['adam', 'sgd']), metrics=['accuracy'])\
    
    return model


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
    glob_acoustic = layers.Input(shape=30)

    embedded = layers.Embedding(input_dim=embeddings.shape[0], output_dim=embeddings.shape[1], input_length=25, weights=[embeddings], trainable=False)(text)

    embeddingConvs = list()
    for p_size in range(hp.Int("conv_layers", 1, 5, step=1)):
        conv = layers.Conv1D(hp.Int("conv{}_size".format(p_size), 16, 264, step=8), hp.Int("pooling{}_size".format(p_size), 3, 10, step=1), activation='relu')(embedded)
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

    model.compile(loss='sparse_categorical_crossentropy', optimizer=hp.Choice("optimizer", ['adam', 'sgd']), metrics=['accuracy'])
    
    return model


def buildModel_LSTM_CNN_withText(hp):

    #Load pre-trained embeddings (thanks Google)
    w2v = KeyedVectors.load_word2vec_format('/home/hmgent2/Data/GoogleNews-vectors-negative300.bin', binary=True)

    embeddings = np.zeros((len(TOKENIZER.word_index)+1, 300))
    for word, i in TOKENIZER.word_index.items():
        if word in w2v:
            embeddings[i] = w2v[word]

    #input layers (shapes based on knowledge of input shapes)
    text = layers.Input(shape=(25))
    seq_acoustic = layers.Input(shape=(10, 48))

    embedded = layers.Embedding(input_dim=embeddings.shape[0], output_dim=embeddings.shape[1], input_length=25, weights=[embeddings], trainable=False)(text)

    embeddingConvs = list()
    for p_size in range(hp.Int("conv_layers", 1, 5, step=1)):
        conv = layers.Conv1D(hp.Int("conv{}_size".format(p_size), 16, 264, step=8), hp.Int("pooling{}_size".format(p_size), 3, 10, step=1), activation='relu')(embedded)
        conv = layers.GlobalMaxPooling1D()(conv)
        embeddingConvs.append(conv)

    uniSize = hp.Int("concat_size", 16, 400, step=4)

    embedded = layers.Concatenate()(embeddingConvs)
    embedded = layers.Dense(uniSize, activation='relu')(embedded)

    lstm = layers.LSTM(hp.Int("lstm_size", 16, 400, step=4), activation='relu', dropout=hp.Float("lstm_dropout", min_value=0.0, max_value=0.9, step=0.05), recurrent_dropout=hp.Float("recurrent_dropout", min_value=0.0, max_value=0.5, step=0.05))(seq_acoustic)
    lstm = layers.Dense(uniSize, activation='relu')(lstm)

    fused = layers.Concatenate()([embedded, lstm])
    
    dropout = layers.Dropout(hp.Float("dropout", min_value=0.0, max_value=0.9, step=0.05))(fused)
    x = keras.Model(inputs=[text, seq_acoustic], outputs=dropout)
        
    z = layers.Dense(hp.Int("z1_size", 4, 32, step=2), activation='relu')(x.output)
    z = layers.Dense(2, activation=hp.Choice("outActivation", ["sigmoid", "softmax"]))(z)

    model = keras.Model(inputs=x.input, outputs=z)

    model.compile(loss='sparse_categorical_crossentropy', optimizer=hp.Choice("optimizer", ['adam', 'sgd']), metrics=['accuracy'])
    
    return model


class CVTuner(kt.engine.tuner.Tuner):
  def run_trial(self, trial, x, y, batch_size=32, epochs=1):
    val_acc = []
    for train, test in zip(x, y):
        x_train, x_test = train
        y_train, y_test = test
        model = self.hypermodel.build(trial.hyperparameters)
        model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs)
        val_acc.append(model.evaluate(x_test, y_test, return_dict=True)['accuracy'])
    self.oracle.update_trial(trial.trial_id, {'val_acc': np.mean(val_acc)})
    self.save_model(trial.trial_id, model)


def scaleIt(bigList, seq_scaler, seq_imputer):

    chunkSize = np.shape(bigList[0])[0]
    biggestFriend = np.zeros((len(bigList), chunkSize, np.shape(bigList[0])[1]))
    longLad = list()

    for utt in bigList:
        for chunk in utt:
            longLad.append(chunk)
    
    longLad = np.array(longLad)
    if np.isnan(np.sum(longLad)):
        longLad = seq_imputer.transform(longLad)
    scaled = seq_scaler.transform(longLad)

    start = 0
    stop = chunkSize
    for i in range(len(bigList)):
        small = scaled[start:stop]
        biggestFriend[i, :, :] = small
        start += chunkSize
        stop += chunkSize

    return biggestFriend


def chunkStats(x):
    stats = list()
    numChunks = int(100/10)
    chunkSize = int(len(x)/numChunks)
    start = 0

    for n in range(numChunks):
        chunk = x[start:start+chunkSize]
        chunkMean = np.mean(chunk)
        chunkSD = sd(chunk, chunkMean)
        stats.append([chunkMean, chunkSD])
        start += chunkSize
    
    return stats


def Hz2Mels(value):
    return (1/np.log(2)) * (np.log(1 + (value/1000))) * 1000


#   Assembles a numpy array with all measures from all files of shape (n, x, y) where
#       n = the length of fileList
#       x = frame_max (the number of frames padded/truncated to per file)
#       y = the number of acoustic measures (possibly multiple per item in measureList e.g. ams=375)
#   the speaker variable is the speaker left OUT of the training and dev data
def assembleArray(smallList, measureList):

    unscaled = list()

    for fileName in smallList:

        speaker = fileName.split("_")[1][0]

        s = Speaker(speaker, "/home/hmgent2/Data/AcousticData/SpeakerMetaData/{}_f0.txt".format(speaker.upper()), "/home/hmgent2/Data/AcousticData/SpeakerMetaData/{}_avgDur.txt".format(speaker.upper()))

        toStack = list()

        if "f0" in measureList:
            f0 = pd.read_csv("/home/hmgent2/Data/AcousticData/f0/{}.csv".format(fileName))["0"].tolist()
            f0 = Hz2Mels(np.array(f0))
            f0 = chunkStats(f0)
            toStack.append(f0)

        if "hnr" in measureList:
            hnr = np.array(pd.read_csv("/home/hmgent2/Data/AcousticData/hnr/{}.csv".format(fileName))["0"].tolist())
            hnr = chunkStats(hnr)
            toStack.append(hnr)

        if "mfcc" in measureList:
            for i in range(13):
                mfcc = np.array(pd.read_csv("/home/hmgent2/Data/AcousticData/mfcc/{}.csv".format(fileName))[str(i)].tolist())
                mfcc = chunkStats(mfcc)
                toStack.append(mfcc)

        if "plp" in measureList:
            for i in range(9):
                plp = np.array(pd.read_csv("/home/hmgent2/Data/AcousticData/plp/{}.csv".format(fileName), header=None)[i].tolist())
                plp = chunkStats(plp)
                toStack.append(plp)

        assert len(list(set([len(item) for item in toStack]))) == 1

        wholeFriend = np.hstack(toStack)

        unscaled.append(wholeFriend)

    return unscaled


def transformLabs(x_list):
    new_list = list()
    for x in x_list:
        if type(x) == str or type(x) == np.str_:
            if x.upper() == "I":
                new_list.append(1.0)
            elif x.upper() == "N":
                new_list.append(0.0)
            else:
                print("Invalid label")
                raise Exception
        else:
            new_list.append(x)
    return new_list


def prepareData(speakerSplit, speakerList, fileList, inputType, subDir, fileMod, measureList):

    train_list, dev_list = list(), list()
    glob_acoustic, seq_acoustic, text = inputType

    if glob_acoustic:
        glob_file = pd.read_csv("{}/{}/all_inputs.csv".format(dataPath, glob_acoustic))
    with open("/home/hmgent2/Data/ModelInputs/{}/{}/scaler.pkl".format(seq_acoustic, subDir), "rb") as f:
        seq_scaler = pickle.load(f)
    with open("/home/hmgent2/Data/ModelInputs/{}/{}/imputer.pkl".format(seq_acoustic, subDir), "rb") as f:
        seq_imputer = pickle.load(f)

    if speakerSplit == "independent":
        for speaker in speakerList:

            tr = [item for item in fileList if item.split("_")[1][0] not in speaker]
            d = [item for item in fileList if item.split("_")[1][0] in speaker]

            tr_labs = [item[-1] for item in tr]
            d_labs = [item[-1] for item in d]

            train_list.append((tr, tr_labs, speaker))
            dev_list.append((d, d_labs, speaker))
        
    else:
        #Do 5-fold cross validation
        lab_list = np.array([item[-1] for item in fileList])

        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=6)
        count = 0
        for train_index, test_index in skf.split(fileList, lab_list):
            tr = np.array(fileList)[train_index]   
            d = np.array(fileList)[test_index]
                        
            tr_labs = [item[-1] for item in tr]
            d_labs = [item[-1] for item in d]

            train_list.append((tr, tr_labs, count))
            dev_list.append((d, d_labs, count))
            count += 1
        
    for train, dev in zip(train_list, dev_list):
    
        X_train, y_train, counter = train
        X_dev, y_dev, counter = dev

        if seq_acoustic:

            if not os.path.exists("{}/{}/{}/speaker-{}_train-{}_acoustic.npy".format(dataPath, seq_acoustic, subDir, speakerSplit, counter)):
                X_train = assembleArray(X_train, measureList)
                X_train = scaleIt(X_train, seq_scaler, seq_imputer)
                with open("{}/{}/{}/speaker-{}_train-{}_acoustic.npy".format(dataPath, seq_acoustic, subDir, speakerSplit, counter), "wb") as f:
                    np.save(f, X_train)

            if not os.path.exists("{}/{}/{}/speaker-{}_train-{}_labels.npy".format(dataPath, seq_acoustic, subDir, speakerSplit, counter)):
                y_train = np.array(transformLabs(y_train))
                with open("{}/{}/{}/speaker-{}_train-{}_labels.npy".format(dataPath, seq_acoustic, subDir, speakerSplit, counter), "wb") as f:
                    np.save(f, y_train)

            if not os.path.exists("{}/{}/{}/speaker-{}_dev-{}_acoustic.npy".format(dataPath, seq_acoustic, subDir, speakerSplit, counter)):
                X_dev= assembleArray(X_dev, measureList)
                X_dev = scaleIt(X_dev, seq_scaler, seq_imputer)
                with open("{}/{}/{}/speaker-{}_dev-{}_acoustic.npy".format(dataPath, seq_acoustic, subDir, speakerSplit, counter), "wb") as f:
                    np.save(f, X_dev)

            if not os.path.exists("{}/{}/{}/speaker-{}_dev-{}_labels.npy".format(dataPath, seq_acoustic, subDir, speakerSplit, counter)):
                y_dev = np.array(transformLabs(y_dev))
                with open("{}/{}/{}/speaker-{}_dev-{}_labels.npy".format(dataPath, seq_acoustic, subDir, speakerSplit, counter), "wb") as f:
                    np.save(f, y_dev)

        X_train, y_train, counter = train
        X_dev, y_dev, counter = dev

        if glob_acoustic:

            if not os.path.exists("{}/{}/speaker-{}_train-{}_acoustic.npy".format(dataPath, glob_acoustic, speakerSplit, counter)):
                newX = pd.DataFrame()
                for name in X_train:
                    subset = glob_file[glob_file["fileName"] == name]
                    newX = newX.append(subset)
                X_train = newX
                train_meta = pd.DataFrame()
                train_meta["fileName"] = X_train.pop("fileName")
                train_meta["speaker"] = X_train.pop("speaker")
                train_meta["label"] = X_train.pop("label")
                train_meta.to_csv("{}/speaker-{}_train_{}.meta".format(dataPath, speakerSplit, counter))
                X_train = np.array(X_train)
                with open("{}/{}/speaker-{}_train-{}_acoustic.npy".format(dataPath, glob_acoustic, speakerSplit, counter), "wb") as f:
                    np.save(f, X_train)

            if not os.path.exists("{}/{}/speaker-{}_train-{}_labels.npy".format(dataPath, glob_acoustic, speakerSplit, counter)):
                y_train = np.array(transformLabs(y_train))
                with open("{}/{}/speaker-{}_train-{}_labels.npy".format(dataPath, glob_acoustic, speakerSplit, counter), "wb") as f:
                    np.save(f, y_train)

            if not os.path.exists("{}/{}/speaker-{}_dev-{}_acoustic.npy".format(dataPath, glob_acoustic, speakerSplit, counter)):
                newX = pd.DataFrame()
                for name in X_dev:
                    subset = glob_file[glob_file["fileName"] == name]
                    newX = newX.append(subset)
                X_dev = newX
                dev_meta = pd.DataFrame()
                dev_meta["fileName"] = X_dev.pop("fileName")
                dev_meta["speaker"] = X_dev.pop("speaker")
                dev_meta["label"] = X_dev.pop("label")
                dev_meta.to_csv("{}/speaker-{}_dev_{}.meta".format(dataPath, speakerSplit, counter))
                X_dev = np.array(X_dev)
                with open("{}/{}/speaker-{}_dev-{}_acoustic.npy".format(dataPath, glob_acoustic, speakerSplit, counter), "wb") as f:
                    np.save(f, X_dev)

            if not os.path.exists("{}/{}/speaker-{}_dev-{}_labels.npy".format(dataPath, glob_acoustic, speakerSplit, counter)):
                y_dev = np.array(transformLabs(y_dev))
                with open("{}/{}/speaker-{}_dev-{}_labels.npy".format(dataPath, glob_acoustic, speakerSplit, counter), "wb") as f:
                    np.save(f, y_dev)

        X_train, y_train, counter = train
        X_dev, y_dev, counter = dev

        if text:

            if not os.path.exists("{}/Text/speaker-{}_train-{}_tokenized.npy".format(dataPath, speakerSplit, counter)):
                textList = list()
                for f in X_train:
                    text = open("/home/hmgent2/Data/TextData/{}_asr/{}.txt".format(fileMod, f)).read()
                    textList.append(text)
                sequences = TOKENIZER.texts_to_sequences(textList)
                X_train = sequence.pad_sequences(sequences, padding="post", truncating="post", maxlen=25) #maxlen chosen as 95th percentile of sentence lengths
                with open("{}/Text/speaker-{}_train-{}_tokenized.npy".format(dataPath, speakerSplit, counter), "wb") as f:
                    np.save(f, X_train)

            if not os.path.exists("{}/Text/speaker-{}_train-{}_labels.npy".format(dataPath, speakerSplit, counter)):
                y_train = np.array(transformLabs(y_train))
                with open("{}/Text/speaker-{}_train-{}_labels.npy".format(dataPath, speakerSplit, counter), "wb") as f:
                    np.save(f, y_train)

            if not os.path.exists("{}/Text/speaker-{}_dev-{}_tokenized.npy".format(dataPath, speakerSplit, counter)):
                textList = list()
                for f in X_dev:
                    text = open("/home/hmgent2/Data/TextData/{}_asr/{}.txt".format(fileMod, f)).read()
                    textList.append(text)
                sequences = TOKENIZER.texts_to_sequences(textList)
                X_dev = sequence.pad_sequences(sequences, padding="post", truncating="post", maxlen=25) #maxlen chosen as 95th percentile of sentence lengths
                with open("{}/Text/speaker-{}_dev-{}_tokenized.npy".format(dataPath, speakerSplit, counter), "wb") as f:
                    np.save(f, X_dev)

            if not os.path.exists("{}/Text/speaker-{}_dev-{}_labels.npy".format(dataPath, speakerSplit, counter)):
                y_dev = np.array(transformLabs(y_dev))
                with open("{}/Text/speaker-{}_dev-{}_labels.npy".format(dataPath, speakerSplit, counter), "wb") as f:
                    np.save(f, y_dev)


def main(fileMod, dataPath, inputType, measureList, speakerSplit):

    global_acoustic, sequential_acoustic, text = inputType

    #Create tokenizer
    fileList = glob("../../AudioData/Gated{}/*.wav".format(fileMod))
    baseList = np.array([item.split("/")[-1][:-4] for item in fileList])
    global TOKENIZER
    TOKENIZER = makeTokenizer(fileMod, baseList)

    es = keras.callbacks.EarlyStopping(monitor='loss', patience=3)

    x, y = list(), list()

    if speakerSplit == "dependent":
        counters = [0, 1, 2, 3, 4]
        subDir = "speakerDependent-{}".format("-".join(measureList))
    elif speakerSplit == "independent":
        counters = ["c", "d"]
        subDir = "newSplit-{}".format("-".join(measureList))

    if not os.path.isdir("{}/{}/{}".format(dataPath, sequential_acoustic, subDir)):
        os.mkdir("{}/{}/{}".format(dataPath, sequential_acoustic, subDir))

    prepareData(speakerSplit, ["c", "d"], baseList, inputType, subDir, fileMod, measureList)
    
    for c in counters:
        train_seq = np.load("{}/{}/{}/speaker-{}_train-{}_acoustic.npy".format(dataPath, sequential_acoustic, subDir, speakerSplit, c))
        train_glob = np.load("{}/{}/speaker-{}_train-{}_acoustic.npy".format(dataPath, global_acoustic, speakerSplit, c))
        train_text = np.load("{}/Text/speaker-{}_train-{}_tokenized.npy".format(dataPath, speakerSplit, c))
        train_data = [train_text, train_seq, train_glob]

        dev_seq = np.load("{}/{}/{}/speaker-{}_dev-{}_acoustic.npy".format(dataPath, sequential_acoustic, subDir, speakerSplit, c))
        dev_glob = np.load("{}/{}/speaker-{}_dev-{}_acoustic.npy".format(dataPath, global_acoustic, speakerSplit, c))
        dev_text = np.load("{}/Text/speaker-{}_dev-{}_tokenized.npy".format(dataPath, speakerSplit, c))  
        dev_data = [dev_text, dev_seq, dev_glob]

        x.append((train_data, dev_data))
        train_labs = np.load("{}/{}/{}/speaker-{}_train-{}_labels.npy".format(dataPath, sequential_acoustic, subDir, speakerSplit, c))
        dev_labs = np.load("{}/{}/{}/speaker-{}_dev-{}_labels.npy".format(dataPath, sequential_acoustic, subDir, speakerSplit, c))
        y.append((train_labs, dev_labs))

    # my_build_model = buildModel_LSTM_FFNN_CNN_withText
    # my_build_model = buildModel_LSTM_CNN_withText

    if speakerSplit == "dependent":
        my_build_model = buildModel_LSTM_FFNN_CNN_withText_speakerDep
    elif speakerSplit == "independent":
        my_build_model = buildModel_LSTM_FFNN_CNN_withText_speakerInd

    tuner = CVTuner(hypermodel=my_build_model,
        oracle=kt.oracles.RandomSearchOracle(
        objective='val_acc',
        max_trials=400))
        
    tuner.search(x, y, batch_size=64, epochs=30)

    tuner.results_summary()


if __name__=="__main__":

    fileMod = "newTest"

    dataPath = "/home/hmgent2/Data/newTest_ModelInputs"
    # speakerSplits = ["dependent", "independent"]

    # inputTypes = [("PCs", "percentChunks", True), ("2PCs_feats", "percentChunks", True)]
    # measureLists = [["f0", "hnr", "mfcc"], ["f0", "mfcc", "plp"]]

    speakerSplits = ["independent"]
    inputTypes = [("2PCs_feats", "percentChunks", True)]
    measureLists = [["f0", "mfcc", "plp"]]

    i = 0
    for inputType, measureList, speakerSplit in zip(inputTypes, measureLists, speakerSplits):

        if os.path.isdir("untitled_project"):
            print("trial directory not successfully cleared")
            sys.exit()

        print("Beginning hyperparameter tuning for the speaker {} model".format(speakerSplit))
        main(fileMod, dataPath, inputType, measureList, speakerSplit)
        print("Successfully completed speaker {} hyperparameter tuning".format(speakerSplit))
        
        if speakerSplit == "dependent":
            bashCommand = "mv untitled_project speakerDependent"
            subprocess.run(bashCommand, shell=True)
        elif speakerSplit == "independent":
            bashCommand = "mv untitled_project speakerIndependent"
            subprocess.run(bashCommand, shell=True)

        i += 1
        
