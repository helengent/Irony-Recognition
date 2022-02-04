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

    model.compile(loss='sparse_categorical_crossentropy', optimizer=hp.Choice("optimizer", ['adam', 'sgd']), metrics=['accuracy'])\
    
    return model


class CVTuner(kt.engine.tuner.Tuner):
  def run_trial(self, trial, x, y, batch_size=32, epochs=1):
    val_losses = []
    for train, test in zip(x, y):
        x_train, x_test = train
        y_train, y_test = test
        model = self.hypermodel.build(trial.hyperparameters)
        model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs)
        val_losses.append(model.evaluate(x_test, y_test))
    self.oracle.update_trial(trial.trial_id, {'val_loss': np.mean(val_losses)})
    self.save_model(trial.trial_id, model)


def main(fileMod, dataPath, inputType, measureList, speakerSplit):

    global_acoustic, sequential_acoustic, text = inputType

    X_train, X_dev, X_test = list(), list(), list()

    #Create tokenizer
    fileList = glob("../../AudioData/Gated{}/*.wav".format(fileMod))
    baseList = np.array([item.split("/")[-1][:-4] for item in fileList])
    global TOKENIZER
    TOKENIZER = makeTokenizer(fileMod, baseList)

    #Load data
    # if text:
    #     text_train = np.load("{}/Text/oldSplit/speaker-independent_train-c_tokenized.npy".format(dataPath))
    #     text_dev = np.load("{}/Text/oldSplit/speaker-independent_dev-c_tokenized.npy".format(dataPath))
    #     text_test = np.load("{}/Text/oldSplit/speaker-independent_test-c_tokenized.npy".format(dataPath))

    #     X_train.append(text_train)
    #     X_dev.append(text_dev)
    #     X_test.append(text_test)

    #     y_train = np.load("{}/Text/oldSplit/speaker-independent_train-c_labels.npy".format(dataPath))
    #     y_dev = np.load("{}/Text/oldSplit/speaker-independent_dev-c_labels.npy".format(dataPath))
    #     y_test = np.load("{}/Text/oldSplit/speaker-independent_test-c_labels.npy".format(dataPath))

    # if sequential_acoustic:
    #     seq_train = np.load("{}/{}/oldSplit-{}/speaker-independent_train-c_acoustic.npy".format(dataPath, sequential_acoustic, "-".join(measureList)))
    #     seq_dev = np.load("{}/{}/oldSplit-{}/speaker-independent_dev-c_acoustic.npy".format(dataPath, sequential_acoustic, "-".join(measureList)))
    #     seq_test = np.load("{}/{}/oldSplit-{}/speaker-independent_test-c_acoustic.npy".format(dataPath, sequential_acoustic, "-".join(measureList)))

    #     X_train.append(seq_train)
    #     X_dev.append(seq_dev)
    #     X_test.append(seq_test)

    #     y_train = np.load("{}/{}/oldSplit-{}/speaker-independent_train-c_labels.npy".format(dataPath, sequential_acoustic, "-".join(measureList)))
    #     y_dev = np.load("{}/{}/oldSplit-{}/speaker-independent_dev-c_labels.npy".format(dataPath, sequential_acoustic, "-".join(measureList)))
    #     y_test = np.load("{}/{}/oldSplit-{}/speaker-independent_test-c_labels.npy".format(dataPath, sequential_acoustic, "-".join(measureList)))

    # if global_acoustic:
    #     glob_train = np.load("{}/{}/oldSplit-{}/speaker-independent_train-c_acoustic.npy".format(dataPath, global_acoustic, "-".join(measureList)))
    #     glob_dev = np.load("{}/{}/oldSplit-{}/speaker-independent_dev-c_acoustic.npy".format(dataPath, global_acoustic, "-".join(measureList)))
    #     glob_test = np.load("{}/{}/oldSplit-{}/speaker-independent_test-c_acoustic.npy".format(dataPath, global_acoustic, "-".join(measureList)))

    #     X_train.append(glob_train)
    #     X_dev.append(glob_dev)
    #     X_test.append(glob_test)

    #     y_train = np.load("{}/{}/oldSplit-{}/speaker-independent_train-c_labels.npy".format(dataPath, global_acoustic, "-".join(measureList)))
    #     y_dev = np.load("{}/{}/oldSplit-{}/speaker-independent_dev-c_labels.npy".format(dataPath, global_acoustic, "-".join(measureList)))
    #     y_test = np.load("{}/{}/oldSplit-{}/speaker-independent_test-c_labels.npy".format(dataPath, global_acoustic, "-".join(measureList)))


    # if sequential_acoustic and global_acoustic and text and speakerSplit == "dependent":
    #     tuner = kt.RandomSearch(buildModel_LSTM_FFNN_CNN_withText_speakerDep, objective='val_accuracy', max_trials=500, directory="Results", project_name="Dep_PCs_f0-hnr-mfcc")
    # elif sequential_acoustic and global_acoustic and text and speakerSplit == "independent":
    #     tuner = kt.RandomSearch(buildModel_LSTM_FFNN_CNN_withText_speakerInd, objective='val_accuracy', max_trials=500, directory="Results", project_name="newSplit_PCs-feats_f0-mfcc-plp")

    es = keras.callbacks.EarlyStopping(monitor='loss', patience=3)
    # tuner.search(X_train, y_train, epochs=100, validation_data=(X_dev, y_dev), callbacks=[es])

    x, y = list(), list()

    if speakerSplit == "dependent":
        counters = [0, 1, 2, 3, 4]
        subDir = "speakerDependent-{}".format("-".join(measureList))
        textSubDir = "speakerDependent"
    elif speakerSplit == "independent":
        counters = ["c", "d", "ejou", "fhkqst"]
        subDir = "newSplit-{}".format("-".join(measureList))
        textSubDir = "newSplit"
    
    for c in counters:
        train_seq = np.load("{}/{}/{}/speaker-{}_train-{}_acoustic.npy".format(dataPath, sequential_acoustic, subDir, speakerSplit, c))
        train_glob = np.load("{}/{}/{}/speaker-{}_train-{}_acoustic.npy".format(dataPath, global_acoustic, subDir, speakerSplit, c))
        train_text = np.load("{}/Text/{}/speaker-{}_train-{}_tokenized.npy".format(dataPath, textSubDir, speakerSplit, c))
        train_data = [train_text, train_seq, train_glob]

        dev_seq = np.load("{}/{}/{}/speaker-{}_dev-{}_acoustic.npy".format(dataPath, sequential_acoustic, subDir, speakerSplit, c))
        dev_glob = np.load("{}/{}/{}/speaker-{}_dev-{}_acoustic.npy".format(dataPath, global_acoustic, subDir, speakerSplit, c))
        dev_text = np.load("{}/Text/{}/speaker-{}_dev-{}_tokenized.npy".format(dataPath, textSubDir, speakerSplit, c))  
        dev_data = [dev_text, dev_seq, dev_glob]
        x.append((train_data, dev_data))
        train_labs = np.load("{}/{}/{}/speaker-{}_train-{}_labels.npy".format(dataPath, sequential_acoustic, subDir, speakerSplit, c))
        dev_labs = np.load("{}/{}/{}/speaker-{}_dev-{}_labels.npy".format(dataPath, sequential_acoustic, subDir, speakerSplit, c))
        y.append((train_labs, dev_labs))


    if speakerSplit == "dependent":
        my_build_model = buildModel_LSTM_FFNN_CNN_withText_speakerDep
    elif speakerSplit == "independent":
        my_build_model = buildModel_LSTM_FFNN_CNN_withText_speakerInd

    tuner = CVTuner(hypermodel=my_build_model,
        oracle=kt.oracles.RandomSearchOracle(
        objective='val_loss',
        max_trials=500))
        
    tuner.search(x, y, batch_size=64, epochs=30)

    tuner.results_summary()

    # best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]

    # model = tuner.hypermodel.build(best_hps)
    # history = model.fit(X_train, y_train, epochs=150, validation_data = (X_dev, y_dev))
    # val_acc_per_epoch = history.history["val_accuracy"]
    # best_epoch = val_acc_per_epoch.index(max(val_acc_per_epoch)) + 1
    # print("Best epoch number: {}".format(best_epoch))

    # model = model = tuner.hypermodel.build(best_hps)
    # model.fit(X_train, y_train, epochs=best_epoch, validation_data = (X_dev, y_dev))

    # eval_result = model.evaluate(X_test, y_test)
    # print("Test loss: {}\tTest accuracy: {}".format(eval_result[0], eval_result[1]))


if __name__=="__main__":

    fileMod = "Pruned3"

    dataPath = "/home/hmgent2/Data/ModelInputs"
    speakerSplits = ["dependent", "independent"]
    inputTypes = [("PCs", "percentChunks", True), ("2PCs_feats", "percentChunks", True)]
    measureLists = [["f0", "hnr", "mfcc"], ["f0", "mfcc", "plp"]]

    i = 0
    for inputType, measureList, speakerSplit in zip(inputTypes, measureLists, speakerSplits):
        if i != 0:

            print("Beginning hyperparameter tuning for the speaker {} model".format(speakerSplit))
            main(fileMod, dataPath, inputType, measureList, speakerSplit)
            print("Successfully completed speaker {} hyperparameter tuning".format(speakerSplit))
        i += 1