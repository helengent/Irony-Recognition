#!/usr/bin/env python3

import sys
import numpy as np
import pandas as pd
from glob import glob
from tensorflow.keras.preprocessing.text import Tokenizer

sys.path.append("../../Models")

# from LSTM_FFNN_CNN_withText import acousticTextLSTM_CNN_FFNN
from LSTM_FFNN_CNN_withText_TUNED import acousticTextLSTM_CNN_FFNN


class ModelTrainer:

    def __init__(self, fileMod, fileList, modelName, train_data, dev_data, test_data, outDir):
        self.fileMod = fileMod
        self.fileList = fileList
        self.modelName = modelName
        self.train_data = train_data
        self.dev_data = dev_data
        self.test_data = test_data
        self.outDir = outDir
        self.class_weights = {0.0: 1.0, 1.0: 1.0}

        self.tokenizer = self.makeTokenizer()

        self.loadData()


    def makeTokenizer(self):

        textList = list()

        for f in self.fileList:
            text = open("/home/hmgent2/Data/TextData/{}_asr/{}.txt".format(self.fileMod, f)).read()
            textList.append(text)

        tokenizer = Tokenizer(oov_token="UNK")
        tokenizer.fit_on_texts(textList)
        vocab_size = len(tokenizer.word_index) + 1
        print("VOCABULARY SIZE: {}".format(vocab_size))
        return tokenizer


    def loadData(self):

        train_glob, train_seq, train_text, train_labs = self.train_data
        dev_glob, dev_seq, dev_text, dev_labs = self.dev_data
        test_glob, test_seq, test_text, test_labs = self.test_data

        self.train_glob = np.load(train_glob)
        self.train_seq = np.load(train_seq)
        self.train_text = np.load(train_text)
        self.train_labs = np.load(train_labs)

        assert self.train_glob.shape[0] == self.train_seq.shape[0] == self.train_text.shape[0] == self.train_labs.shape[0]

        self.dev_glob = np.load(dev_glob)
        self.dev_seq = np.load(dev_seq)
        self.dev_text = np.load(dev_text)
        self.dev_labs = np.load(dev_labs)

        assert self.dev_glob.shape[0] == self.dev_seq.shape[0] == self.dev_text.shape[0] == self.dev_labs.shape[0]

        self.test_glob = np.load(test_glob)
        self.test_seq = np.load(test_seq)
        self.test_text = np.load(test_text)
        self.test_labs = np.load(test_labs)

        assert self.test_glob.shape[0] == self.test_seq.shape[0] == self.test_text.shape[0] == self.test_labs.shape[0]


    def trainModel(self):

        self.model = acousticTextLSTM_CNN_FFNN(self.train_seq, self.dev_seq, self.test_seq, 
                                                self.train_glob, self.dev_glob, self.test_glob,
                                                self.train_text, self.dev_text, self.test_text, 
                                                self.train_labs, self.dev_labs, self.test_labs,
                                                self.tokenizer, "{}/{}_checkpoints.csv".format(self.outDir, self.modelName), 
                                                "{}/{}_checkpoints.ckpt".format(self.outDir, self.modelName), 
                                                "{}/{}_history.png".format(self.outDir, self.modelName), self.class_weights)

        self.model.train()


if __name__=="__main__":

    fileMod = "Pruned3"
    fileList = glob("../../AudioData/Gated{}/*.wav".format(fileMod))
    fileList = np.array([item.split("/")[-1][:-4] for item in fileList])
    
    speakerSplit = "independent"

    if speakerSplit == "dependent":
        modelName = "TUNED_speakerDependent_final"

        train_seq = "/home/hmgent2/Data/ModelInputs/percentChunks/speakerDependent-f0-hnr-mfcc/speaker-dependent_train-unified_acoustic.npy"
        train_glob = "/home/hmgent2/Data/ModelInputs/PCs/speakerDependent-f0-hnr-mfcc/speaker-dependent_train-unified_acoustic.npy"
        train_text = "/home/hmgent2/Data/ModelInputs/Text/speakerDependent/speaker-dependent_train-unified_tokenized.npy"
        train_labs = "/home/hmgent2/Data/ModelInputs/percentChunks/speakerDependent-f0-hnr-mfcc/speaker-dependent_train-unified_labels.npy"
        train_data = (train_glob, train_seq, train_text, train_labs)

        dev_seq = "/home/hmgent2/Data/ModelInputs/percentChunks/speakerDependent-f0-hnr-mfcc/speaker-dependent_dev-unified_acoustic.npy"
        dev_glob = "/home/hmgent2/Data/ModelInputs/PCs/speakerDependent-f0-hnr-mfcc/speaker-dependent_dev-unified_acoustic.npy"
        dev_text = "/home/hmgent2/Data/ModelInputs/Text/speakerDependent/speaker-dependent_dev-unified_tokenized.npy"
        dev_labs = "/home/hmgent2/Data/ModelInputs/percentChunks/speakerDependent-f0-hnr-mfcc/speaker-dependent_dev-unified_labels.npy"
        dev_data = (dev_glob, dev_seq, dev_text, dev_labs)

        test_seq = "/home/hmgent2/Data/newTest_ModelInputs/percentChunks/speakerDependent-f0-hnr-mfcc/speaker-dependent_test-0_acoustic.npy"
        test_glob = "/home/hmgent2/Data/newTest_ModelInputs/PCs/speakerDependent-f0-hnr-mfcc/speaker-dependent_test-0_acoustic.npy"
        test_text = "/home/hmgent2/Data/newTest_ModelInputs/Text/speakerDependent/speaker-dependent_test-0_tokenized.npy"
        test_labs = "/home/hmgent2/Data/newTest_ModelInputs/percentChunks/speakerDependent-f0-hnr-mfcc/speaker-dependent_test-0_labels.npy"
        test_data = (test_glob, test_seq, test_text, test_labs)
        outDir = "Tuned/SpeakerDependent"


    #NOPE This won't work. You  don't have any additional speaker independent data to give it other than 
    #What you already have. Just use the other script because there's no need for this.
    # elif speakerSplit == "independent":
    #     modelName = "speakerIndependent_final"

    #     train_seq = "/home/hmgent2/Data/ModelInputs/percentChunks/oldSplit-f0-mfcc-plp/speaker-independent_train-unified_acoustic.npy"
    #     train_glob = "/home/hmgent2/Data/ModelInputs/PCs/oldSplit-f0-mfcc-plp/speaker-independent_train-unified_acoustic.npy"
    #     train_text = "/home/hmgent2/Data/ModelInputs/Text/oldSplit/speaker-independent_train-unified_tokenized.npy"
    #     train_labs = "/home/hmgent2/Data/ModelInputs/percentChunks/oldSplit-f0-mfcc-plp/speaker-independent_train-unified_labels.npy"
    #     train_data = (train_glob, train_seq, train_text, train_labs)

    #     dev_seq = "/home/hmgent2/Data/ModelInputs/percentChunks/oldSplit-f0-mfcc-plp/speaker-independent_dev-unified_acoustic.npy"
    #     dev_glob = "/home/hmgent2/Data/ModelInputs/PCs/oldSplit-f0-mfcc-plp/speaker-independent_dev-unified_acoustic.npy"
    #     dev_text = "/home/hmgent2/Data/ModelInputs/Text/oldSplit/speaker-independent_dev-unified_tokenized.npy"
    #     dev_labs = "/home/hmgent2/Data/ModelInputs/percentChunks/oldSplit-f0-mfcc-plp/speaker-dependent_dev-unified_labels.npy"
    #     dev_data = (dev_glob, dev_seq, dev_text, dev_labs)

    #     test_seq = "/home/hmgent2/Data/newTest_ModelInputs/percentChunks/oldSplit-f0-mfcc-plp/speaker-independent_test-c_acoustic.npy"
    #     test_glob = "/home/hmgent2/Data/newTest_ModelInputs/PCs/oldSplit-f0-mfcc-plp/speaker-independent_test-c_acoustic.npy"
    #     test_text = "/home/hmgent2/Data/newTest_ModelInputs/Text/oldSplit/speaker-independent_test-c_tokenized.npy"
    #     test_labs = "/home/hmgent2/Data/newTest_ModelInputs/percentChunks/oldSplit-f0-mfcc-plp/speaker-independent_test-c_labels.npy"
    #     test_data = (test_glob, test_seq, test_text, test_labs)
    #     outDir = "Tuned/SpeakerIndependent"

    trainer = ModelTrainer(fileMod, fileList, modelName, train_data, dev_data, test_data, outDir)
    trainer.trainModel()