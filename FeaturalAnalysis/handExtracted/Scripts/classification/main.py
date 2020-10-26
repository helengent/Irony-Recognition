#!/usr/bin/env python3

import os
import sys
import glob
import argparse
import numpy as np
import pandas as pd

import adaBoost
import gaussianNB
import decisionTree 
import randomForest
import gaussianProcess
import nearestNeighbors
import linearSupportVector
import multiLayerPerceptron
import linearDiscriminantAnalysis
from sklearn.preprocessing import LabelEncoder


def writeReport(classifiers, exp_dir):

    scoreList = list()
    for clf in classifiers:
        labels, scores = clf._make_report()
        scoreList.append(scores)

    df = pd.DataFrame(scoreList, columns = labels)
    df.to_csv(os.path.join(exp_dir, 'reports', 'classifiers.csv'), index = False)


def getLabels(exp, data, which):

    y, files = list(), list()
    allFiles = glob.glob(os.path.join(exp, 'data', which, 'csv/*'))

    for speaker in allFiles:
        # I wrote this blind without testing, so check it if things get weird
        label = os.path.basename(speaker).split('.')[0].split('_')[1]
        y.append(label)
        files.append(speaker)

    return y, files


def getData(exp, data):

    y_train, trainFiles = getLabels(exp, data, 'train')
    y_test, testFiles = getLabels(exp, data, 'dev')

    # Encode from categorical to numerical
    le = LabelEncoder()
    le.fit(y_train)

    trainArrays = [pd.read_csv(csv).to_numpy().flatten() for csv in trainFiles]
    testArrays = [pd.read_csv(csv).to_numpy().flatten() for csv in testFiles]
    allArrays = trainArrays + testArrays

    M = np.max([array.shape[0] for array in allArrays])
    paddedTrainArrays = [np.pad(array, (0, M-array.shape[0]), 'constant') for array in trainArrays]
    paddedTestArrays = [np.pad(array, (0, M-array.shape[0]), 'constant') for array in testArrays]

    X_train = np.stack(paddedTrainArrays)
    X_test = np.stack(paddedTestArrays)

    return X_train, le.transform(y_train), X_test, le.transform(y_test), le


def makeCalls(exp_dir, data_dir):

    X_train, y_train, X_test, y_test, le = getData(exp_dir, data_dir)

    classifiers = list()
    for module in [nearestNeighbors,
                   linearSupportVector,
                   gaussianProcess,
                   decisionTree,
                   randomForest,
                   multiLayerPerceptron,
                   adaBoost,
                   gaussianNB,
                   linearDiscriminantAnalysis
                   ]:

        clf = module.main(X_train, y_train, X_test, y_test, le.classes_)
        classifiers.append(clf)

    writeReport(classifiers, exp_dir)


def main():

    parser = argparse.ArgumentParser(description='Call all of the regression algorithms and consolidate a global report.')
    parser.add_argument('exp_dir', type=str, help='Temporary experiment directory.')
    parser.add_argument('data_dir', type=str, help='Location for the map for the scalar score (e.g. MMSE).')

    args = parser.parse_args()
    makeCalls(args.exp_dir, args.data_dir)


if __name__ == "__main__":

    main()

