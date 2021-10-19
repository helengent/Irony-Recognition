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
from sklearn.model_selection import train_test_split


def writeReport(classifiers, exp_dir):

    scoreList = list()
    for clf in classifiers:
        labels, scores = clf._make_report()
        scoreList.append(scores)

    df = pd.DataFrame(scoreList, columns = labels)
    # df.to_csv(os.path.join("../Data", exp_dir.split("/")[2], 'classifiers.csv'), index = False)
    df.to_csv("../Data/{}_classifiers.csv".format(exp_dir.split("/")[2]))


def getLabels(exp, data):

    y, files = list(), list()

    df = pd.read_csv(data, delimiter="\t")

    for _, row in df.iterrows():
        y.append(row.label)
        files.append(row.filename)

    return y, files


def getData(exp, data):

    y, files = getLabels(exp, data)

    # Encode from categorical to numerical
    le = LabelEncoder()
    le.fit(y)

    allArrays = [pd.read_csv("{}/{}.csv".format(exp, csv)).to_numpy().flatten() for csv in files]

    M = np.max([array.shape[0] for array in allArrays])
    paddedArrays = [np.pad(array, (0, M-array.shape[0]), 'constant') for array in allArrays]

    X = np.stack(paddedArrays)

    return X, le.transform(y), le


def makeCalls(exp_dir, data_dir):

    X, y, le = getData(exp_dir, data_dir)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=6, stratify=y)

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

    # exp_dir = "../Data/ams"
    # metaData = "../../../AudioData/metaData.txt"

    # makeCalls(exp_dir, metaData)

    main()

