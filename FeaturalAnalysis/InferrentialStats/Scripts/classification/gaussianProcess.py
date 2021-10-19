#!/usr/bin/env python3

from classificationValidation import ClassificationValidator
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF

def validate(clf, X_test, y_true, labelClasses):

    return ClassificationValidator("Gaussian Process", clf, X_test, y_true, labelClasses)


def model(X_train, y_train):

    return GaussianProcessClassifier(1.0 * RBF(1.0)).fit(X_train, y_train)


def main(X_train, y_train, X_test, y_test, labelClasses):

    # Construct the model
    clf = model(X_train, y_train)

    # Validate the model
    scores = validate(clf, X_test, y_test, labelClasses)

    return scores

