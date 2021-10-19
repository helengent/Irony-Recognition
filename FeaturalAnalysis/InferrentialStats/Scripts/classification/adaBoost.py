#!/usr/bin/env python3

from classificationValidation import ClassificationValidator
from sklearn.ensemble import AdaBoostClassifier


def validate(clf, X_test, y_true, labelClasses):

    return ClassificationValidator("Ada Boost", clf, X_test, y_true, labelClasses)


def model(X_train, y_train):

    return AdaBoostClassifier().fit(X_train, y_train)


def main(X_train, y_train, X_test, y_test, labelClasses):

    # Construct the model
    clf = model(X_train, y_train)

    # Validate the model
    scores = validate(clf, X_test, y_test, labelClasses)

    return scores

