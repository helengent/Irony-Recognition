#!/usr/bin/env python3

from classificationValidation import ClassificationValidator
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis


def validate(clf, X_test, y_true, labelClasses):

    return ClassificationValidator("Linear Discriminant Analysis", clf, X_test, y_true, labelClasses)


def model(X_train, y_train):

    return LinearDiscriminantAnalysis().fit(X_train, y_train)


def main(X_train, y_train, X_test, y_test, labelClasses):

    # Construct the model
    clf = model(X_train, y_train)

    # Validate the model
    scores = validate(clf, X_test, y_test, labelClasses)

    return scores

