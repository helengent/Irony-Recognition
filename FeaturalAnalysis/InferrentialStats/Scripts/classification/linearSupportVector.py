#!/usr/bin/env python3

from classificationValidation import ClassificationValidator
from sklearn.svm import SVC


def validate(clf, X_test, y_true, labelClasses):

    return ClassificationValidator("Linear Support Vector", clf, X_test, y_true, labelClasses)


def model(X_train, y_train):

    return SVC(kernel = 'linear', C=0.025).fit(X_train, y_train)


def main(X_train, y_train, X_test, y_test, labelClasses):

    # Construct the model
    clf = model(X_train, y_train)

    # Validate the model
    scores = validate(clf, X_test, y_test, labelClasses)

    return scores

