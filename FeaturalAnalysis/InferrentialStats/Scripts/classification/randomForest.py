#!/usr/bin/env python3

from classificationValidation import ClassificationValidator
from sklearn.ensemble import RandomForestClassifier


def validate(clf, X_test, y_true, labelClasses):

    return ClassificationValidator("Random Forest", clf, X_test, y_true, labelClasses)


def model(X_train, y_train):

    return RandomForestClassifier(max_depth = 5, n_estimators = 10, max_features = 1).fit(X_train, y_train)


def main(X_train, y_train, X_test, y_test, labelClasses):

    # Construct the model
    clf = model(X_train, y_train)

    # Validate the model
    scores = validate(clf, X_test, y_test, labelClasses)

    return scores

