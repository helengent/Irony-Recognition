#!/usr/bin/env python3

import keras
import numpy as np
import pandas as pd
from keras import layers
from keras import models
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split

def pilotNN(X_train, X_test, y_train, y_test):
    
    input_dim = np.shape(X_train)[1]

    model = models.Sequential()
    model.add(layers.Dense(12, input_dim=input_dim, activation='relu'))
    model.add(layers.Dense(8, activation='relu'))
    model.add(layers.Dense(8, activation='relu'))
    model.add(layers.Dense(2, activation='softmax'))

    model.summary()
    print(1)

    # model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

def main(df):
    labs = np.array(df.pop("label"))
    newdf = np.array(df)

    X_train, X_test, y_train, y_test = train_test_split(newdf, labs, test_size=0.1, random_state=6)


    print(1)

if __name__ == "__main__":
    with open("baseline_consolidated.pkl", "rb") as p:
        bigDF = pd.read_pickle(p)

    main(bigDF)