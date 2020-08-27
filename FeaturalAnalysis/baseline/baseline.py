#!/usr/bin/env python3

import sys
import torch
import pickle
import numpy as np
import pandas as pd
import torch.nn as nn
from glob import glob
from ast import literal_eval
import torch.nn.functional as F
import torchvision.datasets as dsets
from matplotlib import pyplot as plt
from sklearn.decomposition import PCA
from sklearn.model_selection import KFold
import torchvision.transforms as transforms
from sklearn.preprocessing import StandardScaler

#Taken from https://www.deeplearningwizard.com/deep_learning/practical_pytorch/pytorch_feedforward_neuralnetwork/
class FeedforwardNeuralNetModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(FeedforwardNeuralNetModel, self).__init__()
        # Linear function
        self.fc1 = nn.Linear(input_dim, hidden_dim) 
        # Non-linearity
        self.relu = nn.ReLU()
        # Linear function (readout)
        self.fc2 = nn.Linear(hidden_dim, output_dim)  

    def forward(self, x):
        # Linear function
        out = self.fc1(x)
        # Non-linearity
        out = self.relu(out)
        # Linear function (readout)
        out = self.fc2(out)
        return out

def makeTensors(data):
    for row in data:
        t = torch.tensor(row)
        yield t

def pilotNN(x, y):

    #tList = list(makeTensors(x))



    batch_size = 100
    n_iters = 3000
    num_epochs = n_iters / (len(train_dataset) / batch_size)
    num_epochs = int(num_epochs)

    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)

    test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

    net = Net()
    params = list(net.parameters())
    output = list(net(tens) for tens in tList)
    print(output)

def doPCA(x, y):
    # Standardizing the features
    x = StandardScaler().fit_transform(x)
    print("Features standardized. Starting PCA")

    pca = PCA(n_components=2)

    principalComponents = pca.fit_transform(x)
    principalDf = pd.DataFrame(data = principalComponents, columns = ['principal component 1', 'principal component 2'])
    finalDF = pd.concat([principalDf, df[['label']]], axis = 1)

    print("PCA complete. Plotting")
    plotPCA(finalDF)

def main(df):
    newdf = df[:]
    print(1)

    
    


    

if __name__ == "__main__":

    with open("baseline_consolidated.pkl", "rb") as p:
        bigDF = pd.read_pickle(p)

    main(bigDF)
