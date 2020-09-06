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
from sklearn.model_selection import KFold
import torchvision.transforms as transforms
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import balanced_accuracy_score, precision_score, recall_score, f1_score

#Taken from https://www.deeplearningwizard.com/deep_learning/practical_pytorch/pytorch_feedforward_neuralnetwork/
'''
STEP 3: CREATE MODEL CLASS
'''
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

def pilotNN(train_dataset, train_labels, test_dataset, test_labels):

    batch_size = len(train_dataset)
    n_iters = 3000
    num_epochs = n_iters / (len(train_dataset) / batch_size)
    num_epochs = int(num_epochs)

    # train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=False)
    # test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

    # train_loader = [(torch.tensor(vector).type('torch.FloatTensor'), torch.tensor(label).type('torch.FloatTensor')) for vector, label in zip(train_dataset, train_labels)]
    # test_loader = [(torch.tensor(vector).type('torch.FloatTensor'), torch.tensor(label).type('torch.FloatTensor')) for vector, label in zip(test_dataset, test_labels)]

    train_loader = [(torch.tensor(train_dataset).type('torch.FloatTensor'), torch.tensor(train_labels).type('torch.LongTensor'))]
    test_loader = [(torch.tensor(test_dataset).type('torch.FloatTensor'), torch.tensor(test_labels).type('torch.LongTensor'))]

    '''
    STEP 4: INSTANTIATE MODEL CLASS
    '''
    input_dim = np.shape(train_dataset)[1]
    hidden_dim = 100
    output_dim = 10
    net = FeedforwardNeuralNetModel(input_dim=input_dim, hidden_dim=hidden_dim, output_dim=output_dim)

    '''
    STEP 5: INSTANTIATE LOSS CLASS
    '''
    criterion = nn.CrossEntropyLoss()

    '''
    STEP 6: INSTANTIATE OPTIMIZER CLASS
    '''
    learning_rate = 0.1
    optimizer = torch.optim.SGD(net.parameters(), lr=learning_rate)

    '''
    STEP 7: TRAIN THE MODEL
    '''
    iter = 0
    for epoch in range(num_epochs):
        for i, (features, labels) in enumerate(train_loader):

            # Clear gradients w.r.t. parameters
            optimizer.zero_grad()

            # Forward pass to get output/logits
            outputs = net(features)

            # Calculate Loss: softmax --> cross entropy loss
            loss = criterion(outputs, labels)

            # Getting gradients w.r.t. parameters
            loss.backward()

            # Updating parameters
            optimizer.step()

            iter += 1

            if iter % 5 == 0:
                # Calculate Accuracy         
                # correct = 0
                # total = 0
                # Iterate through test dataset
                # for features, labels in test_loader:

                #     # Forward pass only to get logits/output
                #     outputs = net(features)

                #     # Get predictions from the maximum value
                #     _, predicted = torch.max(outputs.data, 1)

                #     # Total number of labels
                #     total += labels.size(0)

                #     # Total correct predictions
                #     correct += (predicted == labels).sum()

                # accuracy = 100 * correct / total

                # # Print Loss
                # print('Iteration: {}. Loss: {}. Accuracy: {}'.format(iter, loss.item(), accuracy))
                
                outputs = net(test_loader[0][0])
                # Get predictions from the maximum value
                _, predicted = torch.max(outputs.data, 1)

                recall = recall_score(test_loader[0][1], predicted)
                precision = precision_score(test_loader[0][1], predicted)
                f1 = f1_score(test_loader[0][1], predicted)
                balanced_accuracy = balanced_accuracy_score(test_loader[0][1], predicted)
                print('Iteration: {}. Loss: {}. Accuracy: {}. Precision: {}. Recall: {}. F1: {}'.format(iter, loss.item(), balanced_accuracy, precision, recall, f1))

    print(1)

def main(df):
    labs = np.array(df.pop("label"))
    newdf = np.array(df)

    X_K, X_dev, y_K, y_dev = train_test_split(newdf, labs, test_size=0.1, random_state=6)

    kf = KFold(n_splits=5, shuffle=True)
    for train_index, test_index in kf.split(X_K):
        # print("TRAIN:", train_index, "TEST:", test_index)
        X_train, y_train, X_test, y_test = X_K[train_index], y_K[train_index], X_K[test_index], y_K[test_index]

        pilotNN(X_train, y_train, X_test, y_test)

    print(1)

if __name__ == "__main__":
    with open("baseline_consolidated.pkl", "rb") as p:
        bigDF = pd.read_pickle(p)

    main(bigDF)
