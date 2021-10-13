#!/usr/bin/env python3

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d

from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler


def plotPCs3D(finalDF):

    fig = plt.figure()
    ax = plt.axes(projection='3d')

    ax.set_xlabel('Principal Component 1', fontsize = 10)
    ax.set_ylabel('Principal Component 2', fontsize = 10)
    ax.set_zlabel('Principal Component 3', fontsize = 10)
    ax.set_title('3 component PCA', fontsize = 15)

    targets = ['I', 'N']
    colors = ['g', 'b']
    for target, color in zip(targets,colors):
        indicesToKeep = finalDF['label'] == target
        ax.scatter3D(finalDF.loc[indicesToKeep, 'PC1']
                , finalDF.loc[indicesToKeep, 'PC2']
                , finalDF.loc[indicesToKeep, 'PC3']
                , alpha = 0.5
                , c = color
                , s = 50)
    ax.legend(targets)
    #ax.grid()
    plt.savefig("../Output/3_factorPCA.png")

    plt.clf()
    fig = plt.figure()
    ax = plt.axes(projection='3d')

    ax.set_xlabel('Principal Component 1', fontsize = 10)
    ax.set_ylabel('Principal Component 2', fontsize = 10)
    ax.set_zlabel('Principal Component 3', fontsize = 10)
    ax.set_title('3 component PCA - Ironic', fontsize = 15)

    indicesToKeep = finalDF['label'] == 'I'
    ax.scatter3D(finalDF.loc[indicesToKeep, 'PC1']
            , finalDF.loc[indicesToKeep, 'PC2']
            , finalDF.loc[indicesToKeep, 'PC3']
            , alpha = 0.5
            , c = 'g'
            , s = 50)
    ax.legend(targets)
    plt.savefig("../Output/3_factorPCA_Ironic.png")

    plt.clf()
    fig = plt.figure()
    ax = plt.axes(projection='3d')

    ax.set_xlabel('Principal Component 1', fontsize = 10)
    ax.set_ylabel('Principal Component 2', fontsize = 10)
    ax.set_zlabel('Principal Component 3', fontsize = 10)
    ax.set_title('3 component PCA - Non-Ironic', fontsize = 15)

    indicesToKeep = finalDF['label'] == 'N'
    ax.scatter3D(finalDF.loc[indicesToKeep, 'PC0']
            , finalDF.loc[indicesToKeep, 'PC1']
            , finalDF.loc[indicesToKeep, 'PC2']
            , alpha = 0.5
            , c = 'b'
            , s = 50)
    ax.legend(targets)
    plt.savefig("../Output/3_factorPCA_Non-Ironic.png")


def plotPCs(finalDF):

    #Plot the first two PCs in different colors for ironic and non-ironic samples
    fig = plt.figure(figsize = (8,8))
    ax = fig.add_subplot(1,1,1) 
    ax.set_xlabel('Principal Component 1', fontsize = 15)
    ax.set_ylabel('Principal Component 2', fontsize = 15)
    ax.set_title('2 component PCA', fontsize = 20)
    
    targets = ['I', 'N']
    colors = ['g', 'b']
    for target, color in zip(targets,colors):
        indicesToKeep = finalDF['label'] == target
        ax.scatter(finalDF.loc[indicesToKeep, 'PC1']
                , finalDF.loc[indicesToKeep, 'PC2']
                , c = color
                , s = 50)
    ax.legend(targets)
    ax.grid()
    plt.savefig("../Output/2_factorPCA.png")

    plt.clf()
    fig = plt.figure(figsize = (8,8))
    ax = fig.add_subplot(1,1,1) 
    ax.set_xlabel('Principal Component 1', fontsize = 15)
    ax.set_ylabel('Principal Component 2', fontsize = 15)
    ax.set_title('2 component PCA - Ironic', fontsize = 20)
    
    indicesToKeep = finalDF['label'] == 'I'
    ax.scatter(finalDF.loc[indicesToKeep, 'PC0']
            , finalDF.loc[indicesToKeep, 'PC1']
            , c = 'g'
            , s = 50)
    ax.legend(targets)
    ax.grid()
    plt.savefig("../Output/2_factorPCA_Ironic.png")

    plt.clf()
    fig = plt.figure(figsize = (8,8))
    ax = fig.add_subplot(1,1,1) 
    ax.set_xlabel('Principal Component 1', fontsize = 15)
    ax.set_ylabel('Principal Component 2', fontsize = 15)
    ax.set_title('2 component PCA - Non-Ironic', fontsize = 20)
    
    indicesToKeep = finalDF['label'] == 'N'
    ax.scatter(finalDF.loc[indicesToKeep, 'PC1']
            , finalDF.loc[indicesToKeep, 'PC2']
            , c = 'b'
            , s = 50)
    ax.legend(targets)
    ax.grid()
    plt.savefig("../Output/2_factorPCA_Non-Ironic.png")


def plotVariance(v):

    #Plot the percentage of variance explained with the addition of PCs
    plt.ylabel("% Variance Explained")
    plt.xlabel("# of Features")
    plt.title("PCA Analysis")
    plt.ylim(0, 100)
    plt.style.context("seaborn-whitegrid")

    plt.plot(v)
    plt.savefig("../Output/PCA_analysis.png")


def main(df, numComps):

    features = df.columns[3:].tolist()

    x = df.loc[:, features].values
    y = df.loc[:, ["label"]].values

    x = StandardScaler().fit_transform(x)

    imp_mean = SimpleImputer()
    x = imp_mean.fit_transform(x)

    pca = PCA(n_components=numComps)
    principalComponents = pca.fit_transform(x)

    colList = ["PC{}".format(i) for i in range(numComps)]

    principalDF = pd.DataFrame(data = principalComponents, columns = colList)

    finalDF = pd.concat([df[['fileName', 'speaker','label']], principalDF], axis = 1)
    finalDF.to_csv("../Data/{}factorPCA.csv".format(numComps), index = False)

    variance = pca.explained_variance_ratio_
    var = np.cumsum(np.round(variance, decimals=3)*100)
    print(var)
    plotVariance(var)
    plotPCs(finalDF)
    plotPCs3D(finalDF)




if __name__=="__main__":

    # df = pd.read_csv("~/Data/AcousticData/text_feats/Pruned3_asr_text_feats.csv")
    df = pd.read_csv("../Data/all_narrowed.csv")
    df.drop(columns=["ZCR"])
    print(df)
    print(df.shape)
    numComps = [6]

    for n in numComps:
        main(df, n)
