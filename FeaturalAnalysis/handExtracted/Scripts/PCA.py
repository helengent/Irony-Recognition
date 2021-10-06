#!/usr/bin/env python3

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler


def main(df, numComps):

    # features = ["f0globalMean", "f0globalRange", "f0globalSD", "f0globalMedian",
                # "energyRange", "energySD", "hnrglobalMean", "hnrglobalRange", "hnrglobalSD", 
                # "duration", "sound2silenceRatio", "totalPauses", "Avg. Word Dur.", "Avg. Sil. Dur.", "SyllPerSecond"]
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

    #Plot the percentage of variance explained with the addition of PCs
    plt.ylabel("% Variance Explained")
    plt.xlabel("# of Features")
    plt.title("PCA Analysis")
    plt.ylim(0, 100)
    plt.style.context("seaborn-whitegrid")

    plt.plot(var)
    plt.savefig("../Output/PCA_analysis.png")


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


if __name__=="__main__":

    # df = pd.read_csv("~/Data/AcousticData/text_feats/Pruned3_asr_text_feats.csv")
    df = pd.read_csv("../Data/all_narrowed.csv")
    numComps = [30]

    for n in numComps:
        main(df, n)
