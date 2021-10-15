#!/usr/bin/env python3

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.patches import FancyArrowPatch
from mpl_toolkits.mplot3d import proj3d

from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler


#https://stackoverflow.com/questions/22867620/putting-arrowheads-on-vectors-in-matplotlibs-3d-plot
class Arrow3D(FancyArrowPatch):
    def __init__(self, xs, ys, zs, *args, **kwargs):
        FancyArrowPatch.__init__(self, (0,0), (0,0), *args, **kwargs)
        self._verts3d = xs, ys, zs

    def draw(self, renderer):
        xs3d, ys3d, zs3d = self._verts3d
        xs, ys, zs = proj3d.proj_transform(xs3d, ys3d, zs3d, renderer.M)
        self.set_positions((xs[0],ys[0]),(xs[1],ys[1]))
        FancyArrowPatch.draw(self, renderer)


def plotCorrelation(df, finalDF, pca, features):

    fig = plt.figure(figsize=(15, 15))
    ax = fig.add_subplot(111, projection='3d')

    ax.set_xlabel('Principal Component 1', fontsize = 10)
    ax.set_ylabel('Principal Component 2', fontsize = 10)
    ax.set_zlabel('Principal Component 3', fontsize = 10)

    colList = df.columns[:6]
    i = 0

    textLocations = list()
    colors = ["r", "b", "g"]

    while i < len(colList):
        PCnum = colList[i][0]
        feats = df[colList[i]]
        values = df[colList[i+1]]

        c = colors[int(PCnum[-1]) - 1]

        for f, v in zip(feats, values):

            xCoords = [np.mean(finalDF.loc[:, 'PC1']), pca.components_[0][features.index(f)]]
            yCoords = [np.mean(finalDF.loc[:, 'PC2']), pca.components_[1][features.index(f)]]
            zCoords = [np.mean(finalDF.loc[:, 'PC3']), pca.components_[2][features.index(f)]]

            a = Arrow3D(xCoords, yCoords, zCoords, mutation_scale=20, lw=0.5, arrowstyle="-|>", color=c)
            ax.add_artist(a)

            textLocation = [xCoords[1] + 0.05, yCoords[1] + 0.05, zCoords[1] + 0.05]

            # #Slightly adjust text location to avoid overlapping text
            for h in range(3):
                if h == 0:
                    adjustment = 0.00
                elif h == 1:
                    adjustment = 0.00
                elif h == 2:
                    adjustment = 0.01
                j = [t for t in textLocations if t[h] > textLocation[h] - adjustment and t[h] < textLocation[h] + adjustment]
                while len(j) > 0:
                    textLocation[h] += adjustment
                    j = [t for t in textLocations if t[h] > textLocation[h] - adjustment and t[h] < textLocation[h] + adjustment]

            textLocations.append(textLocation)
            ax.text(textLocation[0], textLocation[1], textLocation[2], f, fontsize=7, color=c)

        i += 2

    ax.set_xlim([-.5, .5])
    ax.set_ylim([-.5, .5])
    ax.set_zlim([-.5, .5])

    plt.title("Top Features Correlated with First 3 PCs")
    plt.draw()
    
    plt.savefig("../Output/CorrelationCircle.png")


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
    ax.scatter3D(finalDF.loc[indicesToKeep, 'PC1']
            , finalDF.loc[indicesToKeep, 'PC2']
            , finalDF.loc[indicesToKeep, 'PC3']
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
    ax.scatter(finalDF.loc[indicesToKeep, 'PC1']
            , finalDF.loc[indicesToKeep, 'PC2']
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

    PCnames = ["PC{}".format(i) for i in range(1, numComps+1)]

    principalDF = pd.DataFrame(data = principalComponents, columns = PCnames)

    finalDF = pd.concat([df[['fileName', 'speaker','label']], principalDF], axis = 1)
    finalDF.to_csv("../Data/{}factorPCA.csv".format(numComps), index = False)

    variance = pca.explained_variance_ratio_
    var = np.cumsum(np.round(variance, decimals=3)*100)

    varDF = pd.DataFrame(index=["Proportion of Variance", "Cumulative Proportion"])

    for n, v, c in zip(PCnames, variance, var):
        varDF[n] = [v, c]

    varDF.to_csv("../Output/PCA_variance.csv")

    plotVariance(var)
    plotPCs(finalDF)
    plotPCs3D(finalDF)

    header = pd.MultiIndex.from_product([PCnames,
                                        ['Features', 'Correlation']])

    outDF = pd.DataFrame(columns=header)
    for p, n in zip(pca.components_, PCnames):
        inds = np.argpartition(np.abs(p), -5)[-5:]
        top_contribs = p[inds]
        top_feats = np.array(features)[inds]
        outDF[n, 'Features'] = top_feats
        outDF[n, 'Correlation'] = top_contribs

    outDF.to_csv("../Output/PCA_top_feats.csv")

    plotCorrelation(outDF, finalDF, pca, features)


if __name__=="__main__":

    # df = pd.read_csv("~/Data/AcousticData/text_feats/Pruned3_asr_text_feats.csv")
    df = pd.read_csv("../Data/all_narrowed.csv")
    df.drop(columns=["ZCR"])
    print(df)
    print(df.shape)
    numComps = [6]

    for n in numComps:
        main(df, n)
