#!/usr/bin/env python3

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.patches import FancyArrowPatch
from mpl_toolkits.mplot3d import proj3d
from pandas.core.base import NoNewAttributesMixin

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


def plotCorrelation(df, finalDF, pca, features, subDir):

    fig = plt.figure(figsize=(20, 20))
    ax = fig.add_subplot(111, projection='3d')

    ax.set_xlabel('Principal Component 1', fontsize = 20)
    ax.set_ylabel('Principal Component 2', fontsize = 20)
    ax.set_zlabel('Principal Component 3', fontsize = 20)

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
                    adjustment = 0.01
                elif h == 1:
                    adjustment = 0.01
                elif h == 2:
                    adjustment = 0.02
                j = [t for t in textLocations if t[h] > textLocation[h] - adjustment and t[h] < textLocation[h] + adjustment]
                while len(j) > 0:
                    textLocation[h] += adjustment
                    j = [t for t in textLocations if t[h] > textLocation[h] - adjustment and t[h] < textLocation[h] + adjustment]

            textLocations.append(textLocation)
            ax.text(textLocation[0], textLocation[1], textLocation[2], f, fontsize=7, color=c)

        i += 2

    PC1mean = [np.mean(finalDF.loc[:, 'PC1'])] * 100
    PC2mean = [np.mean(finalDF.loc[:, 'PC2'])] * 100
    PC3mean = [np.mean(finalDF.loc[:, 'PC3'])] * 100

    l = np.linspace(-.5, .5, num=100)

    ax.plot(PC1mean, PC2mean, l, label="PC3", color="black")
    ax.plot(PC1mean, l, PC3mean, label="PC2", color="black")
    ax.plot(l, PC2mean, PC3mean, label="PC2", color="black")

    ax.set_xlim([-.5, .5])
    ax.set_ylim([-.5, .5])
    ax.set_zlim([-.5, .5])

    plt.title("Top Features Correlated with First 3 PCs", fontsize = 30)
    plt.draw()
    
    plt.savefig("../Output/{}/CorrelationCircle.png".format(subDir))


def plotPCs3D(finalDF, subDir):

    fig = plt.figure(figsize=(20, 20))
    ax = plt.axes(projection='3d')

    ax.set_xlabel('Principal Component 1', fontsize = 30)
    ax.set_ylabel('Principal Component 2', fontsize = 30)
    ax.set_zlabel('Principal Component 3', fontsize = 30)
    # ax.set_title('3 component PCA', fontsize = 30)

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
    # ax.legend(targets)

    plt.savefig("../Output/{}/3_factorPCA.png".format(subDir))

    for i in range(151, 156):
        ax.view_init(0, i)
        plt.savefig("../Output/{}/3Factor/0_{}.png".format(subDir, i))

    angles = [0, 60, 120, 180, 240, 300, 360]

    for angle in angles:
        for rotation in angles:
            ax.view_init(angle, rotation)
            plt.savefig("../Output/{}/3Factor/{}_{}.png".format(subDir, angle, rotation))


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
    plt.savefig("../Output/{}/3_factorPCA_Ironic.png".format(subDir))

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

    plt.savefig("../Output/{}/3_factorPCA_Non-Ironic.png".format(subDir))


def plotPCs(finalDF, subDir):

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
    plt.savefig("../Output/{}/2_factorPCA.png".format(subDir))

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
    plt.savefig("../Output/{}/2_factorPCA_Ironic.png".format(subDir))

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
    plt.savefig("../Output/{}/2_factorPCA_Non-Ironic.png".format(subDir))


def plotVariance(v, subDir):

    #Plot the percentage of variance explained with the addition of PCs
    plt.ylabel("% Variance Explained")
    plt.xlabel("# of Features")
    plt.title("PCA Analysis")
    plt.ylim(0, 100)
    plt.style.context("seaborn-whitegrid")

    plt.plot(v)
    plt.savefig("../Output/{}/PCA_analysis.png".format(subDir))


def main(df, newdf, numComps, subDir):

    #Data preparation
    features = df.columns[3:].tolist()
    n = df[df["label"] == "N"]
    n = n.loc[:, features].values

    x = df.loc[:, features].values
    new_x = newdf.loc[:, features].values
    y = df.loc[:, ["label"]].values

    #Scale data and impute missing values
    scaler = StandardScaler()
    scaler.fit(n)
    x = scaler.transform(x)
    new_x = scaler.transform(new_x)

    imp_mean = SimpleImputer()
    imp_mean.fit(x)
    x = imp_mean.transform(x)
    new_x = imp_mean.transform(new_x)

    #PCA
    pca = PCA(n_components=numComps)
    pca.fit(x)
    principalComponents = pca.transform(new_x)

    PCnames = ["PC{}".format(i) for i in range(1, numComps+1)]

    #Data frames with PC values per sample
    principalDF = pd.DataFrame(data = principalComponents, columns = PCnames)
    finalDF = pd.concat([newdf[['fileName', 'speaker','label']], principalDF], axis = 1)
    finalDF.to_csv("../Output/{}/{}factorPCA.csv".format(subDir, numComps), index = False)

    #Compute variance explained by each PC
    variance = pca.explained_variance_ratio_
    var = np.cumsum(np.round(variance, decimals=3)*100)

    print(variance)
    print(var)

    varDF = pd.DataFrame(index=["Proportion of Variance", "Cumulative Proportion"])
    for n, v, c in zip(PCnames, variance, var):
        varDF[n] = [v, c]
    varDF.to_csv("../Output/{}/PCA_variance.csv".format(subDir))

    #Plotting
    plotVariance(var, subDir)
    # plotPCs(finalDF, subDir)
    # plotPCs3D(finalDF, subDir)


    outFeats = list()
    #Dataframe of  for top features for each PC
    header = pd.MultiIndex.from_product([PCnames,
                                        ['Features', 'Correlation']])
    outDF = pd.DataFrame(columns=header)
    for p, n in zip(pca.components_, PCnames):
        inds = np.argpartition(np.abs(p), -5)[-5:]
        top_contribs = p[inds]
        top_feats = np.array(features)[inds]
        outDF[n, 'Features'] = top_feats
        outDF[n, 'Correlation'] = top_contribs

        for feat, ind in zip(top_feats, inds):
            if (feat, ind) not in outFeats:
                outFeats.append((feat, ind))

    # outDF.to_csv("../Output/{}/PCA_top_feats.csv".format(subDir))

    # print(outDF)

    #Plotting
    # plotCorrelation(outDF, finalDF, pca, features, subDir)

    for f in outFeats:
        feat, ind = f
        finalDF[feat] = new_x[:, ind]

    finalDF.to_csv("../Output/{}/{}Factor_feats.csv".format(subDir, numComps), index=False)


if __name__=="__main__":

    df = pd.read_csv("../Data/all_narrowed.csv")
    new_df = pd.read_csv("../Data/newTest_all_narrowed.csv")
    # df = pd.read_csv("../Data/ComParE.csv", index_col = 0)
    # df = df.drop(columns=["index"])

    numComps = [3]
    subDir = "newTest"

    for n in numComps:
        main(df, new_df, n, subDir)
