#!/usr/bin/env python3

from joblib import Parallel, delayed
from numpy.random import seed
import multiprocessing as mp
from scipy import stats
import pandas as pd
import numpy as np
import glob
import os

seed(42)

#Thank you Chase Adams for this code <3

def multiProcStandardtTest(i):

    ir, ni, feature = i
    statistic, pvalue = stats.ttest_ind(ir, ni, equal_var = True, nan_policy = 'raise')
    return feature, statistic, pvalue


# In the event that leveneHOV returns that a given feature has homogeneity of variance, perform a t-test with that assumption
def standardtTestInd(df, levene):

    desiredCols = ['fileName', 'speaker', 'label']
    for i, row in levene.iterrows():
        if row['Levene p-Value'] > 0.05:
            desiredCols.append(row['Feature'])

    df = df[desiredCols]

    iDF = df[df['label'] == 'I']
    nDF = df[df['label'] == 'N']

    X = Parallel(n_jobs=mp.cpu_count())(delayed(multiProcStandardtTest)(([item for item in iDF[feature].tolist() if np.isnan(item) == False], [item for item in nDF[feature].tolist() if np.isnan(item) == False], feature)) for feature in df.columns[3:])
    featureList, stdTstatList, stdPValueList = map(list, zip(*X))
    df = pd.DataFrame(list(zip(featureList, stdTstatList, stdPValueList)), columns = ['Feature', 'T Test Statistic', 'p-Value'])
    return df


# First we have to determine which of the 'center' variations for the Levene's test we need to use
def determineCenter(ir, ni):

    ###############################
    # Check if normally distributed
    ###############################
    a, b = False, False
    irStat, irP = stats.shapiro(ir)
    if irP > 0.05: # Sample is Gaussian (normal); We should be running parametric statistical methods with these data
        a = True
    niStat, niP = stats.shapiro(ni)
    if niP > 0.5:
        b = True

    # Both need to be normally distributed to use 'mean' (Recommended for symmetric, moderate-tailed distributions)     
    if a and b:
        center = 'mean'
        return center, irStat, irP, niStat, niP
    # Now that we know it's not normally distributed, do we have a skewed distribution or a heavily-tailed distribution?
    else:
        ###############################
        # Check for skewness
        ###############################
        c, d = False, False
        irStatSkew, irPSkew = stats.skewtest(ir)
        if irPSkew > 0.05: # Sample rejects the null hypothesis that the skewness of the observations that the sample was drawn from is the same as that of a corresponding normal (Gaussian) distribution
            c = True
        niStatSkew, niPSkew = stats.skewtest(ni)
        if niPSkew > 0.5:
            d = True

        ###############################
        # Check for kurtosis
        ###############################
        e, f = False, False
        irStatKurt, irPKurt = stats.kurtosistest(ir)
        if irPKurt > 0.05: # Sample rejects the null hypothesis that the kurtosis of the observations are normal
            e = True
        niStatKurt, niPKurt = stats.kurtosistest(ni)
        if niPKurt > 0.5:
            f = True

        # NOTE
        # Skewness is a quantification of how mucha  distribution is pushed left or right; a measure of asymmetry in the distribution
            # If the data are skewed, we need to use the median of the data for the theoretical center
        # Kurtosis quantifies how much of the distribution is in the tail. 
            # If the data are kurtotic, we need to use a "trimmed" center

        # I think the way to do this is to pick whichever array is strongest in terms of p-value for rejecting the null hypothesis and 
        # use that to determine 'median' vs 'trimmed'. The logic of this is tenuous, but it seems a good way to run a bunch of these
        strongest = np.argmax([irPSkew, niPSkew, irPKurt, niPKurt])
        if strongest == 0 or strongest == 1:
            center = 'median'
            return center, irStatSkew, irPSkew, niStatSkew, niPSkew
        elif strongest == 2 or strongest == 3:
            center = 'trimmed'
            return center, irStatKurt, irPKurt, niStatKurt, niPKurt


def multiProcLeveneHOV(i):

    ir, ni = i
    # Figure out the center strategy for the test
    center, irStat, irP, niStat, niP = determineCenter(ir, ni) # I don't know if I really need any of this information saved out
                                                               # All it really indicates is how the decision was arrived at for the Levene strategy. 

    # Run the test
    statistic, pvalue = stats.levene(ir, ni, center = center)
    return center, statistic, pvalue


def leveneHOV(df):

    iDF = df[df['label'] == 'I']
    nDF = df[df['label'] == 'N']

    X = Parallel(n_jobs=mp.cpu_count())(delayed(multiProcLeveneHOV)(([item for item in iDF[feature].tolist() if np.isnan(item) == False], [item for item in nDF[feature].tolist() if np.isnan(item) == False])) for feature in df.columns[3:])
    leveneCenterList, leveneStatisticList, levenePValueList = map(list, zip(*X))
    df = pd.DataFrame(list(zip(leveneCenterList, leveneStatisticList, levenePValueList)), columns = ['Levene Center Strategy', 'Levene Test Statistic', 'Levene p-Value'])
    return df


def multiProctTest(i):

    ir, ni, feature = i
    statistic, pvalue = stats.ttest_ind(ir, ni, equal_var = False, nan_policy = 'raise') # Equal_var means Welch, not student's t-test
    return feature, statistic, pvalue


# Since we cannot guarantee that the size of the arrays will be equal (we have balanced participants here, but not 
# balanced number of utterances), we have to perform the Welch t-test to see if there is a significant difference
def tTestInd(df):

    iDF = df[df['label'] == 'I']
    nDF = df[df['label'] == 'N']

    X = Parallel(n_jobs=mp.cpu_count())(delayed(multiProctTest)(([item for item in iDF[feature].tolist() if np.isnan(item) == False], [item for item in nDF[feature].tolist() if np.isnan(item) == False], feature)) for feature in df.columns[3:])
    featureList, welchStatisticList, welchPValueList = map(list, zip(*X))
    df = pd.DataFrame(list(zip(featureList, welchStatisticList, welchPValueList)), columns = ['Feature', 'Welch Test Statistic', 'Welch p-Value'])
    return df


def main(df, name):

    # We can run a Welch's t-test on arrays of varying sizes because it doesn't assume homogeneity of variance
    output = tTestInd(df)

    # In addition to Welch's t-test, we can determine if we are dealing with samples that equate to roughly equal variances
    levene = leveneHOV(df)
    output = pd.concat([output, levene], axis = 1)

    standardT = standardtTestInd(df, output)

    output = pd.merge(output, standardT, how='outer')

    # We can get at an Analysis of Variance (ANOVA) for *some* of our features
    # here (if there are more than two groups we're interested in), but only if we can 
    # reject the null hypothesis surrounding Levene's test related to 
    # determining if we have homogeneity of variance between the groups of observations
    # anova = anovaCall(df, output['Levene p-Value'].to_list())

    # Save something out for each of the three
    output.to_csv('../Output/{}.csv'.format(name), index = False)


if __name__=="__main__":

    # names = ["prevAttested", "stressGrouped", "all"]
    names = ["all_narrowed"]
    # names = ["30factorPCA"]
    for name in names:
        df = "../Data/{}.csv".format(name)
        df = pd.read_csv(df)
        print(df.shape)

        iDF = df[df['label'] == 'I']
        nDF = df[df['label'] == 'N']

        dropList = list()

        for i, col in iDF.items():
            try:
                if len([item for item in col.tolist() if np.isnan(item) == False]) < 8:
                    dropList.append(i)
            except:
                pass

        for i, col in nDF.items():
            try:
                if len([item for item in col.tolist() if np.isnan(item) == False]) < 8:
                    dropList.append(i)
            except:
                pass

        dropList = list(set(dropList))
        print(len(dropList))
        df = df.drop(columns=dropList)
        print(df.shape)

        main(df, name)