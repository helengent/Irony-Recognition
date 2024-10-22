#!/usr/bin/env python

import os
import shutil
from glob import glob


def saveOut(episodePath, speakers, matchList, speaker1Mismatches, speaker2Mismatches, speaker1odd, speaker2odd, parentDir):

    if not os.path.isdir(parentDir):
        os.mkdir(parentDir)

    os.mkdir("{}/matched".format(parentDir))
    for item in matchList:
        shutil.copy("{}/{}/{}".format(episodePath, speakers[0], item), "{}/matched/{}".format(parentDir, item))

    os.mkdir("{}/{}_mismatches".format(parentDir, speakers[0]))
    for item in speaker1Mismatches:
        shutil.copy("{}/{}/{}".format(episodePath, speakers[0], item), "{}/{}_mismatches/{}".format(parentDir, speakers[0], item))

    os.mkdir("{}/{}_mismatches".format(parentDir, speakers[1]))
    for item in speaker2Mismatches:
        shutil.copy("{}/{}/{}".format(episodePath, speakers[1], item), "{}/{}_mismatches/{}".format(parentDir, speakers[1], item))

    os.mkdir("{}/{}_odd".format(parentDir, speakers[0]))
    for item in speaker1odd:
        shutil.copy("{}/{}/{}".format(episodePath, speakers[0], item), "{}/{}_odd/{}".format(parentDir, speakers[0], item))

    os.mkdir("{}/{}_odd".format(parentDir, speakers[1]))
    for item in speaker2odd:
        shutil.copy("{}/{}/{}".format(episodePath, speakers[1], item), "{}/{}_odd/{}".format(parentDir, speakers[1], item))


#uttDict is a dictionary where the keys are annotator codes and the values 
# are a list of utterances for a given episode from that annotator
def reconcile(uttDict, speakers):

    #Files with the same speaker id, utterance number, and irony label
    matchList = list()

    #Files with the same speaker id and utterance number, but different irony labels
    speaker1Mismatches, speaker2Mismatches = list(), list()

    #Files that only exist for one speaker or the other
    speaker1odd, speaker2odd = list(), list()

    baseList1 = [item[:-5] for item in uttDict[speakers[0]]]
    baseList2 = [item[:-5] for item in uttDict[speakers[1]]]

    for item in uttDict[speakers[0]]:
        if item in uttDict[speakers[1]]:
            matchList.append(item)
        elif item[:-5] in baseList2:
            speaker1Mismatches.append(item)
        else:
            speaker1odd.append(item)

    for item in uttDict[speakers[1]]:
        if item in uttDict[speakers[0]]:
            pass
        elif item[:-5] in baseList1:
            speaker2Mismatches.append(item)
        else:
            speaker2odd.append(item)

    return matchList, speaker1Mismatches, speaker2Mismatches, speaker1odd, speaker2odd


def main(episodePath, recOutDir):

    dirList = glob("{}/AN*".format(episodePath))
    speakers = [item.split("/")[-1] for item in dirList]
    speakers.sort()

    uttDict = dict()

    for s in speakers:
        uttDict[s] = [item.split("/")[-1] for item in glob("{}/{}/*.wav".format(episodePath, s))]
    
    matchList, speaker1Mismatches, speaker2Mismatches, speaker1odd, speaker2odd = reconcile(uttDict, speakers)

    print("{} Matches".format(len(matchList)))
    print("{} Mismatches".format(len(speaker1Mismatches)))
    print("{} odd files for {}".format(len(speaker1odd), speakers[0]))
    print("{} odd files for {}".format(len(speaker2odd), speakers[1]))

    saveOut(episodePath, speakers, matchList, speaker1Mismatches, speaker2Mismatches, speaker1odd, speaker2odd, recOutDir)


if __name__=="__main__":

    nums = ["16"]
    
    for num in nums:
        episodePath = "/Users/helengent/Desktop/reconcile/SBep{}".format(num)
        
        ep = episodePath.split("/")[-1]
        recOutDir = "/Users/helengent/Desktop/reconcileD/{}".format(ep)

        main(episodePath, recOutDir)