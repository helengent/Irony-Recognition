#!/usr/bin/env python3

import os
import numpy as np
from preProcessing.ASR import parseTextGrid
from glob import glob

arpabetConsonantalList = ['K', 'S', 'L', 'M', 'SH', 'N', 'P', 'T', 'Z', 'W', 'D', 'B', 
                            'V', 'R', 'NG', 'G', 'TH', 'F', 'DH', 'HH', 'CH', 'JH', 'Y', 'ZH']
arpabetVocalicList = ['EH2', 'AH0', 'EY1', 'OY2', 'OW1', 'AH1', 'EH1', 'IH1', 'AA1', 
                        'AY1', 'ER0', 'AE1', 'AE2', 'AO1', 'IH0', 'IY2', 'IY1', 'UH1', 
                        'IY0', 'OY1', 'OW2', 'UW1', 'IH2', 'EH0', 'AO2', 'AA0', 'AA2', 
                        'OW0', 'EY0', 'AE0', 'AW2', 'AW1', 'EY2', 'UW0', 'AH2', 'UW2', 
                        'AO0', 'AY2', 'ER1', 'UH2', 'AY0', 'ER2', 'OY0', 'UH0', 'AW0']


def speakerDurationData(input_dir, speakerList):

    fileList = glob("{}/*.TextGrid".format(input_dir))

    for speaker in speakerList:

        wordDict = dict()
        phoneDict = dict()

        fileSublist = [f for f in fileList if os.path.basename(f).split("_")[1][0].upper() == speaker]
        for f in fileSublist:
            _, words, phones = parseTextGrid.main(f)

            for word in words:
                if word[0] not in wordDict:
                    wordDict[word[0]] = [word[-1]]
                else:
                    wordDict[word[0]].append(word[-1])
            
            for phone in phones:
                if phone[0] not in phoneDict:
                    phoneDict[phone[0]] = [phone[-1]]
                else:
                    phoneDict[phone[0]].append(phone[-1])

        for w in wordDict.keys():
            wordDict[w] = np.mean(wordDict[w])
        for p in phoneDict.keys():
            phoneDict[p] = np.mean(phoneDict[p])

        with open("../../Data/AcousticData/SpeakerMetaData/{}_avgDur.txt".format(speaker), "w") as f:
            f.write("speaker\t{}\n".format(speaker))
            f.write("avgWordDur\t{}\n".format(np.mean([wordDict[key] for key in wordDict.keys() if key != 'sp' and key != "{LG}"])))
            f.write("avgVowelDur\t{}\n".format(np.mean([phoneDict[key] for key in phoneDict.keys() if key in arpabetVocalicList])))
            f.write("avgConsonantDur\t{}\n".format(np.mean([phoneDict[key] for key in phoneDict.keys() if key in arpabetConsonantalList])))
            if 'sp' in wordDict.keys():
                f.write("avgPauseDur\t{}\n".format(wordDict['sp']))
            if "{LG}" in wordDict.keys():
                f.write("avgLaughDur\t{}\n".format(wordDict["{LG}"]))
            for p in phoneDict.keys():
                if p in arpabetConsonantalList or p in arpabetVocalicList:
                    f.write("{}\t{}\n".format(p, phoneDict[p]))
                   