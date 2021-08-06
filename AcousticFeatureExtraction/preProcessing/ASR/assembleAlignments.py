#!/usr/bin/env python3

import os
from glob import glob
from parseTextGrid import parse


def writeOut(fileName, words, phones, fileMod):

    with open("../../../../Data/TextData/{}/align.txt".format(fileMod), "a+") as f:
        for word in words:
            f.write("{}\t{:.2f}\t{:.2f}\t{:.2f}\t{}\n".format(fileName, word[1], word[2], word[3], word[0]))
    with open("../../../../Data/TextData/{}/phone_align.txt".format(fileMod), "a+") as f:
        for phone in phones:
            f.write("{}\t{:.2f}\t{:.2f}\t{:.2f}\t{}\n".format(fileName, phone[1], phone[2], phone[3], phone[0]))


def main(fileMod):

    fileList = glob("../../../../Data/TextData/{}/*.TextGrid".format(fileMod))
    
    for f in fileList:
        start, end, words, phones = parse(f)
        fileName = os.path.basename(f).split(".")[0]
        print(fileName)

        writeOut(fileName, words, phones, fileMod)
    

if __name__=="__main__":

    fileMod = "Pruned2_asr"

    main(fileMod)