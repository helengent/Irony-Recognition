#!/usr/bin/env python3

from num2words import num2words
from glob import glob

def main(txtFile):

    with open(txtFile, "r") as f:
        text = f.read()

    oldtext = text.split()

    newText = list()
    for word in oldtext:
        if len(word) != 4:
            try:
                word = int(word)
                word = num2words(word)
                newText.append(word)
            except ValueError:
                newText.append(word)
        else:
            newText.append(word)

    newText = " ".join(newText)
    
    with open(txtFile, "w") as f:
        f.write(newText)

if __name__=="__main__":

    txtList = glob("../../../../Data/TextData/Pruned3_asr/*.txt")

    for t in txtList:
        main(t)
