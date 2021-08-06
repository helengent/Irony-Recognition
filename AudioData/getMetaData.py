#!/usr/bin/env python3

import os
from glob import glob

fileList = glob("./GatedPruned2/*.wav")

#This hasn't been touched since the filename re-work

with open("metaData.txt", "a+") as m:
    m.write("filename\tspeaker\tgender\tlabel\n")

for f in fileList:
    name = f.split("_")[1].split(".")[0]

    if name[0] == "b":
        speaker = 0
        gender = 0
    elif name[0] == "g":
        speaker = 1
        gender = 1
    elif name[0] == "p":
        speaker = 2
        gender = 1
    elif name[0] == "r":
        speaker = 3
        gender = 2
    elif name[0] == "y":
        speaker = 4
        gender = 0

    if name[-1] == "I":
        label = 0
    else:
        label = 1

    with open("metaData.txt", "a+") as m:
        m.write("{}\t{}\t{}\t{}\n".format(name, speaker, gender, label))

