#!/usr/bin/env python3
import sys

class Speaker:
    def __init__(self, speaker, txt, gender=""):
        self.speaker = speaker
        self.f = open(txt, 'r')
        self.txt = self.f.readlines()
        for i, line in enumerate(self.txt):
            self.txt[i] = line.split()
        self.gender = gender
        self.mean = float(self.txt[3][1])
        self.sd = float(self.txt[4][1])
        self.upperLimit = float(self.txt[1][1])
        self.lowerLimit = float(self.txt[2][1])

    def getSpeaker(self):
        return self.speaker

    def getGender(self):
        if self.gender != "":
            return(self.gender)
        else:
            print("gender not set")
            return(self.gender)

    def setGender(self, gender):
        self.gender = gender

    def getUpperLimit(self):
        return self.upperLimit

    def getLowerLimit(self):
        return self.lowerLimit

    def getSpeakerMeanF0(self):
        return self.mean

    def getSpeakerSDF0(self):
        return self.sd