#!/usr/bin/env python3

import sys
import pandas as pd

class Speaker:
    def __init__(self, speaker, f0_txt, dur_txt=None, seg_txt=None):

        genders = pd.read_csv("~/Data/AcousticData/SpeakerMetaData/speakersGenders.txt")
        if speaker in genders["speaker"].tolist():
            self.gender = genders[genders['speaker']==speaker]['gender'].tolist()[0].strip()
        else:
            self.gender = ''

        self.speaker = speaker
        self.f = open(f0_txt, 'r')
        self.f0_txt = self.f.readlines()
        for i, line in enumerate(self.f0_txt):
            self.f0_txt[i] = line.split()
        self.mean = float(self.f0_txt[3][1])
        self.sd = float(self.f0_txt[4][1])
        self.upperLimit = float(self.f0_txt[1][1])
        self.lowerLimit = float(self.f0_txt[2][1])

        if dur_txt:
            self.f = open(dur_txt, 'r')
            self.dur_txt = self.f.readlines()
            for i, line in enumerate(self.dur_txt):
                self.dur_txt[i] = line.split()
            self.avgWordDur = float(self.dur_txt[1][1])
            self.avgVowelDur = float(self.dur_txt[2][1])
            self.avgConsonantDur = float(self.dur_txt[3][1])
            self.avgPauseDur = float(self.dur_txt[4][1])
            self.avgLaughDur = float(self.dur_txt[5][1])

            self.avgPhoneDurs = dict()
            for i in range(6, len(self.dur_txt)):
                if self.dur_txt[i][0] != 'sp' and self.dur_txt[i][0] != "{LG}":
                    self.avgPhoneDurs[self.dur_txt[i][0]] = self.dur_txt[i][1]

        if seg_txt:
            self.g = open(seg_txt, 'r')
            self.seg_txt = self.g.readlines()
            for i, line in enumerate(self.seg_txt):
                self.seg_txt[i] = line.split()
            self.seg_info = dict()
            for i in range(1, len(self.seg_txt)):
                if "-" in self.seg_txt[i][0]:
                    phone = self.seg_txt[i][0].split("-")[0]
                    measure = self.seg_txt[i][0].split("-")[1]
                else:
                    phone = self.seg_txt[i][0].split("_")[0]
                    measure = self.seg_txt[i][0].split("_")[1]   
                value = self.seg_txt[i][1]
                if phone not in self.seg_info.keys():
                    self.seg_info[phone] = dict()
                self.seg_info[phone][measure] = value


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