#!/usr/bin/env python3

import os
import sys
from sd import sd
import numpy as np
import parselmouth
import pandas as pd
from glob import glob
from speaker import Speaker
from numpy import genfromtxt
import matplotlib.pyplot as plt
#from f0_parseltongue import smooth
from scipy.io import wavfile as wave
from preProcessing.silence import highPass, validateTs
# from reaper_f0extractor import f0VecTime
from lib.WAVReader import WAVReader as WR
from preProcessing.limitsUpperLower import giveMean, giveSD
from python_speech_features import mfcc, logfbank
from lib.DSP_Tools import findEndpoint, normaliseRMS

class Extractor:
    def __init__(self, wav, speaker, irony, winSize=5):
        self.name = wav
        # self.text = text
        self.speaker = speaker
        self.irony = irony
        self.winSize = winSize
        self.sound = parselmouth.Sound(wav)
        self.wav = WR(self.name)
        self.ampData = self.wav.getData()
        self.fs = self.wav.getSamplingRate()
        self.dur = self.wav.getDuration()
        self.f0Data = self.getF0Contour()
   

    #Speaker features
    def getSpeaker(self):
        return self.speaker.getSpeaker()


    def getGender(self):
        return self.speaker.getGender()


    #F0 features
    def getF0Contour(self):
        upperLimit = self.speaker.getUpperLimit()
        lowerLimit = self.speaker.getLowerLimit()
        
        f0_array = list()

        #This is the code to get the Parselmouth f0 contour
        pitch = self.sound.to_pitch_cc(pitch_floor=lowerLimit, pitch_ceiling=upperLimit)

        start = 0
        end = self.winSize / 1000

        while end <= self.wav.getDuration():

            t = pitch.get_value_at_time(start)
                
            f0_array.append(t)
            start += self.winSize/1000
            end += self.winSize/1000

        return f0_array
    
        #This is the code to get the reaper f0 contour
        # f0Contour = f0VecTime(self.text, lowerLimit, upperLimit, ms=self.winSize)


    def getMeanf0(self):
        return giveMean(self.getF0Contour()) 


    def getSDF0(self):
        return giveSD(self.getF0Contour())


    def getMedianF0(self):
        f0 = self.getF0Contour()
        mid = int(len(f0))
        return f0[mid]

    #Timing features
    #window size is hard-coded to 5ms for this one
    def findSilences(self):
        lowerCutoff = 40
        order = 1000
        w = 0.005
        bs = highPass(self.fs, lowerCutoff, order)
        ampData = np.array(self.ampData)
        ampData = np.squeeze(ampData)
        ampData = np.convolve(ampData[:], bs)
        ampData = np.reshape(ampData, (len(ampData), 1))
        silences = findEndpoint(ampData, self.fs, win_size=w)
        silences = validateTs(silences[0])
        return silences, w
    

    def getTimingStats(self):
        silence, win_size = self.findSilences()

        #average pause length (in ms)
        pauseLengths = list()
        currentCount = 0
        for i, item in enumerate(silence):
            if i+1 != len(silence):
                nextItem = silence[i+1]
                if item == True:
                    currentCount += 1
                    if item != nextItem:
                        pauseLengths.append(currentCount)
                        currentCount = 0
            else:
                if item == True:
                    currentCount += 1
                    pauseLengths.append(currentCount)

        avgPauseLength = (sum(pauseLengths)/len(pauseLengths))/win_size

        #sound-to-silence ratio
        sound = np.count_nonzero(silence==False)
        shh = np.count_nonzero(silence==True)
        s2sRatio = sound/shh
        
        #total number of pauses
        totalPauses = len(pauseLengths)

        return avgPauseLength, s2sRatio, totalPauses

    #HNR
    def getHNR(self):
        harmonicity = self.sound.to_harmonicity_cc()
        hnr = [value[0] if value != -200 else 0 for value in harmonicity.values.transpose()]
        return hnr


    #Jitter and Shimmer
    #TODO
    def getJitter(self):
        jitter = 0
        return jitter

    def getShimmer(self):
        shimmer = 0
        return shimmer

    #First 13 MFCCs
    def getMFCCs(self):
        ms = self.winSize/1000
        (rate, sig) = wave.read(self.name)
        mfccs = mfcc(sig, samplerate=rate, winlen=ms, winstep=ms)
        return mfccs

    #Perceptual Linear Prediction
    def getPLP(self):
        n = os.path.basename(self.name).split(".")[0]
        rastaFile = "../AcousticData/rastaplp/{}.csv".format(n)
        rasta = genfromtxt(rastaFile, delimiter=',')
        return rasta

    #Amplitude modulation spectrum
    def getAMS(self):
        n = os.path.basename(self.name).split(".")[0]
        amsFile = "../AcousticData/ams/{}.csv".format(n)
        ams = genfromtxt(amsFile, delimiter=',')
        return ams

    #Relative spectral transform
    #TODO
