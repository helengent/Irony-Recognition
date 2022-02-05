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
from parselmouth.praat import call
from scipy.io import wavfile as wave
from preProcessing.silence import highPass, validateTs
from lib.WAVReader import WAVReader as WR
from preProcessing.limitsUpperLower import giveMean, giveSD
from python_speech_features import mfcc, logfbank
from lib.DSP_Tools import findEndpoint, normaliseRMS

class Extractor:
    def __init__(self, wav, speaker, irony, winSize=10):
        self.name = wav
        self.speaker = speaker
        self.irony = irony
        self.winSize = winSize
        self.sound = parselmouth.Sound(wav)
        self.wav = WR(self.name)
        self.ampData = self.wav.getData()
        self.fs = self.wav.getSamplingRate()
        self.dur = self.wav.getDuration()
        self.f0Data, self.energyData = self.getF0Contour()
   

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
        energy_array = list()

        #This is the code to get the Parselmouth f0 contour
        pitch = self.sound.to_pitch_cc(pitch_floor=lowerLimit, pitch_ceiling=upperLimit)

        start = 0
        end = self.winSize / 1000

        while end <= self.wav.getDuration():

            t = pitch.get_value_at_time(start)
            r = call(self.sound, "Get root-mean-square", start, end)
            if np.isnan(t):
                t = 0
                
            f0_array.append(t)
            energy_array.append(r)
            start += self.winSize/1000
            end += self.winSize/1000

        return f0_array, energy_array
    

    def getMeanf0(self):
        return giveMean(self.f0Data) 

    def getSDf0(self):
        return giveSD(self.f0Data)

    def getRangef0(self):
        return (np.max(self.f0Data) - np.min(self.f0Data))

    def getMedianf0(self):
        f0 = self.f0Data
        mid = int(len(f0) / 2)
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
        self.hnr = [value[0] if value != -200 else 0 for value in harmonicity.values.transpose()]
        return self.hnr

    def getHNRstats(self):
        hnrMean = giveMean(self.hnr)
        hnrSD = giveSD(self.hnr)
        hnrRange = np.max(self.hnr) - np.min(self.hnr)

        return hnrMean, hnrRange, hnrSD

    #First 13 MFCCs
    def getMFCCs(self):
        ms = self.winSize/1000
        (rate, sig) = wave.read(self.name)
        mfccs = mfcc(sig, samplerate=rate, winlen=ms, winstep=ms)
        return mfccs

    #Perceptual Linear Prediction
    def getPLP(self):
        n = os.path.basename(self.name).split(".")[0]
        rastaFile = "../../Data/AcousticData/rastaplp_untransposed/{}.csv".format(n)
        rasta = genfromtxt(rastaFile, delimiter=',')
        return rasta

    #Amplitude modulation spectrum
    def getAMS(self):
        n = os.path.basename(self.name).split(".")[0]
        amsFile = "../../Data/AcousticData/ams_untransposed/{}.csv".format(n)
        ams = genfromtxt(amsFile, delimiter=',')
        return ams

    #Amplitude Features
    def getEnergyStats(self):
        energyRange = np.max(self.energyData) - np.min(self.energyData)
        energySD = giveSD(self.energyData)

        return energyRange, energySD, self.energyData

