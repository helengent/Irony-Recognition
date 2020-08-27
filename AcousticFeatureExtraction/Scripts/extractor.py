#!/usr/bin/env python3

import os
import sys
from sd import sd
import numpy as np
import parselmouth
import pandas as pd
from glob import glob
from speaker import Speaker
import matplotlib.pyplot as plt
from f0_parseltongue import smooth
from scipy.io import wavfile as wave
from silence import highPass, validateTs
from reaper_f0extractor import f0VecTime
from lib.WAVReader import WAVReader as WR
from limitsUpperLower import giveMean, giveSD
from python_speech_features import mfcc, logfbank
from lib.DSP_Tools import findEndpoint, normaliseRMS

def plotIt(listoThings, x):
    fig, axs = plt.subplots(len(listoThings), constrained_layout=True)
    for i in range(len(listoThings)):
        axs[i].plot(x[i], listoThings[i])
    plt.show()

class Extractor:
    def __init__(self, wav, text, speaker, irony):
        self.name = wav
        self.text = text
        self.speaker = speaker
        self.irony = irony
        self.sound = parselmouth.Sound(wav)
        self.wav = WR(self.name)
        self.ampData =  self.makeItMono(self.wav.getData())
        self.fs = self.wav.getSamplingRate()
        self.dur = self.wav.getDuration()
        #self.nb_samples = self.wav.getSampleNO()
        #self.dur = self.wav.getDuration()
        self.f0Data = self.getF0Contour()

    def makeItMono(self, stereoInput):
        newAudio = []
        for i in range(len(stereoInput)):
            newAudio.append((stereoInput[i][0]/2)+(stereoInput[i][1]/2))
        return newAudio
   
    #Speaker features
    def getSpeaker(self):
        return self.speaker.getSpeaker()

    def getGender(self):
        return self.speaker.getGender()

    #F0 features
    def getF0Contour(self):
        upperLimit = self.speaker.getUpperLimit()
        lowerLimit = self.speaker.getLowerLimit()

        #This is the code to get the Parselmouth f0 contour
        #pitch = self.sound.to_pitch(time_step=0.005, pitch_floor=lowerLimit, pitch_ceiling=upperLimit)
        #return pitch.selected_array['frequency']

        #This is the code to get the reaper f0 contour
        f0Contour = f0VecTime(self.text, lowerLimit, upperLimit)
        return f0Contour 

    def getMeanf0(self):
        return giveMean(self.f0Data) 

    def getSDF0(self):
        return giveSD(self.f0Data)

    #Timing features
    def findSilences(self):
        lowerCutoff = 40
        order = 1000
        bs = highPass(self.fs, lowerCutoff, order)
        ampData = np.array(self.ampData)
        ampData = np.squeeze(ampData)
        ampData = np.convolve(ampData[:], bs)
        ampData = np.reshape(ampData, (len(ampData), 1))
        silences = findEndpoint(ampData, self.fs, win_size=0.005)
        silences = validateTs(silences[0])
        return silences

    #HNR
    def getHNR(self):
        harmonicity = self.sound.to_harmonicity()
        hnr = harmonicity.values.mean()
        return hnr

    #Jitter and Shimmer
    def getJitter(self):
        jitter = 0
        return jitter

    def getShimmer(self):
        shimmer = 0
        return shimmer

    #First 13 MFCCs
    def getMFCCs(self):
        (rate, sig) = wave.read(self.name)
        mfccs = mfcc(sig, samplerate=rate, winlen=0.005)
        return mfccs[:13]

    #Perceptual Linear Prediction

    #Amplitude modulation spectrum

    #Relative spectral transform



if __name__ == "__main__":
    # #initiate speakers list:
    # B = Speaker("B", "../SpeakerF0Stats/B.txt", gender="m")
    # G = Speaker("G", "../SpeakerF0Stats/G.txt", gender="f")
    # P = Speaker("P", "../SpeakerF0Stats/P.txt", gender="f")
    # R = Speaker("R", "../SpeakerF0Stats/R.txt", gender="nb")
    # Y = Speaker("Y", "../SpeakerF0Stats/Y.txt", gender="m")
    # speakers = [B, G, P, R, Y]

    # #wavs = glob('../../AudioData/SmolWaves/*/*/*.wav')
    # wavs = glob('../../AudioData/TestWaves/*/*/*.wav')
    # wavs.sort()
    # #f0Files = glob('../ReaperF0Results/*/*/*.f0.p')
    # f0Files = [glob("../ReaperF0Results/*/*/{}.f0.p".format(os.path.basename(wav).split(".")[0])) for wav in wavs]
    # f0Files.sort()

    # masterList = []
    # #f0sequential = []

    # upperDur = 8.715

    # for i in range(len(wavs)):

    #     print("Working on file {} of {}".format(i, len(wavs)))

    #     f = open(f0Files[i][0], "r")
    #     f0text = f.read()
    #     f.close()
    #     f0text = f0text.split()

    #     wavfile = os.path.basename(wavs[i]).split(".")[0]

    #     #set speaker variable
    #     speaker = "NULL"
    #     for s in speakers:
    #         if wavfile[8].upper() == s.getSpeaker():
    #             speaker = s

    #     #set irony variable
    #     if wavfile[-1] == "I":
    #         irony = "i"
    #     else:
    #         irony = "n"

    #     #Normalize rms?

    #     extractor = Extractor(wavs[i], f0text, speaker, irony)
    #     mfccs = extractor.getMFCCs()
    #     #f0 = extractor.getF0Contour()
    #     silence = extractor.findSilences()
    #     dur = extractor.dur
        # nb_sample = extractor.wav.getSampleNO()
        # spacing = np.linspace(0+dur/nb_sample, dur, nb_sample)
        # win_spacing = np.linspace(0+dur/len(silence), dur, len(silence))
        # toPlot = [extractor.ampData, silence]
        # x = [spacing, win_spacing]
        # plotIt(toPlot, x)

