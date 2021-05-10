#!/usr/bin/env python3

import os
import sys
import math
import string
import librosa
import parselmouth
import numpy as np
import pandas as pd

from speaker import Speaker
from scipy.io import wavfile as wave
from lib.DynamicRange import calDynamicRange
from silence import highPass, validateTs
from lib.WAVReader import WAVReader as WR
from lib.DSP_Tools import findEndpoint

sys.path.append(os.path.join(os.path.dirname(os.path.dirname(sys.path[0])), 'ASR'))
import parseTextGrid


class Extractor:

    def __init__(self, wav, tg):
        
        self.name = wav # Short, "normalised" waveforms (3-10 seconds)
        self.id = os.path.basename(self.name).split('_')[1][0].upper()

        self.speaker = Speaker(self.id, "../SpeakerMetaData/{}_f0.txt".format(self.id), "../SpeakerMetaData/{}_avgDur.txt".format(self.id))

        self.tg = tg # The TextGrid - from the "full-wave enhanced"
        self.sound = parselmouth.Sound(self.name)
        self.wav = WR(self.name)
        self.data = self.wav.getData()
        self.fs = self.wav.getSamplingRate()
        self.dur = self.wav.getDuration()
        self.arpabetConsonantalList = ['K', 'S', 'L', 'M', 'SH', 'N', 'P', 'T', 'Z', 'W', 'D', 'B', 
                                        'V', 'R', 'NG', 'G', 'TH', 'F', 'DH', 'HH', 'CH', 'JH', 'Y', 'ZH']
        self.arpabetVocalicList = ['EH2', 'AH0', 'EY1', 'OY2', 'OW1', 'AH1', 'EH1', 'IH1', 'AA1', 
                                    'AY1', 'ER0', 'AE1', 'AE2', 'AO1', 'IH0', 'IY2', 'IY1', 'UH1', 
                                    'IY0', 'OY1', 'OW2', 'UW1', 'IH2', 'EH0', 'AO2', 'AA0', 'AA2', 
                                    'OW0', 'EY0', 'AE0', 'AW2', 'AW1', 'EY2', 'UW0', 'AH2', 'UW2', 
                                    'AO0', 'AY2', 'ER1', 'UH2', 'AY0', 'ER2', 'OY0', 'UH0', 'AW0']

        self.wordList = list()

        self.matrix, self.matrix_labels = self.getMatrix()


    # Landing space for extracting all values and placing them into an NxM matrix
    def getMatrix(self):

        dynamicRange = self.getDynamicRange()
        energy = self.getEnergy()
        intensity = self.getIntensity()
        zcr = self.getZeroCrossingRate()
        rms, spl = self.getRootMeanSquare()

        # averageWordDuration, averageSilenceDuration, and averageLaughDuration will be normalized by the speaker's overall average
        # if the speaker has a duration file in SpeakerMetaData
        averageWordDuration, averageSilenceDuration, averageLaughDuration = self.getAverageWordDuration()
        consonantalArray, consonantCount = self.getConsonantalInformation()
        vocalicArray, vowelCount = self.getVocalicInformation()
        try:
            consonantVowelRatio = consonantCount / vowelCount
        except:
            consonantVowelRatio = 0.0

        matrixList = [averageWordDuration, averageSilenceDuration, dynamicRange, energy, intensity, zcr, rms, spl, consonantVowelRatio]
        for c in consonantalArray:
            matrixList.append(c)
        for v in vocalicArray:
            matrixList.append(v)

        matrixLabelsList = ['Avg. Word Dur.', 'Avg. Sil. Dur.', 'Dynamic Range', 'Energy',
                            'Intensity', 'ZCR', 'Root Mean Square', 'Sound Pressure Level', 'Consonant Vowel Ratio']

        # Consonants
        for arpaC in self.arpabetConsonantalList:
            for moment in ['CoG', 'Kur', 'Ske', 'Std']:
                matrixLabelsList.append('{}_{}'.format(arpaC, moment))
        matrixLabelsList.append('Avg. Cons. Dur.')

        # Vowels
        for arpaV in self.arpabetVocalicList:
            for spacing in ['F0_1', 'F0_2', 'F0_3', 'F0_4', 'F0_5']:
                matrixLabelsList.append('{}-{}'.format(arpaV, spacing))
            for spacing in ['1', '2', '3', '4', '5']:
                for formant in ['F1', 'F2', 'F3']:
                    matrixLabelsList.append('{}-{}_{}'.format(arpaV, formant, spacing))
        matrixLabelsList.append('Avg. Voca. Dur.')

        assert len(matrixList) == len(matrixLabelsList), "You are missing labels or data."

        return matrixList, matrixLabelsList


    def returnMatrix(self):

        return self.matrix


    # Calculate the mean word duration for a particular speaker's utterance
    # We do this by looking at the TextGrid for the speaker and subsetting 
        # down to the 'word' tier. Here we further subset the TextGrid into 
        # millisecond-delimited utterances. Each utterance begins and ends with
        # a silence triplet. These are saved along with the words within the utterance
        # All durations are calculated and then averaged.   
    def getAverageWordDuration(self):

        _, words, _ = parseTextGrid.main(self.tg)

        wordDur, pauseDur, laughDur = list(), list(), list()

        for w in words:
            self.wordList.append(w[0])
            if w[0] == 'sp':
                pauseDur.append(w[-1])
            elif w[0] == "{LG}":
                laughDur.append(w[-1])
            else:
                wordDur.append(w[-1])

        # Normalize by average word/pause/laughter duration for specific speaker
        wordMean = (np.mean(wordDur) - self.speaker.avgWordDur) / self.speaker.avgWordDur
        pauseMean = (np.mean(pauseDur) - self.speaker.avgPauseDur) / self.speaker.avgPauseDur
        laughMean = (np.mean(laughDur) - self.speaker.avgLaughDur) / self.speaker.avgLaughDur

        return wordMean, pauseMean, laughMean


    # Dynamic Range in decibels
    def getDynamicRange(self):

        return calDynamicRange(self.data, self.fs)


    def getEnergy(self):

        energy = np.sum(self.data ** 2)
        return energy


    def getIntensity(self):

        intensity = self.sound.to_intensity().get_average()
        return intensity


    def getZeroCrossingRate(self):

        Nz = np.diff((self.data >= 0))

        return np.sum(Nz) / len(self.data)


    # If we're looking at the normalized files, this *should* all be the same, right?
    def getRootMeanSquare(self): 

        rms = np.sqrt(np.mean(self.data**2))
        spl = self.getSoundPressureLevel(rms)
        return rms, spl


    def getSoundPressureLevel(self, rms, p_ref = 2e-5):

        spl = 20 * np.log10(rms/p_ref)
        return spl


    # Landing space for extracting consonant-by-consonant metrics
    def getConsonantalInformation(self):

        # Consonants are returned only if they are in the current utterance
        _, _, phones = parseTextGrid.main(self.tg)
        consonantList = [p for p in phones if p[0] in self.arpabetConsonantalList]

        consonants = dict()
        for c in consonantList:
            if c[0] not in consonants.keys():
                consonants[c[0]] = [(c[1], c[2])]
            else:
                consonants[c[0]].append((c[1], c[2]))

        averageConsonantalDuration = (np.mean([c[-1] for c in consonantList]) - self.speaker.avgConsonantDur) / self.speaker.avgConsonantDur
        spectralArray = list()
        consonantCount = 0

        # Iterate over all of our possible consonants from ARPABET
        for arpaC in self.arpabetConsonantalList:
            # Check to see if the speaker produced the consonant during this particular utterance
            if arpaC in consonants:
                cog, kur, ske, std, count = self.getSpectralMoments(consonants[arpaC])
                spectralArray.append(cog)
                spectralArray.append(kur)
                spectralArray.append(ske)
                spectralArray.append(std)
                consonantCount += count
            else:
                for i in range(4): # Sometimes a speaker won't produced every consonant during every utterance; but we have to fill out the input feature vectors
                    spectralArray.append(np.nan)
        
        spectralArray.append(averageConsonantalDuration)
        return spectralArray, consonantCount


    def getSpectralMoments(self, c):

        cog, kur, ske, std, consonantCount = list(), list(), list(), list(), 0
        # Iterate over all of the time-stamps for the consonants produced during a given utterance
        for start, end in c:
            consonantCount += 1
            part = self.sound.extract_part(from_time = start, to_time = end)
            # Cast the waveform part to its spectrum
            spectrum = part.to_spectrum()
            # Calculate spectral moments
            cog.append(self.getConsonantalCenterOfGravity(spectrum))
            kur.append(self.getConsonantalKurtosis(spectrum))
            ske.append(self.getConsonantalSkewness(spectrum))
            std.append(self.getConsonantalStandardDeviation(spectrum))

        return np.mean(cog), np.mean(kur), np.mean(ske), np.mean(std), consonantCount


    def getConsonantalCenterOfGravity(self, spectrum):

        return spectrum.get_center_of_gravity()


    def getConsonantalKurtosis(self, spectrum):

        return spectrum.get_kurtosis()


    def getConsonantalSkewness(self, spectrum):

        return spectrum.get_skewness()


    def getConsonantalStandardDeviation(self, spectrum):

        return spectrum.get_standard_deviation()


    # Landing space for extracting vowel-by-vowel metrics
    def getVocalicInformation(self):

        # vowels are returned only if they are in the current utterance
        _, _, phones = parseTextGrid.main(self.tg)
        vowelList = [p for p in phones if p[0] in self.arpabetVocalicList]

        vowels = dict()
        for v in vowelList:
            if v[0] not in vowels.keys():
                vowels[v[0]] = [(v[1], v[2])]
            else:
                vowels[v[0]].append((v[1], v[2]))

        averageVowelDuration = (np.mean([v[-1] for v in vowelList]) - self.speaker.avgVowelDur) / self.speaker.avgVowelDur

        vocalicArray = list()
        vowelCount = 0
        for arpaV in self.arpabetVocalicList:
            if arpaV in vowels:
                information, count = self.getVocalicPitch(vowels[arpaV])
                vowelCount += count
                for i in information:
                    vocalicArray.append(i)
            else:
                for _ in range(20):
                    vocalicArray.append(np.nan)

        vocalicArray.append(averageVowelDuration)
        return vocalicArray, vowelCount


    def getVocalicPitch(self, v):

        tempArray, vowelCount = list(), 0
        for start, end in v:
            vowelCount += 1
            part = self.sound.extract_part(from_time = start, to_time = end, preserve_times = True)
            pitch = part.to_pitch_cc()
            spaces = np.linspace(start, end, num = 7)
            pitches = [pitch.get_value_at_time(i) for i in spaces[1:-1]]

            # Prep for Formant extraction
            burg = part.to_formant_burg()
            formants = np.array([self.getVocalicFormants(burg, i) for i in spaces[1:-1]]).flatten().tolist()
            
            # Save out this observation
            tempArray.append(pitches + formants)

        return np.nanmean(tempArray, axis = 0), vowelCount


    def getVocalicFormants(self, burg, i):

        outList = list()
        for j in range(1, 4):
            bark = burg.get_value_at_time(formant_number = j, time = i, unit = parselmouth.FormantUnit.HERTZ)
            outList.append(bark)

        return outList


    def getVocalicDuration(self, vowels):

        durList = list()
        for v in vowels:
            for start, end in vowels[v]:
                durList.append(end - start)

        return np.mean(durList)


