#!/usr/bin/env python3

import os
import sys
import math
import string
import librosa
from numpy.lib.shape_base import apply_along_axis
import parselmouth
import numpy as np
import pandas as pd

from speaker import Speaker
from scipy.io import wavfile as wave
from lib.DynamicRange import calDynamicRange
from preProcessing.silence import highPass, validateTs
from lib.WAVReader import WAVReader as WR
from lib.DSP_Tools import findEndpoint

from preProcessing.ASR import parseTextGrid
from parselmouth.praat import call


class Extractor:

    def __init__(self, wav, tg):
        
        self.name = wav # Short, "normalised" waveforms (3-10 seconds)
        self.id = os.path.basename(self.name).split('_')[1][0].upper()

        self.speaker = Speaker(self.id, "../../Data/AcousticData/SpeakerMetaData/{}_f0.txt".format(self.id), dur_txt="../../Data/AcousticData/SpeakerMetaData/{}_avgDur.txt".format(self.id))

        self.tg = tg # The TextGrid
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
        self.available_sonorants = ['L', 'M', 'N', 'W', 'R', 'NG', 'Y']
        self.available_fricatives = ['S', 'SH', 'Z', 'V', 'TH', 'F', 'DH', 'HH', 'CH', 'JH', 'ZH']
        self.available_stops = ['K', 'P', 'T', 'D', 'B', 'G']

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
        fArray, spectralArray, stopArray, consonantCount, avgerageConsonantalDuration = self.getConsonantalInformation()
        vocalicArray, vowelCount = self.getVocalicInformation()
        try:
            consonantVowelRatio = consonantCount / vowelCount
        except:
            consonantVowelRatio = 0.0

        try:
            SyllPerSecond = vowelCount / self.dur
        except:
            #This shouldn't happen
            SyllPerSecond = 0.0

        matrixList = [averageWordDuration, averageSilenceDuration, dynamicRange, energy, intensity, zcr, rms, spl, consonantVowelRatio, SyllPerSecond]
        for f in fArray:
            matrixList.append(f)
        for s in spectralArray:
            matrixList.append(s)
        for s in stopArray:
            matrixList.append(s)
        matrixList.append(avgerageConsonantalDuration)
        for v in vocalicArray:
            matrixList.append(v)

        matrixLabelsList = ['Avg. Word Dur.', 'Avg. Sil. Dur.', 'Dynamic Range', 'Energy',
                            'Intensity', 'ZCR', 'Root Mean Square', 'Sound Pressure Level', 'Consonant Vowel Ratio', 'SyllPerSecond']

        # Consonants
        for arpaC in self.available_sonorants:
            for spacing in ['F0_1', 'F0_2', 'F0_3', 'F0_4', 'F0_5']:
                matrixLabelsList.append('{}-{}'.format(arpaC, spacing))
            for spacing in ['1', '2', '3', '4', '5']:
                for formant in ['F1', 'F2', 'F3']:
                    matrixLabelsList.append('{}-{}_{}'.format(arpaC, formant, spacing))
        for arpaC in self.available_fricatives:
            for moment in ['CoG', 'Kur', 'Ske', 'Std']:
                matrixLabelsList.append('{}_{}'.format(arpaC, moment))
        for arpaC in self.available_stops:
            for measure in ['VOT']:
                matrixLabelsList.append('{}_{}'.format(arpaC, measure))
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

        so = [c for c in consonantList if c[0] in self.available_sonorants]
        fr = [c for c in consonantList if c[0] in self.available_fricatives]
        st = [c for c in consonantList if c[0] in self.available_stops]

        consonants = dict()
        for c in consonantList:
            if c[0] not in consonants.keys():
                consonants[c[0]] = [(c[1], c[2])]
            else:
                consonants[c[0]].append((c[1], c[2]))

        averageConsonantalDuration = (np.mean([c[-1] for c in consonantList]) - self.speaker.avgConsonantDur) / self.speaker.avgConsonantDur
        consonantCount = len(consonantList)

        sonorants = dict()
        for sonorant in so:
            if sonorant[0] not in sonorants.keys():
                sonorants[sonorant[0]] = [(sonorant[1], sonorant[2])]
            else:
                sonorants[sonorant[0]].append((sonorant[1], sonorant[2]))
        
        fricatives = dict()
        for fricative in fr:
            if fricative[0] not in fricatives.keys():
                fricatives[fricative[0]] = [(fricative[1], fricative[2])]
            else:
                fricatives[fricative[0]].append((fricative[1], fricative[2]))

        stops = dict()
        for stop in st:
            if stop[0] not in stops.keys():
                stops[stop[0]] = [(stop[1], stop[2])]
            else:
                stops[stop[0]].append((stop[1], stop[2]))

        fArray = list()
        for s in self.available_sonorants:
            if s in sonorants:
                information = self.getVocalicPitch(sonorants[s])
                for i in information:
                    fArray.append(i)
            else:
                for _ in range(20):
                    fArray.append(np.nan)

        spectralArray = list()
        for f in self.available_fricatives:
            if f in fricatives:
                cog, kur, ske, std = self.getSpectralMoments(fricatives[f])
                spectralArray.append(cog)
                spectralArray.append(kur)
                spectralArray.append(ske)
                spectralArray.append(std)
            else:
                for i in range(4): # Sometimes a speaker won't produced every consonant during every utterance; but we have to fill out the input feature vectors
                    spectralArray.append(np.nan)

        stopArray = list()
        for s in self.available_stops:
            if s in stops:
                vot = np.mean([stop[1] - stop[0] for stop in stops[s]])
                stopArray.append(vot)
            else:
                for i in range(1):
                    stopArray.append(np.nan)

        return fArray, spectralArray, stopArray, consonantCount, averageConsonantalDuration



    def getSpectralMoments(self, c):

        cog, kur, ske, std = list(), list(), list(), list()
        # Iterate over all of the time-stamps for the consonants produced during a given utterance
        for start, end in c:
            part = self.sound.extract_part(from_time = start, to_time = end)
            # Cast the waveform part to its spectrum
            spectrum = part.to_spectrum()
            # Calculate spectral moments
            cog.append(self.getConsonantalCenterOfGravity(spectrum))
            kur.append(self.getConsonantalKurtosis(spectrum))
            ske.append(self.getConsonantalSkewness(spectrum))
            std.append(self.getConsonantalStandardDeviation(spectrum))

        return np.mean(cog), np.mean(kur), np.mean(ske), np.mean(std)


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
        vowelCount = len(vowelList)
        for arpaV in self.arpabetVocalicList:
            if arpaV in vowels:
                information = self.getVocalicPitch(vowels[arpaV])
                for i in information:
                    vocalicArray.append(i)
            else:
                for _ in range(20):
                    vocalicArray.append(np.nan)

        vocalicArray.append(averageVowelDuration)
        return vocalicArray, vowelCount


    def getVocalicPitch(self, v):

        tempArray = list()
        for start, end in v:
            part = self.sound.extract_part(from_time = start, to_time = end, preserve_times = True)

            try:
                pitch = part.to_pitch_cc()
                spaces = np.linspace(start, end, num = 7)
                pitches = [pitch.get_value_at_time(i) for i in spaces[1:-1]]

                # Prep for Formant extraction
                burg = part.to_formant_burg()
                formants = np.array([self.getVocalicFormants(burg, i) for i in spaces[1:-1]]).flatten().tolist()

            except:
                spaces = np.linspace(start, end, num = 7)
                pitches = [np.nan for i in range(10)]
                formants = [np.nan for i in range(10)]
            
            # Save out this observation
            tempArray.append(pitches + formants)

        return np.nanmean(tempArray, axis = 0)


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


