#!/usr/bin/env python3

import os
import math
import string
import librosa
import parselmouth
import numpy as np
import pandas as pd

from scipy.io import wavfile as wave
from DynamicRange import calDynamicRange
from silence import highPass, validateTs
from lib.WAVReader import WAVReader as WR
from lib.DSP_Tools import findEndpoint


class Extractor:

    def __init__(self, wav, tg, bw):
        
        self.name = wav # Short, "normalised" waveforms (3-10 seconds)
        self.id = self.name.split('-')[0].split('/')[-1]
        self.tg = tg # The TextGrid - from the "full-wave enhanced"
        self.bw = bw # The Big Waveform - "full wave enhanced"
        self.sound = parselmouth.Sound(self.name)
        self.bw_sound = parselmouth.Sound(self.bw)
        self.wav = WR(self.name)
        self.data = self.wav.getData()
        self.fs = self.wav.getSamplingRate()
        self.dur = self.wav.getDuration()
        self.arpabetConsonantalList = ['B', 'CH', 'D', 'DH', 'CX', 'EL', 'EM', 'EN', 'F', 'G',
                                       'H', 'JH', 'K', 'L', 'M', 'N', 'NX', 'NG', 'P', 'Q', 'R',
                                       'S', 'SH', 'T', 'TH', 'V', 'W', 'WH', 'Y', 'Z', 'ZH']
        self.arpabetVocalicList     = ['AA', 'AE', 'AH', 'AO', 'AW', 'AX', 'AXR', 'AY',
                                       'EH', 'ER', 'EY', 'IH', 'IX', 'IY', 'OW', 'OY', 
                                       'UH', 'UW', 'UX']
        self.matrix, self.matrix_labels = self.getMatrix()


    # Landing space for extracting all values and placing them into an NxM matrix
    def getMatrix(self):

        dynamicRange = self.getDynamicRange()
        energy = self.getEnergy()
        intensity = self.getIntensity()
        zcr = self.getZeroCrossingRate()
        rms, spl = self.getRootMeanSquare()

        # NOTE Average silence as a global metric; this has to be the case since the way the smaller waveforms are segmented gets rid of all space
        averageWordDuration, averageSilenceDuration = self.getAverageWordDuration()
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

        startWord, isSegment = False, False
        startData, wordList, wordDuration, silenceDuration = list(), list(), list(), list()
        counter = 0
        a, b = 0.0, 0.0
        # Each utterance-level (from Normalised_audio_chunks) waveform has a basename that can be segmented into 7 parts:
            # Speaker ID
            # ?? (an integer value, unique to the Speaker ID)
            # Chunk-level silence beginning (one chunk usually has 3 - 4 utterances taken from it)
            # Chunk-level silence ending (one chunk usually has 3 - 4 utterances)
            # Chunk number (integer value 1, 2, 3, 4 etc.)
            # Utterance-level silence offset (added to the chunk-level silence beginning to find start)
            # Utterance-level silence offset (added to the chunk-level silence beginning to find end)
        _, _, startSIL, _, _, offsetA, offsetB = self.name.split('_')[0].split('-')
        startSIL = int(startSIL) + int(offsetA) # ex. 6666
        endSIL = int(startSIL) + int(offsetB) # ex. 7736
        with open(self.tg, 'r') as f:
            lastLine = "" # Keep an eye on what we saw previous to this line
            for line in f:
                if startWord: # startWord is flipped to True after we have gotten past the 'phone' IntervalTier
                    if counter == 0: # At the beginning of a Tier there is a duration start (usually 0 here)
                        startData.append(0)
                        counter += 1
                    elif counter == 1: # Tier-beginning duration maximum
                        startData.append(float(line.strip()))
                        counter += 1
                    elif counter == 2: # Tier-beginning number of items in the tier (here words)
                        startData.append(int(line.strip()))
                        counter = 10
                    elif counter == 10: # We have finally hit the triplets; this is the start of each word duration
                        a = float(line.strip())
                        counter += 1
                    elif counter == 11: # End of each word duration
                        b = float(line.strip())
                        counter += 1
                    elif counter == 12: # Actual word segment, could be silence
                        if a < (startSIL / 1000) < b: # Convert from milliseconds to seconds for the comparison here
                            isSegment = True # We have found the beginning of this utterance-level segment
                        elif a < (endSIL / 1000) < b:
                            isSegment = False # We have found the end of this utterance-level segment
                        if isSegment: # We are looking at the correct window for the utterance; go ahead and append the word and silence durations.
                            if line != '"SIL"\n':
                                wordList.append(line.strip())
                                wordDuration.append(b - a)
                            else: silenceDuration.append(b - a)
                        counter = 10 # Reset to the beginning of a new triplet
                    
                else:
                    if line == '"word"\n' and lastLine == '"IntervalTier"\n': # Make sure we skip over the 'phone' IntervalTier
                        startWord = True
                    lastLine = line

        self.wordList = wordList # We're not currently using this, but it might be nice to do an analysis of the types of words used, if we want to get into lexical stuff
        return np.mean(wordDuration), np.mean(silenceDuration) # Return a global average of both silence and word duration for a given speaker


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
        consonants = self.getConsonants()
        
        # Check to make sure we don't have anything non-ARPABET
        for c in consonants:
            if c not in self.arpabetConsonantalList:
                print("There are consonants not known to the ARPABET list: {}".format(c))
                raise AssertionError

        averageConsonantalDuration = self.getConsonantalDuration(consonants)
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


    def getConsonants(self):

        # For comments look to the wordDuration() call
        startWord, isSegment = False, False
        startData, wordList, wordDuration, silenceDuration = list(), list(), list(), list()
        counter = 0
        totalDur = 0.0
        a, b = 0.0, 0.0
        consonants = dict()
        _, _, startSIL, _, _, offsetA, offsetB = self.name.split('_')[0].split('-')
        startSIL = int(startSIL) + int(offsetA)
        endSIL = int(startSIL) + int(offsetB)
        with open(self.tg, 'r') as f:
            lastLine = ""
            for line in f:
                if line == '"IntervalTier"\n' and b == totalDur:
                    return consonants
                if line == '<exists>\n':
                    totalDur = float(lastLine.strip())
                if startWord:
                    if counter == 0:
                        startData.append(0)
                        counter += 1
                    elif counter == 1:
                        startData.append(float(line.strip()))
                        counter += 1
                    elif counter == 2:
                        startData.append(int(line.strip()))
                        counter = 10
                    elif counter == 10:
                        a = float(line.strip())
                        counter += 1
                    elif counter == 11:
                        b = float(line.strip())
                        counter += 1
                    elif counter == 12:
                        if  a < (startSIL / 1000) < b:
                            isSegment = True
                        elif a < (endSIL / 1000) < b:
                            isSegment = False
                        # It's a consonant that we're interested in if it passes the .isdigit() check
                        # LG is laughter, NS is noise, CG is cough, LS is lip smack, and SL can be silence  (https://phon.wordpress.ncsu.edu/lab-manual/forced-alignment/)
                        if isSegment:
                            if line != '"sil"\n' and line != '"lg"\n' and line != '"ns"\n' and line != '"cg"\n' and line != '"ls"\n' and line != '"sl"\n' and not any(map(str.isdigit, line.strip())):
                                consonant = line.strip().translate(str.maketrans('', '', string.punctuation))
                                if consonant == "HH":
                                    consonant = "H" # We'll just collapse this category and leave the NX/NG distinction to allow for the voiced nasal alveolar flap in /winner/
                                if consonant not in consonants:
                                    consonants[consonant] = list()
                                consonants[consonant].append((a,b))
                        counter = 10
                else:
                    if line == '"phones"\n' and lastLine == '"IntervalTier"\n':
                        startWord = True
                    lastLine = line


    def getSpectralMoments(self, c):

        cog, kur, ske, std, consonantCount = list(), list(), list(), list(), 0
        # Iterate over all of the time-stamps for the consonants produced during a given utterance
        for start, end in c:
            consonantCount += 1
            # TODO currently we are extracting from the large waveform and not from the normalised ones. 
                # I need to figure out the mapping here from big to small with the TextGrids
            part = self.bw_sound.extract_part(from_time = start, to_time = end)
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


    # Return a mean for all consonants produced over the course of the utterance
    def getConsonantalDuration(self, consonants):

        durList = list()
        for c in consonants:
            for start, end in consonants[c]:
                durList.append(end - start)

        return np.mean(durList)


    # Landing space for extracting vowel-by-vowel metrics
    def getVocalicInformation(self):

        vowels = self.getVowels()
        
        # Check to make sure we don't have anything non-ARPABET
        for v in vowels:
            if v not in self.arpabetVocalicList: # The -1 indexes for the stress, which we may want to look into later
                print("There are consonants not known to the ARPABET list: {}".format(v))
                raise AssertionError

        averageVocalicDuration = self.getVocalicDuration(vowels)
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

        vocalicArray.append(averageVocalicDuration)
        return vocalicArray, vowelCount


    def getVowels(self):

        startWord, isSegment = False, False
        startData, wordList, wordDuration, silenceDuration = list(), list(), list(), list()
        counter = 0
        totalDur = 0.0
        a, b = 0.0, 0.0
        vowels = dict()
        _, _, startSIL, _, _, offsetA, offsetB = self.name.split('_')[0].split('-')
        startSIL = int(startSIL) + int(offsetA)
        endSIL = int(startSIL) + int(offsetB)
        with open(self.tg, 'r') as f:
            lastLine = ""
            for line in f:
                if line == '"IntervalTier"\n' and b == totalDur:
                    return vowels
                if line == '<exists>\n':
                    totalDur = float(lastLine.strip())
                if startWord:
                    if counter == 0:
                        startData.append(0)
                        counter += 1
                    elif counter == 1:
                        startData.append(float(line.strip()))
                        counter += 1
                    elif counter == 2:
                        startData.append(int(line.strip()))
                        counter = 10
                    elif counter == 10:
                        a = float(line.strip())
                        counter += 1
                    elif counter == 11:
                        b = float(line.strip())
                        counter += 1
                    elif counter == 12:
                        if a < (startSIL / 1000) < b:
                            isSegment = True
                        elif a < (endSIL / 1000) < b:
                            isSegment = False
                        # It's a vowel we're interested in
                        # LG is laughter, NS is noise, CG is cough, LS is lip smack, and SL can be silence  (https://phon.wordpress.ncsu.edu/lab-manual/forced-alignment/)
                        if isSegment:
                            if line != '"sil"\n' and line != '"lg"\n' and line != '"ns"\n' and line != '"cg"\n' and line != '"ls"\n' and line != '"sl"\n' and any(map(str.isdigit, line.strip())):
                                vowel = line.strip().translate(str.maketrans('', '', string.punctuation))[:-1]
                                if vowel not in vowels:
                                    vowels[vowel] = list()
                                vowels[vowel].append((a,b))
                        counter = 10
                else:
                    if line == '"phones"\n' and lastLine == '"IntervalTier"\n':
                        startWord = True
                    lastLine = line


    def getVocalicPitch(self, v):

        tempArray, vowelCount = list(), 0
        for start, end in v:
            vowelCount += 1
            part = self.bw_sound.extract_part(from_time = start, to_time = end, preserve_times = True)
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

    # This is the old version that we got the 87.15% accuracy on with LOO-cross validation with baseline neural model and 230 kbest feature space reduction
    # Hopefully Yan's auditory Dynamic Range calculation works better since it's in dB
    # def getDynamicRange(self):

    #     x_min = np.min(self.data)
    #     x_max = np.max(self.data)
    #     R = np.abs(x_max - x_min)
    #     return R

    # # https://www.programmersought.com/article/6348312465/
    # # Old version prior to RMS normalization
    # def getZeroCrossingRate(self):

    #     frameSize = 512
    #     overLap = 0
    #     wavLen = len(self.data)
    #     step = frameSize - overLap
    #     frameNum = math.ceil(wavLen/step)
    #     zcr = np.zeros((frameNum, 1))
    #     for i in range(frameNum):
    #         currentFrame = self.data[np.arange(i*step, min(i * step + frameSize, wavLen))]
    #         currentFrame = currentFrame - np.mean(currentFrame) # Zero-justified
    #         zcr[i] = sum(currentFrame[0:-1] * currentFrame[1::]<=0)

    #     return np.mean(zcr)
