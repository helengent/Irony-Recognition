#!/usr/bin/env python3

import os
import sys
import shutil
import subprocess
import numpy as np
from glob import glob
from scipy.signal import decimate
from matplotlib import pyplot as plt
from scipy.signal import firwin, lfilter

sys.path.append(os.path.dirname(sys.path[0]))
from lib.WAVReader import WAVReader as WR
from lib.WAVWriter import WAVWriter as WW
from lib.DSP_Tools import rms, normaliseRMS, findEndpoint


def doubleCheck(wavPath):
    wavList = glob("../AudioData/Gated{}/*.wav".format(wavPath))
    rmsList = list()
    for wav in wavList:
        readr = WR(wav)
        data = readr.getData()
        rmsList.append(np.round(rms(data), 2))

    return len(list(set(rmsList))) == 1, list(set(rmsList))


def highPass(fs, low, filterOrder):
    bs = firwin((filterOrder+1), (low/(fs/2)), pass_zero=False)
    return bs


def createDir(name):
    dirs = os.listdir("../AudioData")
    if name in dirs:
        shutil.rmtree("../AudioData/{}".format(name))
    os.mkdir("../AudioData/{}".format(name))


def validateTs(T, nb_cw=2):
    newT = T[:]
    i = 0
    while i < (len(T)-1):
        if (T[i] == True) and (T[i+1] == False):
            n = i
            while (n < (len(T)-1)) and (T[n+1] == False):
                n += 1
            window = T[i+1:n+1]
            if len(window) < nb_cw:
                for t in range(i+1, n+1):
                    newT[t] = True
            i = n
        else:
            i+=1
    return(newT)


def trimSilence(signal, fs, win_size):

    signal = np.reshape(signal, (len(signal), 1))
    silences = findEndpoint(signal, fs, win_size=win_size)
    silences = validateTs(silences[0])

    #remove leading and trailing silence
    samples_per_window = fs * win_size
    onset_window = 0
    while silences[onset_window] == True:
        onset_window += 1

    offset_window = -1
    while silences[offset_window] == True:
        offset_window += -1

    onset = int(onset_window * samples_per_window)
    offset = int(offset_window * samples_per_window)

    gated_sig = signal[onset:offset]

    return gated_sig, silences


def preProcess(wav, tarRMS):
    readr = WR(wav)
    signal = readr.getData()
    fs = readr.getSamplingRate()
    dur = readr.getDuration()
    nb_sample = readr.getSampleNO()

    #normalize rms
    signal, k = normaliseRMS(signal, tarRMS)

    #trim leading and trailing silence
    gated_sig, silences = trimSilence(signal, fs, 0.005)

    return gated_sig, fs, readr.getBitsPerSample()


def findAvgRMS(wavs):
    rmsList = list()
    for wav in wavs:
        readr = WR(wav)
        data = readr.getData()
        rmsList.append(rms(data))

    return sum(rmsList)/len(rmsList)


def main(wavPath, k, attemptCount):

    if attemptCount > 100:
        print("Attempts have exceeded limit. Try something else.")
        print("k = {}".format(k))
        print("attempts = {}".format(attemptCount))
        raise Exception

    wavList = glob('../AudioData/temp{}/*.wav'.format(wavPath))

    createDir("Gated{}".format(wavPath))

    for i, wav in enumerate(wavList):

        if (i+1) % 10 == 0:
            print("Working on {}\tThis is file {}/{}".format(os.path.basename(wav), i+1, len(wavList)))

        newData, fs, bits = preProcess(wav, k)

        #Write newData out as new wavFile in fresh directory
        name = "../AudioData/Gated{}/".format(wavPath) + os.path.basename(wav)

        try:
            writer = WW(name, newData, fs=fs, bits=bits)
            writer.write()
        except:
            attemptCount += 1
            k -= 0.01
            print()
            print("Clipping problem. Starting over with new target rms")
            print("This is attempt {}. Target rms = {}".format(attemptCount, np.round(k, 2)))
            print()
            main(wavPath, k, attemptCount)
            break

        if i == len(wavList) - 1:

            allGood, t = doubleCheck(wavPath)

            if allGood:
                print("Successfully rms normalized all files to {} in {} attempts".format(np.round(k, 2), attemptCount))
                return
            else:
                print("Yo this still didn't work")
                print(t)
                print()
                # raise Exception


def preMain(wavPath):

    #convert stereo audio to mono
    #downsample to 16000 Hz
    bashCommand = "cd ../AudioData/{}; mkdir downsampled; ../../AcousticFeatureExtraction/preProcessing/monoDown.sh; mkdir ../temp{}; mv downsampled/* ../temp{}; rm -r downsampled; cd ../../AcousticFeatureExtraction".format(wavPath, wavPath, wavPath)
    subprocess.run(bashCommand, shell=True)

    wavList = glob('../AudioData/temp{}/*.wav'.format(wavPath))
    avgRMS = findAvgRMS(wavList)
    attemptCount = 1
    
    main(wavPath, avgRMS, attemptCount)

    #Get rid of temp folder. Files now live in Gated{wavPath}
    subprocess.run("rm -r ../AudioData/temp{}".format(wavPath), shell=True)

if __name__ == "__main__":

    preMain("Pruned3")

    