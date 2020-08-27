#!/usr/bin/env python3

import os
import shutil
import numpy as np
from glob import glob
from matplotlib import pyplot as plt
from scipy.signal import firwin, lfilter
from lib.WAVReader import WAVReader as WR
from lib.WAVWriter import WAVWriter as WW
from lib.DSP_Tools import rms, normaliseRMS, findEndpoint

def highPass(fs, low, filterOrder):
    bs = firwin((filterOrder+1), (low/(fs/2)), pass_zero=False)
    return bs

def createDir(name):
    dirs = os.listdir()
    if name in dirs:
        shutil.rmtree(name)
    os.mkdir(name)

def plotIt(listoThings, x, listoLabels, xlabs, ylabs, name):
    fig, axs = plt.subplots(len(listoThings), constrained_layout=True)
    for i in range(len(listoThings)):
        axs[i].plot(x[i], listoThings[i])
        axs[i].set_title(listoLabels[i])
        axs[i].set(xlabel = xlabs[i], ylabel = ylabs[i])
    plt.savefig(name)
    plt.clf()

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

def makeItMono(stereoInput):
    newAudio = np.zeros((len(stereoInput), 1))
    for i in range(len(stereoInput)):
        newAudio[i] = ((stereoInput[i][0]/2)+(stereoInput[i][1]/2))
    return newAudio

def preProcess(wav, tarRMS):
    readr = WR(wav)
    signal = readr.getData()
    fs = readr.getSamplingRate()
    dur = readr.getDuration()
    nb_sample = readr.getSampleNO()

    #make stereo input mono
    signal = makeItMono(signal)

    #normalize rms
    signal, k = normaliseRMS(signal, tarRMS)

    #trim leading and trailing silence
    #TODO investigate if a filter would help make this better?
    gated_sig, silences = trimSilence(signal, fs, 0.005)

    # Plotting to check filter efficacy
    toPlot = [signal, silences, gated_sig]
    labels = ["Original Signal", "Detected Silences", "Trimmed Signal"]
    xlabs = ["Time", "Time", "Time"]
    ylabs = ["Amplitude", "Silence", "Amplitude"]
    spacing = np.linspace(0+dur/nb_sample, dur, nb_sample)
    win_spacing = np.linspace(0+dur/len(silences), dur, len(silences))
    new_dur = len(gated_sig)/fs
    gated_spacing = np.linspace(0+new_dur/len(gated_sig), new_dur, len(gated_sig))
    x = [spacing, win_spacing, gated_spacing]
    name = "../../SilenceTrimmingPlots/" + os.path.basename(wav).split(".")[0] + ".pdf"
    plotIt(toPlot, x, labels, xlabs, ylabs, name)

    return signal, fs, readr.getBitsPerSample()

def findAvgRMS(wavs):
    rmsList = list()
    for wav in wavs:
        readr = WR(wav)
        data = readr.getData()
        rmsList.append(rms(data))

    return sum(rmsList)/len(rmsList)

if __name__ == "__main__":
    wavList = glob('../../AudioData/SmolWaves/*/*/*.wav')
    # wavList = glob('../../AudioData/TestWaves/*/*/*.wav')
    avgRMS = findAvgRMS(wavList)

    createDir("../../AudioData/Gated")
    createDir("../../SilenceTrimmingPlots")

    for i, wav in enumerate(wavList):

        print("Working on {}.\tThis is file {}\t{}".format(os.path.basename(wav), i, len(wavList)))

        newData, fs, bits = preProcess(wav, avgRMS)

        #TODO write newData out as new wavFile in fresh directory
        name = "../../AudioData/Gated/" + os.path.basename(wav)
        writer = WW(name, newData, fs=fs, bits=bits)
        writer.write()
    