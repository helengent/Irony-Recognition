#!/usr/bin/env python3

import os, shutil, glob, sys
from scipy.signal import firwin, lfilter
import matplotlib.pyplot as plt
import numpy as np

sys.path.append(os.path.dirname(sys.path[1]))
from lib.WAVReader import WAVReader as WR
from lib.WAVWriter import WAVWriter as WW
from lib.DSP_Tools import findEndpoint, normaliseRMS, rms

def createDir(name):
    dirs = os.listdir()
    if name in dirs:
        shutil.rmtree(name)
    os.mkdir(name)

def plotIt(listoThings, x, listoLabels, xlabs, ylabs):
    fig, axs = plt.subplots(len(listoThings), constrained_layout=True)
    for i in range(len(listoThings)):
        axs[i].plot(x[i], listoThings[i])
        axs[i].set_title(listoLabels[i])
        axs[i].set(xlabel = xlabs[i], ylabel = ylabs[i])
    plt.show()

def highPass(fs, low, filterOrder):
    bs = firwin((filterOrder+1), (low/(fs/2)), pass_zero=False)
    return bs

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

def rmSIL():
    createDir("gated_test")

    files = glob.glob('../mono_test/*/*/*.wav')

    for item in files:

        reader = WR(item)
        signal = reader.getData()
        fs = reader.getSamplingRate()
        dur = reader.getDuration()
        nb_sample = reader.getSampleNO()

        lowerCutoff = 50
        order = 700
        original_rms = rms(signal)

        bs = highPass(fs, lowerCutoff, order)

        signal = np.squeeze(signal)
        filtered_signal = np.convolve(signal[:], bs)
        filtered_signal = np.reshape(filtered_signal, (len(filtered_signal), 1))

        win_size = 0.02

        silences = findEndpoint(filtered_signal, fs, win_size=win_size)
        silences = validateTs(silences[0])

        filtered_signal, k = normaliseRMS(filtered_signal, 0.01)

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

        gated_sig = filtered_signal[onset:offset]

        #Plotting to check filter efficacy
        toPlot = [signal, filtered_signal, silences, gated_sig]
        labels = ["Original Signal", "Filtered Signal", "Detected Silences", "Trimmed Signal"]
        xlabs = ["Time", "Time", "Time", "Time"]
        ylabs = ["Amplitude", "Amplitude", "Silence", "Amplitude"]
        spacing = np.linspace(0+dur/nb_sample, dur, nb_sample)
        filtered_spacing = np.linspace(0+dur/len(filtered_signal), dur, len(filtered_signal))
        win_spacing = np.linspace(0+dur/len(silences), dur, len(silences))
        new_dur = len(gated_sig)/fs
        gated_spacing = np.linspace(0+new_dur/len(gated_sig), new_dur, len(gated_sig))
        x = [spacing, filtered_spacing, win_spacing, gated_spacing]
        plotIt(toPlot, x, labels, xlabs, ylabs)
        
        #save trimmed signal to "gated" directory
        name = "gated/" + item[20:]
        writer = WW(name, gated_sig, fs=reader.getSamplingRate(), bits=reader.getBitsPerSample())
        #writer.write()

if __name__=="__main__":
    rmSIL()