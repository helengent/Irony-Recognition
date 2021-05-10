# DSP_Tools.py -- a set of functions required for basic DSP
#
# normaliseRMS: adjust root-mean-squre (RMS) of the signal to the target RMS.
# snr: compute speech-to-noise ratio of given speech and noise signals.
# rms: compute RMS of the signal
#
# Python (c) 2020 Yan Tang University of Illinois at Urbana-Champaign (UIUC)
#
# Created: March 14, 2019
# Modified: November 11, 2019
# Modified: Feburary 11, 2020 - Updated mkDataStructFMT to support 32-bit
# Modified: April 23, 2020 - Added LPC implementation for FP


import numpy as np
from numpy.linalg import inv


def normaliseRMS(x, tarRMS):
    # detech the number of channels in the signal
    # enable the function to deal with signals that contain more than one channel
    dim = np.shape(x)
    nb_sample = np.max(dim)  # number of samples
    nb_chan = np.min(dim)   # number of channels

    k = tarRMS * np.sqrt(nb_sample * nb_chan / (np.sum(x**2)))
    return k * x, k


def snr(s, n):
    return 10 * np.log10(np.sum(s**2) / np.sum(n**2))


def rms(x):
    # detech the number of channels in the signal
    # enable the function to deal with signals that contain more than one channel
    dim = np.shape(x)
    nb_sample = np.max(dim)  # number of samples
    nb_chan = np.min(dim)   # number of channels

    return np.sqrt(np.sum(x**2)/(nb_sample * nb_chan))


def scalesig(x, scalar=1.1):
    return x / (scalar * np.max(np.abs(x)))

# Make packing format for number-to-byte conversion and vise versa


def mkDataStructFMT(bits, total_samples):

    if bits == 8:
        tp = "b"  # 1-byte signed char
    elif bits == 16:
        tp = "h"  # 2-byte signed shorts
    elif bits == 32:
        tp = "i"  # 4-byte signed integer
    else:
        raise ValueError("Only supports 8-, 16-bit and 32-bit audio formats.")

    return "{}{}".format(total_samples, tp)


# Implementation of Convolution of two waveforms
# take the input waveform x and impluse response h
def convolve(x, h, durationMatch=False):
    len_x = len(x)
    len_h = len(h)
    len_y = len_x + len_h - 1

    # initialise an empty waveform for convoluation
    y = np.zeros((len_y, 1), dtype=np.float_)

    # match the lengh of the output vector
    data_x = np.concatenate(
        (x, np.zeros((len_y - len_x, 1), dtype=np.float)), axis=0)
    data_h = np.concatenate(
        (h, np.zeros((len_y - len_h, 1), dtype=np.float)), axis=0)

    # for each output sample in y
    idx_x = [i for i in range(0, len_y)]
    idx_h = np.flip(idx_x)
    for n in range(0, len_y):
        # tmp = []
        # for k in range(0, len_y):
        #     tmp.append(data_x[k] * data_h[n - k])
        # y.data[n] = np.sum(tmp)

        ri = np.roll(idx_h, n+1)  # shift for time delay
        y[n] = np.sum(data_x[idx_x] * data_h[ri])

    # return the convolved sequence
    return y


def energy(x):
    return np.sum(x**2)


def power(x):
    return energy(x)/len(x)


def ZCR(x):
    Nz = np.diff((x >= 0))

    return np.sum(Nz) / len(x)


def instPeriodicity(sig, lagstep=1, win_type="rect", fn="ac"):
    nb_sample = len(sig)
    if nb_sample % 2:
        sig = np.pad(sig, ((0, 1), (0, 0)))
        nb_sample += 1

    nb_win = int(nb_sample/2)

    if win_type == "hamming":
        win = np.hamming(nb_win).reshape(-1, 1)
    elif win_type == "rect":
        win = np.ones((nb_win, 1))
    else:
        raise ValueError(
            "Only support hamming ('hamming') and rectangular ('rect') window!")

    base = sig[0:nb_win] * win

    steps = range(0, nb_win, lagstep)
    nb_step = len(steps)
    pdt = np.zeros((nb_step,))
    for step in range(nb_step):
        frame_move = sig[step:step+nb_win, :] * win
        if fn.lower() == "ac":
            pdt[step] = np.dot(base.T, frame_move)
        elif fn.lower() == "amdf":
            pdt[step] = np.sum(np.abs(base - frame_move))
        else:
            raise ValueError(
                "Only support autocorrelatoin ('ac') and average magnitude difference function ('amdf')1")

    return pdt / np.max(pdt), base


def findEndpoint(sig, fs, win_size=0.02, threshold_en=50, threshold_zcr=0.05):
    """
    input:
        sig: input time series
        fs: sampling frequency
        win_size [default: 0.02]: window size in seconds
        hreshold_en [default: 50]: the drop from the peak energy in decibels
        threshold_zcr [default: 0.05]: the threshold for zero-crossing rate
    
    output:
        It returns an array for binaries with 1s indicating silence segments
    """
    
    nb_sample = len(sig)
    nb_spfrme = round(fs * win_size)
    nb_frame = int(np.ceil(nb_sample / nb_spfrme))

    nb_sample2 = nb_spfrme * nb_frame
    nb_pad = nb_sample2 - nb_sample
    if nb_sample2 > 0:
        sig = np.pad(sig, ((0, nb_pad), (0, 0)))

    wins = np.reshape(sig, (nb_frame, nb_spfrme)).T

    win_fn = np.hanning(nb_spfrme)
    en = np.zeros((1, nb_frame))
    zcr = np.zeros((1, nb_frame))
    for idx in range(nb_frame):
        seg = wins[:, idx] * win_fn + np.finfo(float).eps
        en[0, idx] = energy(seg)
        zcr[0, idx] = ZCR(seg)

    en_db = 20 * np.log10(en / np.sqrt(nb_spfrme))   
    lc_en = np.max(en_db) - threshold_en
    lc_zcr = threshold_zcr


    isSil = ((en_db < lc_en) & (zcr < lc_zcr))

    return isSil


def LPC(sig, order=3):
    """
    input:
        sig: the input signal from which the LPC coefficients are estimated
        order: LPC order

    output:
        It returns order+1 LPC coefficients, where the first coefficient is for y(0)
    """

def LPC(sig, order=4):
    nb_sample = len(sig)
    
    sig2 = sig[1:nb_sample, 0]
    X = np.pad(sig, ((0, order), (0,0)))

    # As holder
    A = np.zeros((nb_sample-1, order))

    # Make matrix for the Normal Equation
    for i in range(order):
        tmp = np.roll(X, i)
        A[:, i] = tmp[0:nb_sample-1,0]

    # Calculate As using the Normal Equation
    As = inv(A.T.dot(A)).dot(A.T).dot(sig2)

    return np.hstack((np.ones((1,)), -As))

