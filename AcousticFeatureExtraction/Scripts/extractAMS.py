#!/usr/bin/env python3

import math
import numpy as np
from glob import glob
import scipy.sparse as sps
from lib.WAVReader import WAVReader as WR
from scipy.signal import butter, lfilter, decimate

def LTLAdjust(x, fs):
    max_val = np.max(x)
    x = x/max_val*0.4
    ratio = 0.4/max_val

    return x, ratio

def mel(N, low, high):

    # % This function returns the lower, center and upper freqs
    # % of the filters equally spaced in mel-scale
    # % Input: N - number of filters
    # % 	 low - (left-edge) 3dB point of the first filter
    # %	 high - (right-edge) 3dB point of the last filter
    # %
    # % Copyright (c) 1996 by Philipos C. Loizou
    # % 

    ac=1000
    fc=800
    
    DBG = 0
    LOW = ac * np.log(1+low/fc)
    HIGH = ac * np.log(1+high/fc)
    N1 = N + 1
    e1 = math.exp(1)

    fmel, cen2 = list(), list()
    for i in range(1, N1+1):
        fmel.append(LOW + i * (HIGH-LOW)/N1)

    for item in fmel:
        cen2.append(fc * (e1 ** (item/ac)-1))

    lower = cen2[0:N]
    upper = cen2[1:N+1]
    center = list()
    for i in range(N):
        center.append(0.5 * (lower[i] + upper[i]))

    return lower, center, upper

def create_crit_filter(nFFT, num_crit, Srate, opt):
    crit_filter = np.zeros((int(num_crit),int(np.ceil(nFFT/2))))

    low_f, up_f, lower_ind, upper_ind = dict(), dict(), dict(), dict()

    if opt == 1:

        cent_freq, bandwidth = dict(), dict()
        freqs = [50.0000, 120.000, 190.000, 260.000, 330.000, 400.000, 470.000, 540.000, 617.372, 703.378, 
             798.717, 904.128, 1020.38, 1148.30, 1288.72, 1442.54, 1610.70, 1794.16, 1993.93, 2211.08, 
             2446.71, 2701.97, 2978.04, 3276.17, 3597.63]
        bandwidths = [70.0000, 70.0000, 70.0000, 70.0000, 70.0000, 70.0000, 70.0000, 77.3724, 86.0056, 95.3398, 
                  105.411, 116.256, 127.914, 140.423, 153.823, 168.154, 183.457, 199.776, 217.153, 235.631, 
                  255.255, 276.072, 298.126, 321.465, 346.136]

        for i in range(1, 26):
            cent_freq[i] = freqs[i]
            bandwidth[i] = bandwidths[i]
        
        # equal weights, non-overlap grouping strategy
        for i in range(1, num_crit+1):
            low_f[i] = cent_freq[i] - bandwidth[i]/2
            up_f[i] = cent_freq[i] + bandwidth[i]/2
            lower_ind[i] = np.ceil(low_f[i]/Srate*nFFT)
            upper_ind[i] = np.ceil(up_f[i]/Srate*nFFT)
            if i>1:
                if lower_ind[i]<=upper_ind[i-1]:
                    lower_ind[i] = upper_ind[i-1] + 1
            if upper_ind[i]<lower_ind[i]:
                upper_ind[i] = lower_ind[i]
            crit_filter[i,lower_ind[i]:(upper_ind[i]+1)] = 1 #/(upper_ind(i) - lower_ind(i) + 1);

        CB2FFT = np.transpose(crit_filter)

        for i in range(1, (nFFT/2)+1):
            S = sum(CB2FFT[i,:])
            if S>0:
                CB2FFT[i,:] = CB2FFT[i,:]/S
        
    else:
        lower, cent_freq, upper, = mel(num_crit, 0, Srate/2)
        bandwidth = [np.round(u-l) for u, l in zip(upper, lower)]
        cent_freq = np.round(cent_freq)
        
        for i in range(1, num_crit):
            low_f[i] = cent_freq[i-1] - bandwidth[i-1]/2
            up_f[i] = cent_freq[i-1] + bandwidth[i-1]/2
            lower_ind[i] = np.ceil(low_f[i]/Srate*nFFT)
            upper_ind[i] = np.ceil(up_f[i]/Srate*nFFT)
            if i>1:
                if lower_ind[i]<=upper_ind[i-1]:
                    lower_ind[i] = upper_ind[i-1] + 1
            if upper_ind[i]<lower_ind[i]:
                upper_ind[i] = lower_ind[i]

            crit_filter[i-1,int(lower_ind[i]):(int(upper_ind[i])+1)] = 1 #/(upper_ind(i) - lower_ind(i) + 1);

        # initialize the inverse filter
        CB2FFT = np.transpose(crit_filter)

        for i in range(int(np.ceil(nFFT/2))):
            S = sum(CB2FFT[i,:])
            if S>0:
                CB2FFT[i,:] = CB2FFT[i,:]/S

    return crit_filter, CB2FFT

def AMS_init_FFT(nFFT_env, nFFT_speech, nFFT_ams, nChnl, Srate):

    opt_spacing = 2; #2 is mel
    crit_filter_SNR, CB2FFT = create_crit_filter(nFFT_speech, nChnl, Srate, opt_spacing)
    crit_filter = create_crit_filter(nFFT_env, nChnl, Srate, opt_spacing)

    # uniform triangular filterbank
    st = 1 # starting fft point

    #  ~400Hz
    ed = 27 # ending fft point
    nCh = 15
    # is_modified.fb = 'uniform_trib_15'

    step = (ed-st)/(nCh+1)
    centers = list()
    for i in range(1, nCh+1):
        centers.append(st + i * step)

    slope = 1/step # slope of triangular filter
    MF_T = np.zeros((nCh,int(np.ceil(nFFT_ams/2))))
    for n in range(nCh):
        for i in range(int(np.ceil(centers[n]-step)), int(np.floor(centers[n]+1))):
            MF_T[n,i] = slope * (i-centers[n]) + 1

        for i in range(int(np.ceil(centers[n])), int(np.floor(centers[n]+step)+1)):   
            MF_T[n,i] = -slope * (i-centers[n]) + 1

    parameters = dict()

    parameters["CB2FFT"] = CB2FFT
    parameters["crit_filter"] = crit_filter
    parameters["nFFT_env"] = nFFT_env
    parameters["nFFT_speech"] = nFFT_speech
    parameters["nFFT_ams"] = nFFT_ams
    parameters["nChnl"] = nChnl
    parameters["MF_T"] = MF_T
    parameters["Srate"] = Srate
    parameters["crit_filter_SNR"] = crit_filter_SNR

    return parameters

def gen_analys_filter(nFFT, nChnl, Srate):

    # Generate the analysis filter bank and 
    # the mapping matrix of BM from the channel domain to FFT domain

    FB2FFT = np.zeros((int(np.ceil(nFFT/2)), nChnl))
    nOrd = 6
    FS = Srate/2

    lower, cent_freq, upper = mel(nChnl, 0, Srate/2)
    bandwidth = [np.round(u-l) for u, l in zip(upper, lower)]
    cent_freq = np.round(cent_freq)

    low_f, up_f, lower_ind, upper_ind = dict(), dict(), dict(), dict()
    
    for i in range(1, nChnl+1):
        low_f[i] = cent_freq[i-1] - bandwidth[i-1]/2
        up_f[i] = cent_freq[i-1] + bandwidth[i-1]/2
        lower_ind[i] = np.ceil(low_f[i]/Srate*nFFT)
        upper_ind[i] = np.ceil(up_f[i]/Srate*nFFT)
        if i>1:
            if lower_ind[i]<=upper_ind[i-1]:
                lower_ind[i] = upper_ind[i-1] + 1
        if upper_ind[i]<lower_ind[i]:
            upper_ind[i] = lower_ind[i]

        FB2FFT[int(lower_ind[i]):(int(upper_ind[i])+1), i-1] = 1 #/(upper_ind(i) - lower_ind(i) + 1);

    FB2FFT = np.append(FB2FFT, np.zeros((1, np.shape(FB2FFT)[1])), axis=0)
    FB2FFT = np.append(FB2FFT, [FB2FFT[-2], FB2FFT[-3]], axis=0)

    FB2FFT = sps.csr_matrix(FB2FFT)

    analys_filter = dict()

    analys_filter["A"] = np.zeros((nChnl,nOrd+1))
    analys_filter["B"] = np.zeros((nChnl,nOrd+1))
    useHigh = 1
    for i in range(1, nChnl+1):
        W1 = [low_f[i]/FS, up_f[i]/FS]
        if W1[1] >= 1:
            W1[1] = 0.999
        if i == nChnl:
                b, a = butter(nOrd, W1[1], btype='high')
        else:
            b, a = butter(nOrd, W1[0]) #This is a guess because scipy.signal.butter() demanded it be a scalar

        analys_filter["B"][i-1,:] = b
        analys_filter["A"][i-1,:] = a

    return analys_filter, FB2FFT

def AMS_init(nFFT_speech, nFFT_ams, nChnl, Srate):

    analys_filter, FB2FFT = gen_analys_filter(nFFT_speech, nChnl, Srate)

    # FFT MF->Selected Modulation Frequency Transformation
    MF_T = np.zeros((15,26))
    MF_T[0,0:3] = [0.4082, 0.8165, 0.4082]
    MF_T[1,1:5] = [0.3162, 0.6325, 0.6325, 0.3162]
    MF_T[2,2:6] = [0.3162, 0.6325, 0.6325, 0.3162]
    MF_T[3,3:6] = [0.4082, 0.8165, 0.4082]
    MF_T[4,3:7] = [0.3162, 0.6325, 0.6325, 0.3162]
    MF_T[5,4:8] = [0.3162, 0.6325, 0.6325, 0.3162]
    MF_T[6,5:9] = [0.3162, 0.6325, 0.6325, 0.3162]
    MF_T[7,7:10] = [0.4082, 0.8165, 0.4082]
    MF_T[8,8:12] = [0.3162, 0.6325, 0.6325, 0.3162]
    MF_T[9,10:13] = [0.4082, 0.8165, 0.4082]
    MF_T[10,12:15] = [0.4082, 0.8165, 0.4082]
    MF_T[11,14:17] = [0.4082, 0.8165, 0.4082]
    MF_T[12,16:19] = [0.4082, 0.8165, 0.4082]
    MF_T[13,19:22] = [0.4082, 0.8165, 0.4082]
    MF_T[14,22:25] = [0.4082, 0.8165, 0.4082]

    win = np.hanning(nFFT_speech)

    R = Srate/4000

    lp_B, lp_A = butter(6, 400/Srate*R)

    parameters = dict()
    parameters["analys_filter"] = analys_filter
    parameters["FB2FFT"] = FB2FFT
    parameters["nFFT_speech"] = nFFT_speech
    parameters["nFFT_ams"] = nFFT_ams
    parameters["nChnl"] = nChnl
    parameters["MF_T"] = MF_T
    parameters["Srate"] = Srate
    parameters["step"] = nFFT_speech/2
    parameters["win"] = win
    parameters["env_choice"] = 'abs'
    parameters["lp_B"] = lp_B
    parameters["lp_A"] = lp_A
    parameters["R"] = R

    return parameters

def FB_filter(sig, parameters):

    FB = parameters["analys_filter"]
    nChnl = parameters["nChnl"]

    A = FB["A"]
    B = FB["B"]

    sig_sub = np.zeros((nChnl, len(sig)))

    for n in range(nChnl):
        sig_sub[n,:] = lfilter(B[n,:], A[n,:], np.squeeze(sig))

    return sig_sub

def env_extraction(SIG, parameters):

    # SIG is a matrix of the input signals in subbands
    # ENV is a matrix of the envelope of the input in subbands

    R = parameters["R"]
    choice = parameters["env_choice"]
    lp_A = parameters["lp_A"]
    lp_B = parameters["lp_B"]
    nChnl = parameters["nChnl"]

    dSIG = SIG

    if choice == 'abs':
        ENV = np.abs(dSIG)
    elif choice == "square":
        ENV = np.abs(dSIG**2)
    else:
        print('Unknown envelope detection strategy\n')

    # decimation
    env = list()
    for n in range(nChnl):
        env.append(decimate(ENV[n], int(np.round(R)), ftype='fir'))

    env = np.asarray(env)

    return env

def extractAMS(x, fs, nChnl, nb_frames):

    length = np.floor(4*fs/1000); # 4ms, frame size in samples, envelope length
    if length % 2 ==1:
        length = length+1

    env_step = 0.25; # 1.00ms or 0.25ms, advance size, envelope step
    length2 = np.floor(env_step*fs/1000)
    Nframes = np.floor(len(x)/length2)-length/length2+1
    fs_env = 1/(env_step/1000) # Since we calculate the envelope every 0.25ms, the sampling rate for envelope is this.
    win = np.hanning(length)
    s_frame_len = 20 #32ms for each frame

    nFFT_speech = s_frame_len/1000*fs
    AMS_frame_len = s_frame_len/env_step # 128 frames of envelope corresponding to 128*0.25 = 32ms
    AMS_frame_step = AMS_frame_len/2 # step size

    nFFT_env = AMS_frame_len
    nFFT_ams = AMS_frame_len*2

    k = 1 # sample position of the speech signal
    kk = 1
    KK = nb_frames
    ss = 1 # sample position of the noisy speech for synthesize
    ns_ams = np.zeros((nChnl*15,KK))

    parameters = AMS_init_FFT(nFFT_env, nFFT_speech, nFFT_ams, nChnl, fs)
    parameters_FB = AMS_init(nFFT_speech, 20, nChnl, fs) #64 isn't used in this routine

    X_sub = FB_filter(x, parameters_FB) # time domain signals in subbands

    ENV_x = env_extraction(X_sub, parameters_FB) #time domain envelope in subbands

    ns_env = ENV_x

    win_ams = np.hanning(AMS_frame_len)
    repwin_ams = np.tile(win_ams, (1, nChnl))

    for kk in range(1, KK+1):
        start_idx = AMS_frame_step*(kk-1)
        end_idx = 1 + (AMS_frame_len + (AMS_frame_step*(kk-1)))
    
        if end_idx<=np.shape(ns_env)[1]:
            ns_env_frm = ns_env[:, int(start_idx):int(end_idx)]
        else:
            zero_padding =  np.zeros((np.shape(ns_env)[0], int(end_idx) - np.shape(ns_env)[1]))
            ns_env_frm = ns_env[:,int(start_idx):len(ns_env)]
            ns_env_frm = np.append(ns_env_frm, zero_padding, axis=0)

        ns_env_frm = np.transpose(ns_env_frm)
        ams = np.abs(np.fft.fft(ns_env_frm*repwin_ams,int(nFFT_ams)))
        # ams = parameters["MF_T"]*ams[0:int(np.round(nFFT_ams/2)), :]
        ams = parameters["MF_T"].dot(ams[0:int(np.round(nFFT_ams/2)), :])
        ams = np.transpose(ams)
        newSize = np.shape(ams)[0] * np.shape(ams)[1]
        ns_ams[:,kk] = np.reshape(ams, (newSize, 1), order="F")

    return ns_ams


if __name__ == "__main__":
    wavs = glob('../../AudioData/GatedAll/*.wav')

    for i, wav in enumerate(wavs):

        print("Working on file {} of {}".format(i, len(wavs)))
        readr = WR(wav)
        x = readr.getData()
        fs = readr.getSamplingRate()
        nChnl = readr.getChannelNO()
        nb_frames = readr.getSampleNO()

        x, ratio = LTLAdjust(x, fs)

        ns_ams = extractAMS(x, fs, nChnl, nb_frames)
        print(ns_ams)