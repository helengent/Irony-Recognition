"""
# DynamicRange.py -- A method calculating the dynamic range of a speech signal from its envelope
# 
# Python (c) 2020 Yan Tang University of Illinois at Urbana-Champaign (UIUC)
#
# Created: April 17, 2020
"""

from scipy.signal.filter_design import ellip, freqz
from scipy.signal import lfilter
import numpy as np

def calDynamicRange(x, fs, resolution = 512, bound_lower=.05, bound_upper=.99):

    ## Fullwave rectification
    x_rect = np.abs(x)

    ## Use a IIR filter to extract the "coarse" envelope of the input
    ## The filter is a simple 1st-order IIR filter with a cutoff of 160 Hz
    order = 1
    cutoff = 160
    b, a = ellip(order,rp = 1, rs = 33, Wn=160, btype="lowpass", fs=fs)
    
    ## filtering 
    x_env = lfilter(b, a, x_rect, axis=0)
    ## dB conversion and determine the levels relative to the peak value
    x_db = 20 * np.log10(x_env + np.finfo(float).eps)
    val_peak = np.max(x_db)
    val_relative = np.abs(x_db - val_peak)

    ## get the histogram for the energy distribution
    nb_bin = resolution
    bin_count, bin_val = np.histogram(val_relative, bins=resolution)

    ## Determins the lower and upper bounds using the given ratio aross all the entire range
    bound_lower = int(np.floor(resolution * bound_lower))
    bound_upper = int(np.ceil(resolution * bound_upper))

    ## compute the dynamic range in dB
    DR = bin_val[bound_upper] - bin_val[bound_lower]

    return DR
