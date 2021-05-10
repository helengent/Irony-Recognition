# Waveform.py -- create an empty vector with given length
# 
# Python (c) 2019 Yan Tang University of Illinois at Urbana-Champaign (UIUC)
#
# Created: March 13, 2019

import numpy as np


class Waveform:
    def __init__(self, nb_sample, sampleRate):
        self.nb_sample = nb_sample
        self.sampleRate = sampleRate
        self.data = np.zeros((self.nb_sample, 1))

    def count(self):
        return self.nb_sample