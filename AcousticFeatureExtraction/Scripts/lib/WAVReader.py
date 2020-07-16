# WAVReader.py -- a wrapper of Python Wave_read object, providing simple interface for students to read in a WAV file
#   It converts the raw data in bytes to amplitude values
# 
# 
# Python (c) 2020 Yan Tang University of Illinois at Urbana-Champaign (UIUC)
#
# Created: August 18, 2019
# Modified: November 09, 2019

import wave
import struct
import numpy as np
import DSP_Tools as dsp

class WAVReader():
    def __init__(self, wavfile):
        self.filename = wavfile
        self.__wav_obj = wave.open(self.filename, "rb")

        self.__fs = self.__wav_obj.getframerate()
        self.__bits = self.__wav_obj.getsampwidth() * 8
        self.__nb_chan = self.__wav_obj.getnchannels()
        self.__nb_sample = self.__wav_obj.getnframes()
        self.__data = self.__decode()
        
        self.__wav_obj.close()

    # Get the array of amplitude values
    def getData(self):
        return self.__data

    # Get the quantisation resolution in bits 
    def getBitsPerSample(self):
        return self.__bits

    # Get the sampling frequency/rate
    def getSamplingRate(self):
        return self.__fs

    # Get the number of channels in the file
    def getChannelNO(self):
        return self.__nb_chan

    # Get the number of sample points
    def getSampleNO(self):
        return self.__nb_sample

    # Get the duration of the signal in second
    def getDuration(self):
        return self.__nb_sample / self.__fs

    # Core function for coverting bytes to amplitude values
    def __decode(self):
        raw_bytes = self.__wav_obj.readframes(self.__nb_sample)
        total_samples = self.__nb_sample * self.__nb_chan

        fmt = dsp.mkDataStructFMT(self.__bits, total_samples)

        # Convert bytes to integers
        raw_int = struct.unpack(fmt, raw_bytes)

        # Dequantisation with given resolution
        data = np.array([float(int_quant) / pow(2, self.__bits - 1) for int_quant in raw_int])
        data.shape = (self.__nb_sample, self.__nb_chan)

        return data