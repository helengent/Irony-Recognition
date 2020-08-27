# WAVWriter.py -- a wrapper of Python Wave_write object, providing simple interface for students to write data to a WAV file
#   It converts amplitude values into raw data in bytes
# 
# Python (c) 2020 Yan Tang University of Illinois at Urbana-Champaign (UIUC)
#
# Created: August 18, 2019
# Modified: November 09, 2019

import wave
import struct
import numpy as np
import lib.DSP_Tools as dsp

class WAVWriter():
    def __init__(self, wavfile, data, fs=44100, bits=16):
        # Path of output wav file
        self.filename = wavfile
        # Samples to write out
        self.data = data
        # sampling frequency/rate
        self.fs = fs
        # Quatisation resolution
        self.bits = bits

        # Check the dimmension of the input matrix, in order to determine how data is organised.
        # Conventionally, row for sample points and column for channels 
        dim = self.data.shape
        if dim[0] < dim[1]:
            self.__SampleNo =  dim[1]
            self.__ChannelNO = dim[0]
        else:
            self.__SampleNo =  dim[0]
            self.__ChannelNO = dim[1]

    # Write encoded data (in bytes) to wav file
    def write(self):
        # Initialised a Wave_write object
        wav_obj = wave.open(self.filename, "wb")
        wav_obj.setnchannels(self.__ChannelNO)
        wav_obj.setsampwidth(int(self.bits / 8)) # convert bits to bytes
        wav_obj.setframerate(self.fs)
        wav_obj.setnframes(self.__SampleNo)

        wav_obj.writeframes(self.__encode())
        wav_obj.close()

    # Core function for coverting amplitude values to bytes
    def __encode(self):
        total_samples = self.__SampleNo * self.__ChannelNO
        array_data = self.data
        if np.max(np.abs(array_data)) > 1:
            array_data = dsp.scalesig(array_data) 
            print("WARNING: The peak amplitude exceeds 1. The data will be scaled before writing to WAV file.")

        array_data.shape = (1, total_samples)

        # Quantisation with given resolution
        raw_int = [int(sp * pow(2, self.bits-1)) for sp in np.nditer(array_data)]

        fmt = dsp.mkDataStructFMT(self.bits, total_samples)
        
        # Convert integers to bytes
        raw_bytes = struct.pack(fmt, *raw_int)

        return raw_bytes
