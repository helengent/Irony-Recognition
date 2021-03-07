#!/usr/bin/env python3

import os
import extract
import preProcess
import subprocess
import limitsUpperLower

def main(wavPath, speakerList, outputType, winSize=10, prune=True, needReaper=False, needAMS=False, needPLP=False):

    if not os.path.isdir("../../AudioData/Gated{}".format(wavPath)):

        #convert stereo audio to mono
        #downsample to 16000 Hz
        bashCommand = "cd ../../AudioData/{}; mkdir downsampled; ../../AcousticFeatureExtraction/Scripts/monoDown.sh; mkdir ../temp{}; mv downsampled/* ../temp{}; rm -r downsampled; cd ../../AcousticFeatureExtraction/Scripts".format(wavPath, wavPath, wavPath)
        subprocess.run(bashCommand, shell=True)

        #preProcess.py normalizes rms to the average for all files (time-consuming), and trims leading and trailing silences
        preProcess.main(wavPath)

        #Get rid of temp folder. Files now live in Gated{wavPath}
        subprocess.run("rm -r ../../AudioData/temp{}".format(wavPath), shell=True)

    if needReaper == True:
        bashCommand = "mkdir ../ReaperTxtFiles/{}_{}ms_ReaperF0Results; cd ../../AudioData/Gated{}; ../../AcousticFeatureExtraction/Scripts/REAPER/reaper.sh; ../../AcousticFeatureExtraction/Scripts/REAPER/formatReaperOutputs.sh; mv *.p ../../AcousticFeatureExtraction/ReaperTxtFiles/{}_{}ms_ReaperF0Results/; rm *.f0; cd ../../AcousticFeatureExtraction/Scripts".format(wavPath, winSize, wavPath, wavPath, winSize)
        subprocess.run(bashCommand, shell=True)

    #TODO
    if needAMS == True:
        pass

    if needPLP == True:
        pass

    #Calculates speaker-dependent f0 statistics for outlier removal
    limitsUpperLower.main(wavPath, winSize, speakerList)

    #Extract acoustic features

    extract.main(wavPath, speakerList, outputType, winSize=winSize, prune=prune)


if __name__=="__main__":
    wavPath = "Pruned"
    speakerList = ["B", "G", "P", "R", "Y"]
    # outputList = ['global', 'sequential', 'long', 'individual']
    outputList = ['global', 'sequential', 'individual']
    main(wavPath, speakerList, outputList, prune=False, needReaper=True)