#!/usr/bin/env python3

import extract
import preProcess
import subprocess
import limitsUpperLower

def main(wavPath, speakerList, winSize=10, prune=True, needReaper=False, needAMS=False, needPLP=False):

    #convert stereo audio to mono
    #downsample to 16000 Hz
    bashCommand = "cd ../../AudioData/{}; mkdir downsampled; ../../AcousticFeatureExtraction/Scripts/monoDown.sh; mkdir ../temp{}; mv downsampled/* ../temp{}; rm -r downsampled; cd ../../AcousticFeatureExtraction/Scripts".format(wavPath, wavPath, wavPath)
    subprocess.run(bashCommand, shell=True)

    #preProcess.py normalizes rms to the average for all files (time-consuming), and trims leading and trailing silences
    preProcess.main(wavPath)

    #Get rid of temp folder. Files now live in Gated{wavPath}
    subprocess.run("rm -r ../../AudioData/temp{}".format(wavPath), shell=True)

    #TODO
    if needReaper == True:
        pass

    if needAMS == True:
        pass

    if needPLP == True:
        pass

    #Calculates speaker-dependent f0 statistics for outlier removal
    limitsUpperLower.main(wavPath, winSize, speakerList)

    #Extract acoustic features
    #TODO streamline output format options
    extract.main(wavPath, winSize=winSize, prune=prune)


if __name__=="__main__":
    speakerList = ["B", "G", "P", "R", "Y"]
    main("Pruned", speakerList, prune=False)