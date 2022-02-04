Before running acoustic feature extraction scripts, there are a few other things to be aware of.

  1) These scripts rely on the existence of a directory called Data with subdirectories AcousticData and TextData. The Data directory should be located within the same parent directory as Irony-Recognition.
  2) reconcile.py, located in the preProcessing folder, is the script used to reconcile responses from two annotators for the same set of wave files and sort them into matches, mismatches, and odd files. The inputs and outputs from this scripts should also be separate from the Irony-Recognition directory.
  3) After running reconcile.py, move the matched files to Irony-Recognition/AudioData/All3. If the new data is a new data set, create a new subdirectory to house it. If it is meant to be included with another data set, place it in the appropriate subdirectory.
  4) pruneByDur.py, located in the preProcessing folder, is used to remove outliers by duration and balance the representation of ironic and non-ironic samples. If only the removal of outliers is desired (leaving the data unbalanced), comment out lines 65-114.

Acoustic Feature Extraction Scripts

These scripts perform the following steps to extract usable dataframes of acoustic measures from raw, utterance-length .wav files:

  1) Preprocessing
  
    i) Convert stereo audio to mono and downsample to 16 kHz (using sox)
    ii) Normalize rms for each file to the average rms for all files
    iii) Remove leading and trailing silences
    
  2) Generate appropriate text/csv files for features extracted using non-python methods
    
    iii) AMS (matlab)
    iv) RASTA-PLP (matlab)
    
  3) Identify mean F0 and F0 standard deviation for each speaker. Use these values to identify upper and lower limits for F0 to remove statistical outliers.
  
  4) Identify mean durations for words, vowels, consonants, pauses, and each ARPABET symbol used in the Penn Forced Aligner
  
  5) Extract remaining desired acoustic features using python and save to small, individual output files
        
  6) Perform ASR and forced alignment
  
  7) Extract text-dependent acoustic features
  
  8) Identify speaker averages for each segmental for each measure taken for that segmental in text dependent feature extraction.
    
Scripts/master.py is intended to streamline all these steps into a single script that can be run once for a given directory of audio files. It is not yet fully complete.

At the moment, master.py can complete the preprocessing step and acoustic feature extraction only if the csv files for AMS and RASTA-PLP have already been generated.
  
master.py has two functions, preProcess() and extractFeats(). preProcess performs steps 1, 6, 3, and 4 above and should be run first. preProcess() takes the following arguments:
  1) wavPath - this is the directory of the .wav files in Irony-Recognition/AudioData
  
  2) speakerList - the list of unique identifier strings for available speakers
  
  3) winSize (default=10) - At the moment this must be set to 10 unless the matlab scripts to extract AMS and RASTA-PLP are also manually modified for this change
  
  4) needAMS (default=False) - This is nonfunctional at this time. Aspirationally, it would be nice for the matlab scripts to be called through master.py
  
  5) needPLP (default=False) - This is nonfunctional at this time. Aspirationally, it would be nice for the matlab scripts to be called through master.py
  
  6) haveManualT (default=False) - If set to True, the files considered when calculating speaker duration averages will come from manual transcription files, rather than ASR output.
  
After preProcess has been run, AMS and RASTA-PLP should be extracted using the scripts in matlab_lib in matlab. csv outputs from these scripts should be saved to Data/AcousticData/ams_untransposed and Data/AcousticData/rastaplp_untransposed.

extractFeats() performs steps 5, 7 and 8 above and should be run last. extractFeats() takes the following arguments:
  1) wavPath - this is the directory of the .wav files in Irony-Recognition/AudioData
  
  2) speakerList - the list of unique identifier strings for available speakers
  
  3) outputType - ("individual", "global") determines whether outputs for utterance-global measures are saved to individual csv files or to one large csv file.
  
  4) winSize (default=10) - At the moment this must be set to 10 unless the matlab scripts to extract AMS and RASTA-PLP are also manually modified for this change
  
  5) tg_mod (default="asr", other option "manual") - identifies if working with asr or manually transcribed text data
  
  6) saveWhole (default="False") - If set to True, saves a unified csv file with all text dependent and text independent utterance-global acoustic features.
