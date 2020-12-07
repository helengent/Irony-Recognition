Acoustic Feature Extraction Scripts

These scripts perform the following steps to extract usable dataframes of acoustic measures from raw, utterance-length .wav files:

  1) Preprocessing
  
    i) Convert stereo audio to mono and downsample to 16 kHz (using sox)
    ii) Normalize rms for each file to the average rms for all files
    iii) Remove leading and trailing silences
    
  2) Generate appropriate text/csv files for features extracted using non-python methods
    
    i) f0 (REAPER on command line)
    ii) AMS (matlab)
    iii) RASTA-PLP (matlab)
    
  3) Identify mean f0 and f0 standard deviation for each speaker. Use these values to identify upper and lower limits for f0 to remove statistical outliers.
  
  4) Extract remaining desired acoustic features using python and organize into dataframes
    
    i) Various options for dataframe shapes and formats depending on desired applications
    
Scripts/master.py is intended to streamline all these steps into a single script that can be run once for a given directory of audio files. It is not yet fully complete.

At the moment, master.py can complete the preprocessing step and acoustic feature extraction only if the csv files for AMS and RASTA-PLP have already been generated.
  
In order to run master.py, the following arguments to main() may be modified:

  1) wavPath - this is the directory of the .wav files in Irony-Recognition/AudioData
     
     i) At present, the options for this parameter are "All" or "Pruned" where "Pruned" is a balanced selection of files with equal numbers of ironic and non-ironic samples per speaker
     
     ii) If wavPath is set to "All", it is recommended that the "Prune" parameter be set to True in order to remove statistical outliers based on file length
  
  2) speakerList - the list of unique identifier strings for available speakers
     
     i) At present, the speaker identifiers for the pilot data are ["B", "G", "P", "R", "Y"]
  
  3) outputType - desired output formats. Options:
     
     i) "global" - utterance global measures for all audio files in one dataframe
     
     ii) "sequential" - all sequential measures for all audio files in one dataframe (with padding to account for variant file lengths)
     
     iii) "long" - each sequential measure for all audio files in a long dataframe (this prepares data for GAM analysis - time-consuming)
     
     iv) "individual" - an individual dataframe for each audio file for each sequential measure and for all global measures (for exploratory classifier analysis)
  
  4) prune - boolean. If True, extract.main() will remove outliers based on file length. This is recommended to be set to False if wavPath is "Prune" and True if wavPath is "All".
  
  5) needReaper - boolean. If True, the scripts in the Scripts/REAPER directory will be run. If False, pre-existing REAPER f0 files will be used.
  
The following arguments may not be modified at this time:

  1) winSize - this indicates the size, in miliseconds, of the desired windows for sequential measures. It cannot yet be modified in master.py and must instead be individually modified in:
    
    i) Scripts/REAPER/reaper.sh
    ii) Scripts/matlab_lib/AMS/extract_AMS.m
    iii) Scripts/matlab_lib/RASTA-PLP/rastaplp.m
    
    In the future, I do intend for this to be changeable at the time of running master.py, but for the moment it must be set to 10ms.
  
  2) needAMS - boolean. This must be set to False. In the future, setting it to True should generate AMS csv files in appropriate folder.
  
  3) needPLP - boolean. This must be set to False. In the future, setting it to True should generate RASTA-PLP csv files in appropriate folder.
