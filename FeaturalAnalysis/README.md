This directory contains code for running inferential statistics and machine learning experiments on the acoustic features extracted using the code from the AcousticFeatureExtraction directory.

The subdirectories are as follows

1)  ComParE - the ComParE subdirectory contains only the script prepBaselineDF.py, which requires two arguments
    
    i) data_dir = the path to the directory containing individual output csv files from AcousticFeatureExtraction/openSmile.py
    
    ii) out_dir - the path to the directory where the output dataframe should be saved
    prepBaselineDF.py simply compiles the individual files containing the utterance-level values for the ComParE feature set into a single dataframe.
    
2)  InferentialStats - The InferentialStats subdirectory contains code for a number of purposes
    
    i)  Scripts/makeGlobalDatasets.py - 
