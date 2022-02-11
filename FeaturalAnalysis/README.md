This directory contains code for running inferential statistics and machine learning experiments on the acoustic features extracted using the code from the AcousticFeatureExtraction directory.

The subdirectories are as follows

1)  ComParE - the ComParE subdirectory contains only the script prepBaselineDF.py, which requires two arguments
    
    i) data_dir = the path to the directory containing individual output csv files from AcousticFeatureExtraction/openSmile.py
    
    ii) out_dir - the path to the directory where the output dataframe should be saved
    
    prepBaselineDF.py simply compiles the individual files containing the utterance-level values for the ComParE feature set into a single dataframe.
    
2)  InferentialStats - The InferentialStats subdirectory contains code for a number of purposes
    
    i)  Scripts/makeGlobalDatasets.py - this script accepts input in the form of the utterance-level acoustic features extraced in the AcousticFeatureExtraction directory, normalizes each measure by the speaker's average value for that measure, and then combines segment specific values into broader categories (e.g. fricatives, stressed vowels, etc.)
    
    ii) Scripts/makeLongData_new.py - this script converts time-series acoustic data into long data appropriate for the application of a Generalized Additive Model (GAM). The GAM itself is applied in R (see the Rcode directory).
    
    iii) Scripts/PCA.py - this script performs an n-factor PCA on a provided input dataframe and outputs information about the amount of variance explained by each PC and what each PC's top five corresponding features are, as well as dataframes with PC values for each sample for use in a Logisitic Mixed Effects Regression Model (see the Rcode directory).
    
3)  modelOptimization - the ModelOptimization subdirectory contains only the script modelOptimization.py, which performs cross-validated hyper-parameter tuning. Due to the priority in my dissertation research being placed on the comparison of different input feature combinations, hyper-parameter tuning was not prioritized and this script was not used.

4)  nnets - the nnets subdirectory contains scripts for training and evaluating neural networks, as well as subdirectories containing evaluation metric results for each model run during experimentation. Some scripts are primarily the result of tinkering during the research process, but the primary scripts are below:

    i)  train.py - this is the primary model training and evaluation script. Documentation and expected inputs can be found in the script's commments
    ii) parseResults.py - this script is used to compile results of many different models and construct usable tables from them.
