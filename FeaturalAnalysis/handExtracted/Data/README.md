This directory contains:

  1) csv files produced by Irony-Recognition/AcousticFeatureExtraction/Scripts/extract.py
  2) csv files produced by Irony-Recognition/AcousticFeatureExtraction/Scripts/matlab_lib/AMS/extract_AMS.m
  3) csv files produced by Irony-Recognition/AcousticFeatureExtraction/Scripts/matlab_lib/RASTA-PLP/rastaplp.m
  4) exploratory classifier evaluation metrics produced by csv files produced by Irony-Recognition/FeaturalAnalysis/handExtracted/Scripts/classify.sh
  
The subdirectories "ams", "f0", "globalVector", "mfccs", and "rastaplp" contain individual csv files for each corresponding audio file for a given measure.

Any subdirectory of the format {wavPath}_{winSize}ms (e.g. "Pruned_10ms") will house long data, and data prepared for GAM analysis for all audio files divided by sequential measure.
