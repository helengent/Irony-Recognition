This directory will hold code for building the eventual neural network for irony recognition. At present, however: 

  1) classify.sh (courtesy of Chase Adams) evaluates a variety of ML classifiers on a single acoustic measure each. 
  2) makeLongData.py takes (already long) data as outputted from Irony-Recognition/AcousticFeatureExtraction/Scripts/extract.py and selects 75 evenly-spaced time points per audio file. It is admittedly extremely inefficient and takes a ridiculously long time to run.
