This directory contains the utterance-level, irony annotated WAV files that comprise the Sad Boyz corpus. Each filename contains information about the podcast episode it is from, the speaker of the utterance, its relative position in the episode, and its irony label. This is illustrated with the following example:

      SBep23_o1117-N.wav
      
      "SBep23" --> This sample is from the 23rd episode of the Sad Boyz podcast
      "o"      --> This sample was spoken by speaker O
      "1117"   --> This is the 1,117th utterance labeled in this episode (over the course of the duration of the episode)
      "N"      --> This sample has been labeled as non-ironic

The All3 directory contains two subdirectories:
  1)  researchFiles
  2)  newTest

These subdirectories are not class balanced and contain some outliers in terms of duration.

The Pruned3 and newTest directories have had outliers (exceptionally long files) removed. Pruned3 has had extra non-ironic samples removed to achieve class balance.

The GatedPruned3 and GatednewTest directories have undergone the following preprocessing steps:
  1)  downsampling to 16kHz
  2)  normalization of all stereo files to single channel
  3)  RMS normalization
  4)  trimming of leading and trailing silences
