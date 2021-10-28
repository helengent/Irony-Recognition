#!/usr/bin/env python3

import os
import opensmile
from glob import glob


def main(wav_dir, out_dir):
    if not os.path.isdir(out_dir):
        os.mkdir(out_dir)

    for wav in glob("{}/*.wav".format(wav_dir)):
        basename = os.path.basename(wav).split(".")[0]
        smile = opensmile.Smile(feature_set=opensmile.FeatureSet.ComParE_2016,
                                feature_level=opensmile.FeatureLevel.Functionals
                                )
        y = smile.process_file(wav)
        y.to_csv(os.path.join(out_dir, basename + '.csv'), index = False)

if __name__=="__main__":

    wav_dir = "../AudioData/GatedPruned3"
    out_dir = "../../Data/AcousticData/ComParE/baselinePruned3"

    main(wav_dir, out_dir)