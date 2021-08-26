#!/usr/bin/env python3

import os
import pandas as pd
from glob import glob
import speech_recognition as sr


def main(in_dir, out_dir):

    if os.path.isdir(out_dir):
        done = [line.strip("\n") for line in open("{}/done.txt".format(out_dir))]
        bad = [line.strip("\n") for line in open("{}/bad.txt".format(out_dir))]
    else:
        os.mkdir(out_dir)
        done = list()
        bad = list()

    wit_key = "R5533FE2CVI32BM2LVQRLFMDPMSZ55L3"

    # with open("/Users/helengent/Desktop/Irony-Recognition/AcousticFeatureExtraction/preProcessing/ASR/api-key.json") as f:
    #     GOOGLE_CLOUD_SPEECH_CREDENTIALS = f.read()

    r = sr.Recognizer()
    textDict = {"filename": [], "speaker": [], "label": [], "transcription": []}

    wavList = glob("{}/*.wav".format(in_dir))
    wavList.sort() 

    for i, wav in enumerate(wavList):

        if (i + 1) % 100 == 0:
            print("working on file {}/{}".format(i+1, len(wavList)))

        name = os.path.basename(wav)

        if name.split(".")[0] not in done and name.split(".")[0] not in bad:
            with sr.AudioFile(wav) as source:
                w = r.listen(source)

                print(name)

                try:
                    text = r.recognize_wit(w, wit_key)
                    # text = r.recognize_google_cloud(w, credentials_json=GOOGLE_CLOUD_SPEECH_CREDENTIALS)

                    print("File {}/{}\t{}".format(i+1, len(wavList), name.split(".")[0]))
                    print(text)
                    print()

                    with open("{}/{}.txt".format(out_dir, name.split(".")[0]), "w") as f:
                        f.write(text)

                    with open("{}/done.txt".format(out_dir), "a+") as f:
                        f.write(name.split(".")[0] + "\n")
                
                except sr.UnknownValueError:
                    with open("{}/bad.txt".format(out_dir), "a+") as f:
                        f.write(name.split(".")[0] + "\n")


if __name__=="__main__":

    input_dir = "../../../AudioData/ANH"

    wavList = glob("{}/*.wav".format(input_dir))
    wavList.sort()

    main(input_dir, "../../../TextData/{}_asr".format(os.path.basename(input_dir)))