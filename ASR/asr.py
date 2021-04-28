#!/usr/bin/env python3

import os
import pandas as pd
from glob import glob
import speech_recognition as sr


def main(in_dir, out_dir):

    os.mkdir(out_dir)

    wit_key = "R5533FE2CVI32BM2LVQRLFMDPMSZ55L3"

    r = sr.Recognizer()
    textDict = {"filename": [], "speaker": [], "label": [], "transcription": []}

    wavList = glob("{}/*.wav".format(in_dir))
    wavList.sort()

    for wav in wavList:
        name = os.path.basename(wav)
        print(name)
        textDict["filename"].append(name)
        textDict["speaker"].append(name.split("_")[1][0])
        if name[-6] == "F":
            textDict["label"].append(name[-6:-4])
        else:
            textDict["label"].append(name[-5])
        with sr.AudioFile(wav) as source:
            w = r.listen(source)
            text = r.recognize_wit(w, wit_key)
            textDict["transcription"].append(text)
            print(text)
    
    textDF = pd.DataFrame(textDict)
    textDF.to_csv("out_dir/asr_transcriptions.csv", index=False)
    
    for i, row in textDF.iterrows():
        print(row)

        with open("out_dir/{}.txt".format(out_dir, row.filename.split(".")[0]), "w") as f:
            f.write(row.transcription)


if __name__=="__main__":

    input_dir = "../../ANH"

    wavList = glob("{}/*.wav".format(input_dir))
    wavList.sort()

    main(input_dir, "data/{}_asr".format(os.path.basename(input_dir)))