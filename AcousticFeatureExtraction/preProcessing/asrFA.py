#!/usr/bin/env python3

sys.path.append(os.path.join(os.path.dirname(sys.path[0], 'ASR')))
import asr


def main(wavPath, haveManualT=False):

    asr.main("../../AudioData/Gated{}".format(wavPath), "../../TextData/{}_asr".format(wavPath))

    bashCommand = "cd ASR; ./run_Penn.sh ../../../AudioData/Gated{} ../../../TextData/{}_asr; cd ..".format(wavPath, wavPath)
    subprocess.run(bashCommand, shell=True)

    if haveManualT == True:
        bashCommand = "cd ASR; ./run_Penn.sh ../../../AudioData/Gated{} ../../../TextData/{}_manual; cd ..".format(wavPath, wavPath)
        subprocess.run(bashCommand, shell=True)

if __name__=="__main__":

    main("ANH", haveManualT=True)