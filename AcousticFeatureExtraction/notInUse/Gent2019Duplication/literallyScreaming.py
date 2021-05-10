#!/usr/bin/env python3

import pandas as pd

def main():
    filesToRead = ["../f0comparisonFile/duplicationAll.csv", "../f0comparisonFile/duplicationB.csv", 
                    "../f0comparisonFile/duplicationG.csv", "../f0comparisonFile/duplicationP.csv", 
                    "../f0comparisonFile/duplicationR.csv", "../f0comparisonFile/duplicationY.csv"]
    for f in filesToRead:
        df = pd.read_csv(f)
        nameList = list(set(df.Name))
        for name in nameList:
            if name[0] == "P":
                name = "S" + name
        N = len(nameList)

        names = []
        labs = []
        speaker = []
        time = []
        for name in nameList:
            label = list(set(df.Irony[df.Name==name]))
            s = list(set(df.Speaker[df.Name==name]))

            for n in range(1,76):
                names.append(name)
                labs.append(label[0])
                speaker.append(s[0])
                time.append(n)

        print(len(names), len(labs), len(speaker), len(time))

        out = {"name": names, "speaker": speaker, "lab": labs, "time": time}

        fileName = f[-5]
        if fileName == "l":
            fileName = "A"

        outdf = pd.DataFrame(out)
        outdf.to_csv("../f0comparisonFile/{}.csv".format(fileName))

            



if __name__ == "__main__":
    main()