
#Orphan code from a bad idea
def make3Ddf(seqDict, listTup):

    print("Compiling 3D time-series data")

    #takes a tuple of lists, unpacks, and converts into a 3D dataframe

    f0List, mfccs, amsList, plpList = listTup

    fileNameList = seqDict["filename"]
    labList = seqDict["label"]
    speakerList = [filename[0] for filename in fileNameList]

    timeDict = dict()
    innerColNames = ["filename", "label", "speaker", "f0"]

    #Append all variable names to innerColNames
    for i in range(13):
        string = "mfcc" + str(i)
        innerColNames.append(string)
    for i in range(225):
        string = "ams" + str(i)
        innerColNames.append(string)
    for i in range(9):
        string = "plp" + str(i)
        innerColNames.append(string)

    print("Assembling 3D array")

    #Each element in timeDict will have a list of all variables for that element
    #IN THE ORDER THEY APPEAR IN innerColNames!!!
    for i in range(len(f0List[0])):
        timeString = "t" + str(i)

        timeDict[timeString] = [fileNameList, labList, speakerList]

        #All f0 measures at timestep i for all samples
        allTf0 = [f0[i] for f0 in f0List]
        timeDict[timeString].append(allTf0)

        #All mfcc coefficients
        for j in range(13):
            #All mfcc[j] coefficients at timestep i for all samples
            allTmfcc = [mfcc[i][j] for mfcc in mfccs]
            timeDict[timeString].append(allTmfcc)

        #All ams measures
        for j in range(225):
            #All ams[j] measures at timestep i for all samples
            allTams = [ams[j][i] for ams in amsList]
            timeDict[timeString].append(allTams)
        
        #All plp measures
        for j in range(9):
            #All plp[j] measures at timestep i for all samples
            allTplp = [plp[j][i] for plp in plpList]
            timeDict[timeString].append(allTplp)

    with open("../../FeaturalAnalysis/handExtracted/Data/Pruned/Pruned_25ms_timeDict.pkl", "wb") as f:
        pickle.dump(timeDict, f)

    print("3D processes complete")

if __name__=="__main__":

    # This function makes it take FOREVER.
    # Only uncomment if you need the 3D dataframe
    # Arguments below would need to come out midway through extract.pruneAndSave()
    # make3Ddf(newSequentialDict, (f0Pruned, newMFCCsPruned, newAMSpruned, newPLPpruned))

