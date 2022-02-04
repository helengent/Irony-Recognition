#!/usr/bin/env python3

import pandas as pd
from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay


dep = pd.read_csv("Inference/predictions_speaker-dependent_PCs_percentChunks_text_3", index_col=0)

for s in list(set(dep.speaker.tolist())):
    subset = dep[dep["speaker"] == s]
    sub_I = subset[subset["label"] == "I"]
    sub_N = subset[subset["label"] == "N"]
    labs = subset.label.tolist()
    preds = subset.prediction.tolist()
    cm = confusion_matrix(labs, preds)

    print("Speaker {}".format(s.upper()))
    print("Ironic:\t\t{} correct\t{} incorrect".format(sub_I.match.tolist().count(True), sub_I.match.tolist().count(False)))
    print("Non-Ironic:\t{} correct\t{} incorrect\n\n".format(sub_N.match.tolist().count(True), sub_N.match.tolist().count(False)))

    if len(list(set(labs))) > 1:
        cm = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["I", "N"])
        cm.plot(cmap=plt.cm.Blues)
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.savefig("Inference/CM_speaker-dependent_{}-ONLY_3.png".format(s))