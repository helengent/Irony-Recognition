The .Rmd files in this folder are meant to be executed in the folder containing the data. That being Irony-Recognition/FeaturalAnalysis/handExtracted/Data/Pruned_10ms/

This requirement is already accounted for in the R code. The first command in each file is setwd(choose.dir())

Depending on the processing power of your computer, these scripts may take quite a while to complete - particularly amsGamms.Rmd - due to the size of the models.
