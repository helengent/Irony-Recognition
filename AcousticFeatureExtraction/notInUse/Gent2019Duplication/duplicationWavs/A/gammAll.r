cleanbyf0 = function(x) {
  n=length(x)
  
  #exclude f0 samples above or below a threshold 
  for (i in 1:n) {
    if(is.na(x[i]) == FALSE && (x[i] < 50 || x[i] > 300)) {x[i] = NA}
  }
  
  #exclude leading and trailing NA 
  while(is.na(x[1])==TRUE) {
    x=x[-1]
  }
  while(is.na(x[n])==TRUE) {
    x=x[-n]
    n=length(x)
  }
  return(x)
}

cleangf0 = function(x) {
  n=length(x)
  
  #exclude f0 samples above or below a threshold
  for (i in 1:n) {
    if(is.na(x[i]) == FALSE && (x[i] < 100 || x[i] > 470)) {x[i] = NA}
  }               
  
  #exclude leading and trailing NA
  while(is.na(x[1])==TRUE) {
    x=x[-1]
  }
  while(is.na(x[n])==TRUE) {
    x=x[-n]
    n=length(x)
  }
  return(x)
} 

cleanprf0 = function(x) {
  n=length(x)
  
  #exclude f0 samples above or below a threshold
  for (i in 1:n) {
    if(is.na(x[i]) == FALSE && (x[i] < 70 || x[i] > 450)) {x[i] = NA}
  }               
  
  #exclude leading and trailing NA
  while(is.na(x[1])==TRUE) {
    x=x[-1]
  }
  while(is.na(x[n])==TRUE) {
    x=x[-n]
    n=length(x)
  }
  return(x)
} 

library(stringr)

ilistAll=scan("ilist", "")
N=length(ilistAll)
S=rep(str_replace(str_extract(ilistAll, "_[a-z]"), "_", ""), each = 75)

idataAll=array(1:(N*75), dim=c(N,75))
for (i in 1:N) {
  if (S[i]=="b"){
    idataAll[i,] = approx(cleanbyf0(scan(ilistAll[i])),n=75)$y}
  else if (S[i]=="y"){
    idataAll[i,] = approx(cleanbyf0(scan(ilistAll[i])),n=75)$y}
  else if (S[i]=="g"){
    idataAll[i,] = approx(cleangf0(scan(ilistAll[i])),n=75)$y}
  else if (S[i]=="p"){
    idataAll[i,] = approx(cleanprf0(scan(ilistAll[i])),n=75)$y}
  else if (S[i]=="r"){
    idataAll[i,] = approx(cleanprf0(scan(ilistAll[i])),n=75)$y}
}

nlistAll=scan("nlist", "")
N=length(nlistAll)
ndata=array(1:(N*75), dim=c(N,75))
for (i in 1:N) {
  if (S[i]=="b"){
    ndataAll[i,] = approx(cleanbyf0(scan(nlistAll[i])),n=75)$y}
  else if (S[i]=="y"){
    ndataAll[i,] = approx(cleanbyf0(scan(nlistAll[i])),n=75)$y}
  else if (S[i]=="g"){
    ndataAll[i,] = approx(cleangf0(scan(nlistAll[i])),n=75)$y}
  else if (S[i]=="p"){
    ndataAll[i,] = approx(cleanprf0(scan(nlistAll[i])),n=75)$y}
  else if (S[i]=="r"){
    ndataAll[i,] = approx(cleanprf0(scan(nlistAll[i])),n=75)$y}
}

inlabAll=rep(c("i", "n"), each=(N*75))
intimeAll=rep(c(1:75), (N*2))
indataAll=c(t(idataAll)[1:75,],t(ndataAll)[1:75,])

mydataAll=list(S, inlabAll, intimeAll, indataAll)
mydataAll=as.data.frame(mydataAll)
colnames(mydataAll) = list("speaker", "lab", "time", "f0")

library(mgcv)
library(devtools)
library(itsadug)
par(mfrow=c(2,2))
m1all=bam(f0 ~ lab + te(time, by=lab) + s(speaker, bs="re") + s(time, speaker, bs="fs", m=1), data=mydata)
summary(m1all)
plot(m1all, select=1, main = "All Speakers: Ironic", shade=TRUE, shade.col = "seagreen2")
plot(m1all, select=2, main = "All Speakers: Non-Ironic", shade=TRUE, shade.col = "wheat")
plot(m1all, select=1, ylab = "te(time):lab", xlab = "time", main = "All Speakers: Both Curves"); par(new=TRUE); plot(m1all, select=2, ylab = "", xlab = "", main = "", col="indianred4")
plot_diff(m1all, view = 'time', shade=TRUE, comp=list(lab = c("i", "n")))




