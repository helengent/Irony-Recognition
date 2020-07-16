
ilistB=scan("ilist", "")
N=length(ilistB)

idataB=array(1:(N*75), dim=c(N,75))
for (i in 1:N) {
  idataB[i,] = approx(cleanbyf0(scan(ilistB[i])),n=75)$y
}

nlistB=scan("nlist", "")
N=length(nlistB)
ndataB=array(1:(N*75), dim=c(N,75))
for (i in 1:N) {
  ndataB[i,] = approx(cleanbyf0(scan(nlistB[i])),n=75)$y
}

inlabB=rep(c("i", "n"), each=(N*75))
intimeB=rep(c(1:75), (N*2))
indataB=c(t(idataB)[1:75,],t(ndataB)[1:75,])

mydataB=list(inlabB, intimeB, indataB)
mydataB=as.data.frame(mydataB)
colnames(mydataB) = list("lab", "time", "f0")



m1B=bam(f0 ~ lab + te(time, by=lab) , data=mydata)
summary(m1b)
par(mfrow=c(2,2))
plot(m1y,select=1)
plot(m1y,select=2)
plot_diff(m1y, view = 'time', comp=list(lab = c("i", "n")))




