#install.packages("png")
#install.packages("effsize")
#install.packages("R.matlab")
#install.packages("ScottKnott")
library(R.matlab)
library(png)
library(effsize)
library(ScottKnott)


GetMagnitude <- function(str) {
  lbl <- NULL
  if (str == "large") {
    lbl <- "L"
  } else if (str == "medium") {
    lbl <- "M"
  } else if (str == "small") {
    lbl <- "S"
  } else if (str == "negligible") {
    lbl <- "T"
  } else {
    stop("error")
  }
  return(lbl)
}


plot_hist_single <- function(validation=validation, criteria=criteria,method=method,titlename=titlename) {
  gmnames <- c("Recall","Precision","Pf","F-measure","AUC","MCC","Popt","Recall@20%","Precision@20%","F-measure@20%","PMI","IFA")
  cnames <- c("recall","precision","Pf","F1","AUC","MCC","Popt","Erecall","Eprecision","Efmeasure","PMI", "IFA")
  
  title <- NULL
  data <-  NULL
  
  for (i in seq(projects)) {
    
    project <- projects[i] 
    smodata=NULL
    optdata=NULL
    rdata=NULL
    
    smoout.fname <- sprintf(paste(infpath, "%s-%s-smo.csv", sep="/"), method,project)
    smodata <- read.csv(smoout.fname)
    smodata<- smodata[ ,colnames(smodata) %in% cnames]
    
    optout.fname <- sprintf(paste(infpath, "%s-%s-opt.csv", sep="/"), method,project)
    optdata <- read.csv(optout.fname)
    optdata<- optdata[ ,colnames(optdata) %in% cnames]
    ####取出某一个指标的结果LR-brackets-none
    
    rdata <- optdata-smodata
    data  <- rbind(data, rdata)
  }
  
  
  
  pdf(file=sprintf(paste(outfpath,"hist-opt-%s.pdf",  sep="/"),  method), width=3.5, height=0.78)
  
  par(mfrow=c(1, 1), mai=c(0.05, 0, 0.1, 0), omi=c(0.18, 0.15, 0, 0), mex=0.4, cex=0.5)
  #画布下，左，上，右
  ylab.names <- projects
  #length(projects)+1存储所有数据集的均值颜色
  boxplot(data, xlab=NA, ylab=NA, xaxt="n", yaxt="n",   boxwex=0.6, frame=FALSE, outline=FALSE) ### xlim=c(2, len+12),
  ### tck: length of tick marks记号长度 ### las: vertical or horizontal垂直/水平
  #lwd
  axis(2, mgp=c(0.5, 0.5, 0), las=0, tck=-0.02, cex.axis=0.8, lwd=1.0)
  box(bty="L", lwd=1.5)
  abline(h=0,lty=3,col = c("green"))
  temp<- cnames[1]
  text(x=seq(12), y=par("usr")[3], srt=20, adj=c(1, 1.2), labels=gmnames, xpd=NA,cex=0.7)
  title(main = sprintf("%s",titlename),cex.main =0.6,font=1,line=0.3)
  
  dev.off()
}


infpath <- "D:/software-defect_caiyang/output-opt"  ## using LOC
outfpath <- "D:/software-defect_caiyang/results-opt"


titlenames=c("LR","NB","RF")
methods<- c("LR","NB","RF")

projects <- c("fabric","jgroups","camel","tomcat","brackets","neutron","spring-integration","broadleaf","nova","npm")

#主函数

for (i in seq(methods)) {
  method <- methods[i]
  titlename<- titlenames[i]
  cat(method, "\n")
  plot_hist_single(validation=validation, criteria=criteria,method=method,titlename=titlename)
}


