#note: all measure plot
#install.packages("png")
#install.packages("effsize")
#install.packages("R.matlab")
#install.packages("ScottKnott")
library(R.matlab)
library(png)
library(effsize)
library(ScottKnott)

#require(effsize)
#projects <- c("tomcat","camel")

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

#??????????????maybe
GetOverall <- function(data, xnames, base, fname, pth=0.05, mth=c("negligible")) {
  tt.pvs <- wt.pvs <- NULL
  for (i in seq(xnames)) {
    treat <- data[, xnames[i]]
    tt.pvs <- c(tt.pvs, t.test(treat, base, paired=TRUE)$"p.value")
    apvs <- wilcox.test(treat, base, paired=TRUE)$"p.value"
    if (is.na(apvs)) { apvs <- 1 }
    wt.pvs <- c(wt.pvs, apvs)
  }
  
  wt.pvs <- p.adjust(wt.pvs, method="BH")
  tt.pvs <- p.adjust(tt.pvs, method="BH")
  
  dt.out <- data.frame(matrix(0, nrow=length(xnames), ncol=12))
  rownames(dt.out) <- xnames
  colnames(dt.out) <- c("avg", "improve.avg", "cohend", "cohend.magnitude", "tt.p-value", "med", "improve.med", "cliffd", "cliffd.magnitude", "wt.p-value", "overall", "colsall")
  
  base.out <- c(mean(base), 0, NA, NA, NA, median(base), NA, NA, NA, NA, NA, NA)
  dt.out[, "tt.p-value"] <- tt.pvs
  dt.out[, "wt.p-value"] <- wt.pvs
  dt.out[, "med"] <- apply(data[, xnames], MARGIN=2, FUN=median)
  dt.out[, "avg"] <- apply(data[, xnames], MARGIN=2, FUN=mean)
  
  colsall <- overall <- NULL
  cohend <- cliffd <- NULL
  cohend.magnitudes <- cliffd.magnitudes <- NULL
  for (i in seq(xnames)) {
    d2 <- data[, xnames[i]]
    
    cd <- effsize::cohen.d(d2, base, paired=TRUE)
    cohend <- c(cohend, cd$estimate)
    cohend.magnitudes <- c(cohend.magnitudes, cd$magnitude)
    
    cd <- effsize::cliff.delta(d2, base)
    cliffd <- c(cliffd, cd$estimate)
    cliffd.magnitudes <- c(cliffd.magnitudes, cd$magnitude)
    
    if (wt.pvs[i]<=pth) {
      if (median(d2, na.rm=TRUE) > median(base, na.rm=TRUE)) {
        if (cd$magnitude %in% mth) {
          overall <- c(overall, "==")
          colsall <- c(colsall, "black")
        } else {
          overall <- c(overall, "vv")
          colsall <- c(colsall, "blue")
        }
      } else {
        if (cd$magnitude %in% mth) {
          overall <- c(overall, "==")
          colsall <- c(colsall, "black")
        } else {
          overall <- c(overall, "xx")
          colsall <- c(colsall, "red")
        }
      }
    } else {
      overall <- c(overall, "==")
      colsall <- c(colsall, "black")
    }
  }
  
  return(data.frame(overall=overall, colsall=colsall, stringsAsFactors=FALSE))
}



plot_hist_single <- function(validation=validation, criteria=criteria,method=method,titlename=titlename) {
  
  mnames<- c("none","rum","nm","enn","tlr","rom","oss","smo","bsmote","csmote","cenn")
  cnames<- paste(mnames, criteria, sep=".") 
  
  gmnames=c("NONE","RUM","NearMiss","ENN","TomekLink","ROM","OSS","SMOTE","BSMOTE","SMOTE+Tomek","SMOTE+ENN")
  title <- NULL
  n.row <- length(projects)
  a.names <- projects
  
  data <- data.all <- NULL
  data.out.r <- matrix(0, nrow=n.row+3, ncol=length(cnames))
  data.out.r <- as.data.frame(data.out.r)
  rownames(data.out.r) <- c(a.names, "AVG", "WTL", "Improve.e") 
  colnames(data.out.r) <- cnames  
  for (i in seq(projects)) {
    rdata=NULL
    project <- projects[i]
    for (j in seq(mnames)){
      mname <- mnames[j]
      out.fname <- sprintf(paste(infpath, "%s-%s-%s.csv", sep="/"), method,project,mname)
      idata <- read.csv(out.fname)
      idata<-idata[,criteria]
      rdata=cbind(rdata, idata )
      colnames(rdata)[j] <-paste(mname, criteria, sep=".")
    }
    data.all[[i]] <- rdata
    data  <- rbind(data, rdata)
  }
  overall <- data.frame(matrix(0, nrow=length(cnames), ncol=length(projects)))
  colnames(overall) <- projects
  overall <- apply(overall, c(1, 2), as.character)
  colsall <- overall
  for (i in seq(projects)) {
    project <- projects[i]
    ares <- GetOverall(data=data.all[[i]], xnames=cnames, base=data.all[[i]][, 1], fname=sprintf("results/%s-all-out-%s.csv", project, criteria))
    overall[, i] <- ares$overall
    colsall[, i] <- ares$colsall
  }
  
  zhongzhi<- apply(data[, cnames, drop=FALSE], 2, median)
  ares <- GetOverall(data=data, xnames=cnames, base=data[, 1], fname=sprintf("results/(%s)all-out-%s.csv", method, criteria))
  overall <- cbind(overall, ares$overall)
  colsall <- cbind(colsall, ares$colsall)
  pdf(file=sprintf(paste(outfpath,method,"hist-%s-%s.pdf",  sep="/"),  method,criteria), width=3.5, height=0.58)
  
  par(mfrow=c(1, 1), mai=c(0, 0, 0.1, 0), omi=c(0.18, 0.15, 0, 0), mex=0.4, cex=0.5)
  ylab.names <- projects
  cols <- colsall[, length(projects)+1]
  boxplot(data[, cnames], xlab=NA, ylab=NA, xaxt="n", yaxt="n",  border=cols, boxwex=0.6, frame=FALSE, outline=FALSE) ### xlim=c(2, len+12),
  axis(2, mgp=c(0.5, 0.5, 0), las=0, tck=-0.02, cex.axis=0.8, lwd=1.0)
  box(bty="L", lwd=1.5)
  temp<- cnames[1]
  abline(h=zhongzhi[[1]],lty=3)
  text(x=seq(11), y=par("usr")[3], srt=20, adj=c(1, 1.2), labels=gmnames, xpd=NA,cex=0.5)
  title(main = sprintf("%s _ %s",method,titlename),cex.main =0.6,font=1,line=0.3)
  
  dev.off()
}


infpath <- "D:/software-defect_caiyang/output"  ## using LOC
outfpath <- "D:/software-defect_caiyang/results"


titlenames=c("Popt","Recall@20%","Precision@20%","F-measure@20%","Recall","Precision","F-measure","AUC","MCC")
projects <- c("fabric","jgroups","camel","tomcat","brackets","neutron","spring-integration","broadleaf","nova","npm")

methods<- c("RF","LR","NB")

for (method in methods){
  criterias <- c("Popt","Erecall","Eprecision","Efmeasure","recall","precision","F1","AUC","MCC")
  for (i in seq(criterias)) {
    criteria <- criterias[i]
    titlename<- titlenames[i]
    cat(method, criteria, "\n")
    plot_hist_single(validation=validation, criteria=criteria,method=method,titlename=titlename)
  }
}


