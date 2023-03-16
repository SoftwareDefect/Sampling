
#note:pf,PMI,IFA,exchange red and blue 
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
    
    cd <- effsize::cohen.d(d2, base, paired=TRUE)#Cohen's d????95%缃俊鍖洪棿璁＄???
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
          colsall <- c(colsall, "red")
        }
      } else {
        if (cd$magnitude %in% mth) {
          overall <- c(overall, "==")
          colsall <- c(colsall, "black")
        } else {
          overall <- c(overall, "xx")
          colsall <- c(colsall, "blue")
        }
      }
    } else {
      overall <- c(overall, "==")
      colsall <- c(colsall, "black")
    }
  }
  
  return(data.frame(overall=overall, colsall=colsall, stringsAsFactors=FALSE))
  #返回值只有颜色和差异度（√，???，-）
  #之前的那些验证也是为了求出来这两个值
}



plot_hist_single <- function(validation=validation, criteria=criteria,method=method,titlename=titlename) {
  
  mnames<- c("none","rum","nm","enn","tlr","rom","oss","smo","bsmote","csmote","cenn")
  cnames<- paste(mnames, criteria, sep=".") 
  
 #gmnames=c("None","RUM","NearMiss","Edited Nearest Neighbor","Tomek Link Removal","ROM","Condense Nearest Neighbor","SMOTE","Borderline-SMOTE","SMOTE+Tomek Link Removal","SMOTE+ENN")
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
      ####取出某一个指标的结果
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
    # s.index <- which.max(apply(data.a ll[[i]][, smnames, drop=FALSE], 2, median))
    #这一行是为了找出来最优的监督学习的方法，对应文论中的箱图的基线
    #这里把PL作为对比方法，所以把这一行省略掉，base直接等于data.all[[i]][, 22]，PL那一列
    ares <- GetOverall(data=data.all[[i]], xnames=cnames, base=data.all[[i]][, 1], fname=sprintf("results/%s-all-out-%s.csv", project, criteria))
    overall[, i] <- ares$overall
    colsall[, i] <- ares$colsall
  }
  
  zhongzhi<- apply(data[, cnames, drop=FALSE], 2, median)
  #s.index <- which.max(zhongzhi)
  #abline(h=zhongzhi[s.index][[1]],lty=3)取最好的中值做虚线，这个要放在下边，图画出来之后
  #取最好的作为对比，做出来那条 虚线
  #取某一行，就直接
  #取最后一个的中值
  #上边三行，取出最大的中值
  ares <- GetOverall(data=data, xnames=cnames, base=data[, 1], fname=sprintf("results/(%s)all-out-%s.csv", method, criteria))
  overall <- cbind(overall, ares$overall)
  colsall <- cbind(colsall, ares$colsall)
  ###overall前六列是每一个数据集的√，×，—值，第七列是data（所有数据里按列合并）的那些值
  ###colsall前六列是每一个数据集的颜色，第七列是data（所有数据里按列合并）颜色值
  ### write.table(overall, file=sprintf("results/(%s)all-%s.csv", validation, criteria), row.names=TRUE,  col.names=TRUE,  append=FALSE, sep=",")
  # len <- length(smnames)+1
  # pos <- c(1:len, (len+2):(len+14))
  # 
  #png(file=sprintf(paste(outfpath,method,"hist-%s-%s.png",  sep="/"),  method,criteria), width=3, height=1.0, units="in", res=600, pointsize=10)
  pdf(file=sprintf(paste(outfpath,method,"hist-%s-%s.pdf",  sep="/"),  method,criteria), width=3.5, height=0.58)
  
  par(mfrow=c(1, 1), mai=c(0, 0, 0.1, 0), omi=c(0.18, 0.15, 0, 0), mex=0.4, cex=0.5)
  #画布，下，左，上，右边
  ylab.names <- projects
  cols <- colsall[, length(projects)+1]
  #length(projects)+1存储所有数据集的均值颜色
  boxplot(data[, cnames], xlab=NA, ylab=NA, xaxt="n", yaxt="n",  border=cols, boxwex=0.6, frame=FALSE, outline=FALSE) ### xlim=c(2, len+12),
  ### tck: length of tick marks记号长度 ### las: vertical or horizontal垂直/水平
  #lwd
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


#titlenames=c("EA_Recall","EA_Precision","EA_Fmeasure","PMI","IFA","Popt","EA_Recall","EA_Precision","EA_Fmeasure","PMI","IFA","Popt","Recall","Precision","Accuracy","Fmeasure","Pf","G1","AUC","MCC")
titlenames=c("PCI@20%","IFA","Pf")
methods<- c("RF","LR","NB")
projects <- c("fabric","jgroups","camel","tomcat","brackets","neutron","spring-integration","broadleaf","nova","npm")


#主函数
for (method in methods){
  criterias <- c("PMI","IFA","Pf")
  for (i in seq(criterias)) {
    criteria <- criterias[i]
    titlename<- titlenames[i]
    cat(method, criteria, "\n")
    plot_hist_single(validation=validation, criteria=criteria,method=method,titlename=titlename)
  }
}

