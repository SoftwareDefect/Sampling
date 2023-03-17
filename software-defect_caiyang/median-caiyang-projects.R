#install.packages("png")
#install.packages("effsize")
#install.packages("R.matlab")
#install.packages("ScottKnott")
# install.packages("openxlsx")
library(openxlsx)
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

mnames<- c("none","rum","nm","enn","tlr","rom","oss","smo","bsmote","csmote","cenn")
gmnames<-c("NONE","RUM","NearMiss","ENN","Tomek Link","ROM","OSS","SMOTE","BSMOTE","SMOTE+Tomek","SMOTE+ENN")

titlenames <- c("Recall","Precision","Pf","F-measure","AUC","MCC","Popt","Recall@20%","Precision@20%","F-measure@20%","PMI","IFA")
criterias <- c("recall","precision","Pf","F1","AUC","MCC","Popt","Erecall","Eprecision","Efmeasure","PMI", "IFA")

infpath <- "D:/Git/Sampling/software-defect_caiyang/output"  ## using LOC
outfpath <- "D:/Git/Sampling/software-defect_caiyang/results-median-projects"

projects <- c("fabric","jgroups","camel","tomcat","brackets","neutron","spring-integration","broadleaf","nova","npm")

methods<- c("LR","NB","RF")


#主函数
for (method in methods){
  cat(method,"\n")
  data=NULL
  for (i in seq(projects)) {
    project <- projects[i]
    data.out.r <- matrix(0, nrow=length(criterias), ncol=length(mnames))
    data.out.r <- as.data.frame(data.out.r)
    rownames(data.out.r) <- criterias
    colnames(data.out.r) <- gmnames  
    
    for (j in seq(criterias)) {
      criteria <- criterias[j]
      rdata=NULL
      for (k in seq(mnames)){
        mname <- mnames[k]
        out.fname <- sprintf(paste(infpath, "%s-%s-%s.csv", sep="/"), method,project,mname)
        idata <- read.csv(out.fname)
        ####取出某一个指标的结果
        idata<-idata[,criteria]
        rdata=cbind(rdata, idata )
        colnames(rdata)[k] <-mname
      }   
      data.out.r[criteria, ] <- apply(rdata[, mnames], MARGIN=2, FUN=median, na.rm=FALSE)
    }
    rownames(data.out.r) <- titlenames
    data=rbind(data,data.out.r)
  }
  
  sheets = list("fabric" = data.all[[1]],"jgroups" =  data.all[[2]],"camel" =  data.all[[3]],"tomcat" =  data.all[[4]],"brackets" =  data.all[[5]],"neutron" =  data.all[[6]],"spring-integration" =  data.all[[7]],"broadleaf" =  data.all[[8]],"nova" =  data.all[[9]],"npm" =  data.all[[10]])
  fname <-  sprintf(paste(outfpath,"median-%s.xlsx",  sep="/"),  method)
  write.xlsx(sheets,fname,row.names = TRUE)
}



