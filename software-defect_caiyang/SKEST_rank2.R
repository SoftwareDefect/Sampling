#note:pf,PMI,IFA,change 
#install.packages("ScottKnott")
library("ScottKnottESD")
library("reshape2")
library("car")
library("effsize")
library(R.matlab)
library("ggplot2")
library("RColorBrewer")
library(ggplot2)
library(png)
infpath <- "D:/software-defect_caiyang/output"  ## using LOC

outfpath <- "D:/software-defect_caiyang/output-rank"


#### plot methods #####
methods<- c("RF","LR","NB")
projects <- c("fabric","jgroups","camel","tomcat","brackets","neutron","spring-integration","broadleaf","nova","npm")

mnames<- c("none","rum","nm","enn","tlr","rom","oss","smo","bsmote","csmote","cenn")
gmnames<-c("NONE","RUM","NearMiss","ENN","Tomek Link","ROM","OSS","SMOTE","BSMOTE","SMOTE+Tomek","SMOTE+ENN")

criterias <- c("PMI","IFA","Pf")
titlenames=c("PCI@20%","IFA","Pf")
# criterias <- c("IFA")
# titlenames=c("IFA")

lbnames<-NULL

printResult <- function(sk1st, path, method) {
  out.fname <- sprintf(paste(outfpath,"rank-%s230315.csv",  sep="/"),method)
  write.table(sk1st,out.fname,row.names=TRUE,col.names=TRUE,sep=",")
}

for (method in methods) {
  data.out.r <- matrix(0, nrow=length(criterias), ncol=length(gmnames))
  data.out.r < - as.data.frame(data.out.r)
  rownames(data.out.r) <- criterias 
  colnames(data.out.r) <- gmnames 
  cat(method,"\n")
  for (i in seq(criterias)) {
    criteria = criterias[i]
    titlename = titlenames[i]
    cat(method, criteria, "\n")
    data <-  NULL
    sk1st <- NULL
    skp<- NULL
    cnames<- paste(mnames, criteria, sep=".") 
    for (i in seq(projects)) {
      project <- projects[i]
      idata <-  NULL
      rdata <-  NULL
      for (j in seq(mnames)){
        mname <- mnames[j]
        out.fname <- sprintf(paste(infpath, "%s-%s-%s.csv", sep="/"),method,project,mname)
        idata <- read.csv(out.fname)
        ####取出某一个指标的结果
        idata<-idata[,criteria]
        rdata=cbind(rdata, idata )
        colnames(rdata)[j] <-mname
        #rdata <-as.matrix(rdata)
        ##上面三行是核心代码
      }
      #data  <- rbind(data, rdata)
      rdata <- as.data.frame(rdata)
      rdata <- -rdata
      sk <- sk_esd(rdata)#进行一个数据集的sk检???
      sk1st <- rbind(sk1st, sk$groups[mnames])  
    }
    sk1st <- as.data.frame(sk1st)
    sk <- sk_esd(sk1st)#sk检验结果作为输入进行sk检??? 
    skp<- as.matrix(sk$groups[mnames])
    data.out.r[criteria, ] <- skp
  }
  printResult(data.out.r, outfpath, method)
}


