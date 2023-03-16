#note:all measure SKEST
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

outfpath <- "D:/software-defect_caiyang/results-SKEST20230315"


#### plot methods #####
methods<- c("RF","LR","NB")


projects <- c("fabric","jgroups","camel","tomcat","brackets","neutron","spring-integration","broadleaf","nova","npm")

criterias <- c("Popt","Erecall","Eprecision","Efmeasure","recall","precision","F1","AUC","MCC")
titlenames=c("Popt","Recall@20%","Precision@20%","F-measure@20%","Recall","Precision","F-measure","AUC","MCC")
# criterias <- c("Popt")
# titlenames=c("Popt")



mnames<- c("none","rum","nm","enn","tlr","rom","oss","smo","bsmote","csmote","cenn")
gmnames<-c("NONE","RUM","NearMiss","ENN","Tomek Link","ROM","OSS","SMOTE","BSMOTE","SMOTE+Tomek","SMOTE+ENN")

lbnames<-NULL
for (method in methods) {
  for (i in seq(criterias)) {
    criteria = criterias[i]
    titlename = titlenames[i]
    cat(method, criteria, "\n")
    data <-  NULL
    sk1st <- NULL
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
        colnames(rdata)[j] <-paste(mname, criteria, sep=".")
        #rdata <-as.matrix(rdata)
        ##上面三行是核心代码
      }
      #data  <- rbind(data, rdata)
      rdata <- as.data.frame(rdata)
      sk <- sk_esd(rdata)#进行一个数据集的sk检???
      sk1st <- rbind(sk1st, sk$groups[cnames])  
    }
    sk1st <- as.data.frame(sk1st)
    sk <- sk_esd(sk1st)#sk检验结果作为输入进行sk检??? 
    
    rownames(sk$m.inf) <-cnames[sk$ord]
    ## let former subscript empty 
    rownames(sk$m.inf) <-lbnames[sk$ord]#原标题置空
    #png(file=sprintf(paste(outfpath,method,"SKESD_%s_%s.pdf",  sep="/"), method,criteria), width=6.5, height=2.5, units="in",res=600)
    pdf(file=sprintf(paste(outfpath,method,"SKESD_%s_%s.pdf",  sep="/"), method,criteria), width=3.5, height=0.5)
    
    
    col1<- rainbow(8)
    col2 <- brewer.pal(8,"Dark2")
    col3 <- brewer.pal(9,"Greys")
    col4 <- brewer.pal(4,"RdBu")
    mycolor <- c(col3[9:9],col1[1:7],col2[1:8],col4[1:4])
    
    par(mfrow=c(1, 1), mai=c(0.15,0.13,0.06, 0), mex=0.1, cex=0.4)
    #mex表示刻度线延申长度，cex表示字体，mai页面边距，下，左，上，右边
    plot(sk,rl=FALSE, xlab=NA, ylab=NA, id.col=FALSE, yaxt="n",main="", title="", mex=0.01, cex=0.1, col=mycolor)###, col=rainbow(max(sk.1st$groups)), xlim=c(2, length(lbls)),
    #cex.axis=0.7表示坐标轴字体大小,坐标轴刻度文字的缩放倍数
    text(x=seq(11),y=0.1,srt =20, adj = c(1.0,1.0), labels = gmnames[sk$ord],xpd = TRUE,cex=0.6,col="black")
    #adj，水平偏移和竖直偏移+左下，-右上，x,y分别表示每个横坐标文字的位置坐标，y的间隔选择-0.1，0.8
    title(main = sprintf("%s _ %s",method,titlename),cex.main =0.8,font=1,col="black",line=0.2)
    #srt表示倾斜角度
    dev.off()
  }
}


