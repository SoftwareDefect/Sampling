# Sampling


Environment Setup ：

> Download Pycharm
>python 3.10.4



Main :

>mainLR.py
[Overview]
It is a comparative experiment of sampling methods using Logistic Regression as a classifier.
The sampling method includes Random Under-Sampling, NearMiss, ENN, TomekLink, OSS, Random Over-Sampling, SMOTE, BSMOTE, SMOTE+Tomek Link, SMOTE+ENN 
>mainNB.py
[Overview]
It is a comparative experiment of sampling methods using Naive Bayes as a classifier.
The sampling method includes Random Under-Sampling, NearMiss, ENN, TomekLink, OSS, Random Over-Sampling, SMOTE, BSMOTE, SMOTE+Tomek Link, SMOTE+ENN 
>mainRF.py
[Overview]
It is a comparative experiment of sampling methods using Random Forest as a classifier.
The sampling method includes Random Under-Sampling, NearMiss, ENN, TomekLink, OSS, Random Over-Sampling, SMOTE, BSMOTE, SMOTE+Tomek Link, SMOTE+ENN 

[HowToRun]
In Terminal
> python mainLR.py
> python mainNB.py
> python mainRF.py

[Output]
../output/... are the measures of 50 times of sampling technique for each data set
../results-median/... are the median of the measures



