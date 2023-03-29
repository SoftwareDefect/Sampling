# Sampling

=====================================================
= Environment Setup
=====================================================
-> Download and Pycharm
->python 3.10.4



=====================================================
= Main script
=====================================================
- exeMain.r
[Overview]
It is the script of the experiment for the cross-validation, time-wise-cross-validation, and across project prediction

[HowToRun]
In Terminal
> cd JIT_HOME/jit/script_r
> Python
> source("exeMain.r")

[Output]
../output/cross-validation/... are the results for cross-validation
../output/cross-validation-timewise/... are the results for time-wise-cross-validation
../output/cross-project/... are the results for across-project prediction


- ReportResults.r
[Overview]
It is the script for presentations (figures and tables)

[HowToRun]
In Terminal
> cd JIT_HOME/jit/script_r
> R
> source("ReportResult.r")
> q()

[Output]
all tables and figures will stored in the "results" folder

=====================================================
= Utility
=====================================================
- utils.r
It is the utility script for computing prediction performance