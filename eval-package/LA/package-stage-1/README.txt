This is the script to compute pooled EER and min-tDCF for ASVspoof2021 LA scenario. 

Dependency:
Python3, Numpy, pandas

Usage:
1. download and untar the ground-truth label files
$: download.sh
The downloaded directory will be named as ./keys

2. run python script
$: python PATH_TO_SCORE_FILE PATH_TO_GROUDTRUTH_DIR phase
 
 -PATH_TO_SCORE_FILE: path to the score file 
 -PATH_TO_GROUNDTRUTH_DIR: path to the directory from step1
    Please use ./keys
 -phase: either progress, eval, or hidden_track

Example:
$: python evaluate.py score.txt ./keys eval
