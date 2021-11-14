This is script to compute pooled EER and min tDCF for ASVspoof2021 PA scenario. 

Dependency:
Python3, Numpy, pandas

Usage:
$: python PATH_TO_SCORE_FILE PATH_TO_GROUDTRUTH_DIR phase
 
 -PATH_TO_SCORE_FILE: path to the score file 
 -PATH_TO_GROUNDTRUTH_DIR: path to the directory that has ASV and CM protocols.
    Please use ./keys
 -phase: either progress, eval, hidden_track_1, or hidden_track_2

Example:
$: python evaluate.py score.txt ./keys eval

