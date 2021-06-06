# LFCC-GMM ASVspoof 2021 baseline

By Massimiliano Todisco, EURECOM, 2021

------

Matlab implementation of spoofing detection baseline system based on:
- front-ends:
    - high-frequency resolution linear frequency cepstral coefficients (LFCCs)
- back-end:
    - Gaussian Mixture Models (GMMs)

## Contents of the package

### LFCC
Matlab implementation of linear frequency cepstral coefficients.

For further details on LFCC for antispoofing, refer to the following publication:

- H. Tak, J. Patino, A. Nautsch, N. Evans, M. Todisco, "Spoofing Attack Detection using the Non-linear Fusion of Sub-band Classifiers" in Proc INTERSPEECH, 2020.
- M. Sahidullah, T. Kinnunen, C. Hanilçi, "A Comparison of Features for Synthetic Speech Detection," in Proc INTERSPEECH, 2015.

### GMM
VLFeat open source library that implements the GMMs.

For further details, refer to:
http://www.vlfeat.org/

### LFCC_GMM_ASVspoof_2021_baseline.m
This script implements the LFCC-GMM baseline countermeasure system for ASVspoof 2021 Challenge for the Physical Access (PA) task.
Front-ends include hig-frequency resolution LFCC features, while back-end is based on GMMs.
LFCC features use a 30 ms window with a 15 ms shift, 1024-point Fourier transform, 70-channel linear filterbank, from which 19 coefficients + 0th, with the static, delta and delta-delta coefficients. LFCC is applied with a maximum frequency of 8 kHz.
2-class GMMs are trained on the genuine and spoofed speech utterances of the training dataset, respectively. We use 512-component models, trained with an expectation-maximisation (EM) algorithm with random initialisation. The score is computed as the log-likelihood ratio for the test utterance given the natural and the spoofed speech models.

Results are shown in the ASVspoof 2021 Evaluation Plan.

## Contact information
For any query, please contact organisers at lists.asvspoof.org
