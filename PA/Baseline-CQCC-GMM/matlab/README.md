# CQCC-GMM ASVspoof 2021 baseline

By Massimiliano Todisco, EURECOM, 2021

------

Matlab implementation of spoofing detection baseline system based on:
- front-end:
    - high-time resolution constant Q cepstral coefficients (CQCCs)
- back-end:
    - Gaussian Mixture Models (GMMs)

## Contents of the package

### CQCC
Matlab implementation of constant Q cepstral coefficients.

For further details on CQCC, refer to the following publication:

- M. Todisco, H. Delgado and N. Evans, "Constant Q cepstral coefficients: a spoofing countermeasure for automatic speaker verification", Computer, Speech and Language, vol. 45, pp. 516 –535, 2017.

- M. Todisco, H. Delgado, N. Evans, "A new feature for automatic speaker verification anti-spoofing: Constant Q cepstral coefficients," in Proc. ODYSSEY 2016, The Speaker and Language Recognition Workshop, 2016.

### GMM
VLFeat open source library that implements the GMMs.

For further details, refer to:
http://www.vlfeat.org/

### CQCC_GMM_ASVspoof_2021_baseline.m
This script implements the CQCC-GMM baseline countermeasure system for ASVspoof 2021 Challenge for the Physical Access (PA) task.
Front-ends include high-time resolution CQCC features, while back-end is based on GMMs.
CQCC features use 12 bins per octave. Re-sampling is applied with a sampling period of 16 and the features dimension is set to 19 coefficients + 0th, with the static, delta and delta-delta coefficients. CQCC is applied with a maximum frequency of 8 kHz.
2-class GMMs are trained on the genuine and spoofed speech utterances of the training dataset, respectively. We use 512-component models, trained with an expectation-maximisation (EM) algorithm with random initialisation. The score is computed as the log-likelihood ratio for the test utterance given the natural and the spoofed speech models.

Results are shown in the ASVspoof 2021 Evaluation Plan.

## Contact information
For any query, please contact organisers at lists.asvspoof.org

