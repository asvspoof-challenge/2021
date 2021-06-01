# CQCC-GMM ASVspoof 2021 baseline
By Andreas Nautsch, EURECOM, 2021
<hr/>
## Run baseline
To train a GMM:
```bash
python asvspoof2021_baseline.py
```

To score files:
```bash
python gmm_scoring_asvspoof21.py
```

## Installation
The use of miniconda/anaconda is recommended. One might like to create a specific environment for each project. Python 3.7 is used here.

To install required packages, simply run on your terminal:
```bash
pip install spafe librosa pandas matplotlib samplerate h5py
```

h5py is used to cache extracted features; data is compressed in a database (e.g., an <i>lfcc.h5</i> file). 
Yet, at least 12 GB extra storage are to be expected. The caching of features can be easily deactivated.

## Runtime performance (poor)
The CQCC library is not optimised for runtime speed. It takes a while (e.g., 3s per audio). The purpose of this code is to demonstrate only; there is no optimisation of spectral density estimation, among others.

## Shout out
Thanks to Shentong Mo for converting our Matlab scripts and the two underlying toolboxes to Python!

https://stonemo.github.io

https://github.com/stoneMo/ASVspoof
