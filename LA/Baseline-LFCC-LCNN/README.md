# LFCC-LCNN ASVspoof2021 baseline

By Xin Wang, National Institute of Informatics, 2021

------
## Requirement

Linux with GPU support, conda, and python.

Python dependency:
1. Python 3 (test on python3.8) 
2. Pytorch 1.6 and above (test on pytorch-1.6)
3. numpy (test on  1.18.1)
4. scipy (test on 1.4.1)
5. torchaudio (test on 0.6.0)
6. librosa (0.8.0) with numba (0.48.0)

You may use [Conda](https://docs.conda.io/en/latest/miniconda.html) and [./env.yml](./env.yml) to build the Python dependency environment: 

```
# create environment
$: conda env create -f env.yml

# load environment (whose name is pytorch-asvspoof2021)
$: conda activate pytorch-asvspoof2021
```

## Usage

This repository comes with pre-trained baseline models and a toy database for demonstration (a tiny part of the ASVspoof2019 LA task).

* Evaluate testset trials in toy database using pre-trained model 

```
$: cd project
$: bash 00_download.sh
$: bash 01_wrapper_eval.sh
```


* Train a new model using the toy database train-dev set

```
$: cd project
$: bash 00_download.sh
$: bash 02_toy_example.sh
```

Pre-trained models for LA and DF taskes were trained using ASVspoof2019 LA training set.

Pre-trained model for PA task was trained using ASVspoof2019 PA training set.

After playing with the toy dataset, you will be faimilar with the process to evaluate trials and train a new model.

More can be found in [./project/README](./project/README)

On a Nvidia P100 card, scoring the LA, PA, and DF evaluation sets requires around 1, 4, and 4 hours, respectively. Thus, it is better to run the scripts as background jobs.



## Notes 

#### Q&A

1. To use the pre-trained models and score your own dataset (see example in 01_wrapper_eval.sh), you don't need to provide a protocol file. Neither do you need to change the path to the protocol file in config.py. 

2. The input waveform must be mono-channel. 

3. If waveform is in WAV format, please change the line `input_exts = ['.flac']` to `input_exts = ['.wav']` in project/baseline_LA/config.py and project/baseline_LA/config_auto.py

#### Data format

* Waveform: 16/32-bit PCM or 32-bit float WAV that can be read by [scipy.io.wavfile.read](https://docs.scipy.org/doc/scipy/reference/generated/scipy.io.wavfile.read.html), or FLAC that can be read by [soundfile](https://pysoundfile.readthedocs.io/en/latest/)

* Other data (although not used in this project): binary, float-32bit, litten endian ([numpy dtype <f4](https://numpy.org/doc/1.18/reference/generated/numpy.dtype.html)). The data can be read in python by:
```
# for a data of shape [N, M]
>>> f = open(filepath,'rb')
>>> datatype = np.dtype(('<f4',(M,)))
>>> data = np.fromfile(f,dtype=datatype)
>>> f.close()
```

* I assume that data should be stored in [c_continuous format](https://numpy.org/doc/stable/reference/generated/numpy.ndarray.flags.html) (row-major). 
There are helper functions in ./core_scripts/data_io/io_tools.py to read and write binary data:
```
# create a float32 data array
>>> import numpy as np
>>> data = np.asarray(np.random.randn(5, 3), dtype=np.float32)
# write to './temp.bin' and read it as data2
>>> import core_scripts.data_io.io_tools as readwrite
>>> readwrite.f_write_raw_mat(data, './temp.bin')
>>> data2 = readwrite.f_read_raw_mat('./temp.bin', 3)
>>> data - data2
# result should 0
```

#### Files

Directory | Function
------------ | -------------
./core_scripts | scripts to manage the training process, data io, and so on
./core_modules | finished pytorch modules 
./sandbox | new functions and modules to be test
./project | project directories, and each folder correspond to one model for one dataset
./project/\*/\*/main.py | script to load data and run training and inference
./project/\*/\*/model.py | model definition based on Pytorch APIs
./project/\*/\*/config.py | configurations for training/val/test set data

The motivation is to separate the training and inference process, the model definition, and the data configuration. For example:

* To define a new model, change model.py only
* To run on a new database, change config.py only



## Reference

1. This implementation is used in http://arxiv.org/abs/2103.11326

2. This Pytorch implementation is tailored based on https://github.com/nii-yamagishilab/project-NN-Pytorch-scripts
