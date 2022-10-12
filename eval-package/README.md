# ASVspoof2021 Evaluation Package

By [ASVspoof2021 challenge organizers](https://www.asvspoof.org/)

With the release of the full set of keys and meta-labels (see [ASVspoof.org](https://www.asvspoof.org/index2021.html)), we provide this updated evaluation package to compute min t-DCF and EER. 

Compared with the previous evaluation package ([archived-package-stage-1](./archived-package-stage-1)), this evaluation package 
* downloads and uses the full set of keys and meta-labels,
* computes not only pooled but also decomposed min t-DCFs and EERs on specified conditions,
* allows the users to provide their own ASV scores.

Users are encouraged to use this evaluation package rather than package-stage-1.

## Link to keys and meta-label files

|    | Link | MD5 |
|---|---|---|
| LA  | https://www.asvspoof.org/asvspoof2021/LA-keys-full.tar.gz | 037592a0515971bbd0fa3bff2bad4abc  |
| PA  | https://www.asvspoof.org/asvspoof2021/PA-keys-full.tar.gz | a639ea472cf4fb564a62fbc7383c24cf  |
| DF  | https://www.asvspoof.org/asvspoof2021/DF-keys-full.tar.gz | dabbc5628de4fcef53036c99ac7ab93a  |

You can manually download them. 

On Linux system, you may also use [download.sh](./download.sh) to download them.


Key & Meta-label file is in text format, each line contains the key and meta-labels for one trial. Details of the meta-labels are explained in the [ASVspoof 2021 long summary paper](https://arxiv.org/abs/2210.02437).

We also provide a short explanation at the end of this page.



## How to use evaluation scripts

You can use the Python scripts to compute EERs and min t-DCFs.

### Step 1. Install requirement

```sh
pip install numpy
pip install pandas
pip install matplotlib
```

### Step 2. Download keys and meta-labels

Either use
```sh
bash download.sh
```
or manually download and untar them.


A directory called `./keys` will be available.  It contains:
```sh
   keys
   |- LA                            # Files for LA track
   |  |- CM 
   |  |   |- trial_metadata.txt     # CM protocol with keys and meta-labels
   |  |   |- LFCC-GMM
   |  |   |    |- score.txt         # Score file from a baseline LFCC-GMM 
   |  |   |- ...
   |  |
   |  |- ASV
   |      |- trial_metadata.txt     # ASV protocol with keys and meta-labels
   |      |- ASVtorch_kaldi
   |          |- score.txt          # Score file from the ASV system
   |- DF ...
   |- PA ...
```




### Step 3. Compute EER and min t-DCF 

A help message can be found by 
```
python main.py --help
```

Here are some example use cases. Let's assume we have a CM score file `score.txt` for `LA`, and we want to get the results on `eval` subset (i.e., evaluation subset, which is disjoint from the progress and hidden subsets).

#### Case 1 (most common use case)

Compute results using pre-computed t-DCF C012 coefficients provided by the organizers
```sh
python main.py --cm-score-file score.txt --track LA --subset eval
```

#### Case 2

Recompute C012 using official ASV scores, save it to `./LA-c012.npy`, and use the C012 coefficients to compute min tDCFs

```sh
python main.py --cm-score-file score.txt --track LA --subset eval --recompute-c012 --c012-path ./LA-c012.npy
```

#### Case 3

Recompute C012 using my own ASV scores, save it to `./LA-c012.npy` and use the new C012 to compute min tDCFs

```sh
python main.py --cm-score-file score.txt --track LA --subset eval --recompute-c012 --c012-path ./LA-c012.npy --asv-score-file ./asv-score.txt
```

#### Case 4
Compute min tDCF using my own pre-computed C012 coeffs `./LA-c012.npy`

```sh
python main.py --cm-score-file score.txt --track LA --subset eval --c012-path ./LA-c012.npy
```

#### If you don't have `score.txt` at hand

You may play with the code using baseline CM score files. 

They are available in the downloaded key and meta-label file packages

```
ls keys/*/CM/*/score.txt
```

## How to use notebook

Based on the Python scripts, this interactive [notebook](ASVspoof2021_eval_notebook.ipynb) shows the details of min t-DCF and EER computation. 

It also includes an API, which allows the user to upload score file and get the min t-DCF and EER tables.

You can directly open it through Google Colab. Just click the badge [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/asvspoof-challenge/2021/blob/main/eval-package/ASVspoof2021_eval_notebook.ipynb)



## On meta-labels

Here we briefly explain the meanings of meta-labels, using the first line in LA/CM/trial_metadata.txt, PF/CM/trial_metadata.txt, and DF/CM/trial_metadata.txt.

### LA

```sh
LA_0009 LA_E_9332881 alaw ita_tx A07 spoof notrim eval
```

* `LA_0009`: speaker ID
* `LA_E_9332881`: trial ID
* `alaw`: name of codec. It can be:
    * `none`: LA-C1
    * `alaw`: LA-C2 
    * `pstn`: LA-C3
    * `g722`: LA-C4
    * `ulaw`: LA-C5
    * `gsm` : LA-C6
    * `opus`: LA-C7
* `ita_tx`: name of transmission condition. It can be
    * `ita_tx`: FR-IT, transmission between France and Italy 
    * `sin_tx`: FR-SG, transmission between France and Singapore 
    * `loc_tx`: local transmission
    * `mad_tx`: Transmission through PSTN to Spain
* `A07`: name of spoofing attack. It can be
    * `A07` - `A19` are defined in ASVspoof 2019 LA database
* `spoof`: key. It can be
    * `bonafide`: bona fide
    * `spoof`: spoof
* `notrim`: whether non-speech frames are trimmed. It can be
    * `notrim`:  not trimmed
    * `trim`: trimmed
* `eval`: name of subset. It can be
    * `eval`: evaluation subset
    * `progress`: progress subset
    * `hidden`: hidden subset (there non-speech frames are trimmed)



### PA

```
PA_0010 PA_E_1000001 R3 M3 d4 r1 m1 s4 c4 spoof notrim eval
```


* `PA_0010`: speaker ID
* `PA_E_1000001`: trial ID
* Environment factors:
  * `R1 - R9`: ASV Room IDs
  * `M1 - M3`: ASV microphone IDs
  * `D1 - D6`: Talker-to-ASV Distance distances
* Attacker factors:
  * `r1 - r9`: Attacker Room IDs
  * `m1 - m3`: Attacker microphone IDs
  * `c2 - c4`: Attacker to talker distances
  * `s2 - s4`: Attacker replay device IDs
  * `d1 - d6`: Attacker-replay-device-to-ASV distances
* `spoof`: key. It can be 
    * `bonafide`: bona fide
    * `spoof`: spoof
* `notrim`: whether non-speech frames are trimmed. It can be
    * `notrim`:  not trimmed
    * `trim`: trimmed
* `eval`: name of subset. It can be
    * `eval`: evaluation subset
    * `progress`: progress subset
    * `hidden`: hidden subsets

Note that `hidden` subsets contain:
* `notrim hidden`: hidden subset 1 that contains simulated trials without trimming
* `trim hidden`: hidden subset 2 that contains real-replayed but trimmed trials

Note that, compared with key file released previously, these PA meta-labels are slightly updated:
* Old notation:

```
  d2 - d4: Attacker to talker distances
  D1 - D6: Attacker-replay-device-to-ASV distances

```

* New notation in the full set of key meta-labels

```
  c2 - c4: Attacker to talker distances
  d1 - d6: Attacker-replay-device-to-ASV distances
```

### DF

```
LA_0023 DF_E_2000011 nocodec asvspoof A14 spoof notrim progress traditional_vocoder - - - -
```
* `LA_0009`: speaker ID
* `DF_E_2000011`: trial ID
* `nocodec`: name of codec for compression. It can be:
    * `nocodec`: DF-C1
    * `low_mp3`: DF-C2 
    * `high_mp3`: DF-C3
    * `low_m4a`: DF-C4
    * `high_m4a`: DF-C5
    * `low_ogg` : DF-C6
    * `high_ogg`: DF-C7
    * `mp3m4a` : DF-C8
    * `oggm4a`: DF-C9    
* `asvspoof`: source of data. It can be:
    * `asvspoof`: from ASVspoof 2019 
    * `vcc2018`: from VCC 2018 
    * `vcc2020`: from VCC 2020
* `A14`: name of spoofing attack
    * `A07` - `A19` are defined in ASVspoof 2019 LA database
* `spoof`: key
    * `bonafide`: bona fide
    * `spoof`: spoof
* `notrim`: whether non-speech frames are trimmed
    * `notrim`:  not trimmed
    * `trim`: trimmed
* `progress`: name of subset, which can be
    * `eval`: evaluation subset
    * `progress`: progress subset
    * `hidden`: hidden subset (there non-speech frames are trimmed)
* `traditional_vocoder`: type of vocoder 
    * `bonafide`: this is a bona fide trial
    * `neural_vocoder_autoregressive`: spoofed trial using neural AR vocoder
    * `neural_vocoder_nonautoregressive`: spoofed trial using neural non-AR vocoder
    * `traditional_vocoder`: spoofed trial using traditional DSP-based vocoder
    * `unknown`: spoofed trial with an unknown/unannotated vocoder 
    * `waveform_concatenation`: spoofed trial by waveform concatenation

---
End