# ASVspoof2021 Evaluation Package

By [ASVspoof2021 challenge organizers](https://www.asvspoof.org/)

With the release of the full set of keys and meta-labels (see [ASVspoof.org](https://www.asvspoof.org/index2021.html)), we provide this updated evaluation package to compute min t-DCF and EER. 

Compared with the previous evaluation package ([archived-package-stage-1](./archived-package-stage-1)), this evaluation package 
* downloads and uses the full set of keys and meta-labels,
* computes not only pooled but also decomposed min t-DCFs and EERs on specified conditions,
* allows the users to provide their own ASV scores.

Users are encouraged to use this evaluation package rather than package-stage-1.

## Keys and meta-labels

|    | Link | MD5 |
|---|---|---|
| LA  | https://www.asvspoof.org/asvspoof2021/LA-keys-full.tar.gz | 037592a0515971bbd0fa3bff2bad4abc  |
| PA  | https://www.asvspoof.org/asvspoof2021/PA-keys-full.tar.gz | a639ea472cf4fb564a62fbc7383c24cf  |
| DF  | https://www.asvspoof.org/asvspoof2021/DF-keys-full.tar.gz | dabbc5628de4fcef53036c99ac7ab93a  |

You can manually download them. 

On Linux system, you may also use [download.sh](./download.sh) to download them.

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


## Note

Notations in the PA meta-labels are slightly updated

Old notation:
```
  d2 - d4: Attacker to talker distances
  D1 - D6: Attacker-replay-device-to-ASV distances

```

New notation
```
  c2 - c4: Attacker to talker distances
  d1 - d6: Attacker-replay-device-to-ASV distances
```

---
End