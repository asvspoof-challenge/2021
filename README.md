# ASVspoof 2021 Baseline Systems
Baseline systems are grouped by task:
* Speech Deepfake (DF)
* Logical Access (LA)
* Physical Access (PA)

Please find more details in the [evaluation plan](https://www.asvspoof.org/asvspoof2021/asvspoof2021_evaluation_plan.pdf).

<hr/>

Reference code is provided for the following systems:
* Baseline-CQCC-GMM (Matlab & Python) <br/> CQCC feature extraction with GMM classifier 

* Baseline-LFCC-GMM (Matlab & Python) <br/> LFCC feature extraction with GMM classifier

* Baseline-LFCC-LCNN (PyTorch) <br/> LFCC feature extraction with LCNN classifier (DNN)

* Baseline-RawNet2 (PyTorch) <br/> End-to-End DNN classifier

The following directory contains the scripts to compute EERs and other metrics:

* eval-package

Note that the key files are moved to https://www.asvspoof.org/.
They will be downloaded by a script in the eval-package automatically.
Please check README there.

If automatical downloading fails, please manually download the key files from https://www.asvspoof.org/. 
