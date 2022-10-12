# ASVspoof 2021 Baseline CM & Evaluation Package

By [ASVspoof2021 challenge organizers](https://www.asvspoof.org/)



## Baseline CMs 

Four baseline CMs are available for LA, PA, and DF tracks

* Baseline-CQCC-GMM (Matlab & Python) <br/> CQCC feature extraction with GMM classifier 
* Baseline-LFCC-GMM (Matlab & Python) <br/> LFCC feature extraction with GMM classifier
* Baseline-LFCC-LCNN (PyTorch) <br/> LFCC feature extraction with LCNN classifier (DNN)
* Baseline-RawNet2 (PyTorch) <br/> End-to-End DNN classifier

## Evaluation tools (using the full set of keys and meta-labels!)

[eval-package](./eval-package) contains tools to compute min t-DCFs and EERs:
* a script that downloads the **full set of keys and meta-labels**,
* a set of Python scripts that computes pooled and decomposed min t-DCFs and EERs,
* [a notebook](./eval-package/ASVspoof2021_eval_notebook.ipynb) that computes min t-DCFs and EERs in an interactive way.

Please check [eval-package/README](./eval-package/README.md) for more details on key and meta label files.

You can also manually download the **full set of keys and meta-labels**:

|    | Link | MD5 |
|---|---|---|
| LA  | https://www.asvspoof.org/asvspoof2021/LA-keys-full.tar.gz | 037592a0515971bbd0fa3bff2bad4abc  |
| PA  | https://www.asvspoof.org/asvspoof2021/PA-keys-full.tar.gz | a639ea472cf4fb564a62fbc7383c24cf  |
| DF  | https://www.asvspoof.org/asvspoof2021/DF-keys-full.tar.gz | dabbc5628de4fcef53036c99ac7ab93a  |

## Reference

Please consider citing the following papers:

* ASVspoof 2021 summary paper on [Arxiv](https://arxiv.org/abs/2210.02437) (submitted to IEEE/ACM Trans. ASLP)

```bash
Xuechen Liu, Xin Wang, Md Sahidullah, Jose Patino, Héctor Delgado, Tomi Kinnunen, Massimiliano Todisco, Junichi Yamagishi, Nicholas Evans, Andreas Nautsch, and Kong Aik Lee. ASVspoof 2021: Towards Spoofed and Deepfake Speech Detection in the Wild. arXiv. doi:10.48550/ARXIV.2210.02437. 2022.


@misc{https://doi.org/10.48550/arxiv.2210.02437,
author = {Liu, Xuechen and Wang, Xin and Sahidullah, Md and Patino, Jose and Delgado, H{\'{e}}ctor and Kinnunen, Tomi and Todisco, Massimiliano and Yamagishi, Junichi and Evans, Nicholas and Nautsch, Andreas and Lee, Kong Aik},
doi = {10.48550/ARXIV.2210.02437},
mendeley-groups = {self-arxiv},
publisher = {arXiv},
title = {{ASVspoof 2021: Towards Spoofed and Deepfake Speech Detection in the Wild}},
url = {https://arxiv.org/abs/2210.02437},
year = {2022}
}
```


* ASVspoof 2021 evaluation plan [evaluation plan](https://www.asvspoof.org/asvspoof2021/asvspoof2021_evaluation_plan.pdf)

```bash
Héctor Delgado, Nicholas Evans, Tomi Kinnunen, Kong Aik Lee, Xuechen Liu, Andreas Nautsch, Jose Patino, Md Sahidullah, Massimiliano Todisco, Xin Wang, and others. ASVspoof 2021: Automatic Speaker Verification Spoofing and Countermeasures Challenge Evaluation Plan. ArXiv Preprint ArXiv:2109.00535. 2021.

@article{delgado2021asvspoof,
author = {Delgado, H{\'{e}}ctor and Evans, Nicholas and Kinnunen, Tomi and Lee, Kong Aik and Liu, Xuechen and Nautsch, Andreas and Patino, Jose and Sahidullah, Md and Todisco, Massimiliano and Wang, Xin and Others},
journal = {arXiv preprint arXiv:2109.00535},
title = {{ASVspoof 2021: Automatic speaker verification spoofing and countermeasures challenge evaluation plan}},
year = {2021}
}
```

* ASVspoof 2021 workshop summary paper on [ISCA archive](https://www.isca-speech.org/archive/asvspoof_2021/yamagishi21_asvspoof.html)


```bash
Junichi Yamagishi, Xin Wang, Massimiliano Todisco, Md Sahidullah, Jose Patino, Andreas Nautsch, Xuechen Liu, Kong Aik Lee, Tomi Kinnunen, Nicholas Evans, and Héctor Delgado. ASVspoof 2021: Accelerating Progress in Spoofed and Deepfake Speech Detection. In Proc. ASVspoof Challenge Workshop, 47–54. doi:10.21437/ASVSPOOF.2021-8. 2021.

@inproceedings{yamagishi21_asvspoof,
author = {Yamagishi, Junichi and Wang, Xin and Todisco, Massimiliano and Sahidullah, Md and Patino, Jose and Nautsch, Andreas and Liu, Xuechen and Lee, Kong Aik and Kinnunen, Tomi and Evans, Nicholas and Delgado, H{\'{e}}ctor},
booktitle = {Proc. ASVspoof Challenge workshop},
doi = {10.21437/ASVSPOOF.2021-8},
pages = {47--54},
title = {{ASVspoof 2021: accelerating progress in spoofed and deepfake speech detection}},
year = {2021}
}
```

* Paper on t-DCF published on [IEEE/ACM TASLP](https://doi.org/10.1109/TASLP.2020.3009494)


```bash
Tomi Kinnunen, Hector Delgado, Nicholas Evans, Kong Aik Lee, Ville Vestman, Andreas Nautsch, Massimiliano Todisco, Xin Wang, Md Sahidullah, Junichi Yamagishi, and Douglas A Reynolds. Tandem Assessment of Spoofing Countermeasures and Automatic Speaker Verification: Fundamentals. IEEE/ACM Transactions on Audio, Speech, and Language Processing 28. IEEE: 2195–2210. doi:10.1109/TASLP.2020.3009494. 2020.

@article{kinnunen2020tandem,
author = {Kinnunen, Tomi and Delgado, Hector and Evans, Nicholas and Lee, Kong Aik and Vestman, Ville and Nautsch, Andreas and Todisco, Massimiliano and Wang, Xin and Sahidullah, Md and Yamagishi, Junichi and Reynolds, Douglas A},
doi = {10.1109/TASLP.2020.3009494},
issn = {2329-9290},
journal = {IEEE/ACM Transactions on Audio, Speech, and Language Processing},
pages = {2195--2210},
publisher = {IEEE},
title = {{Tandem Assessment of Spoofing Countermeasures and Automatic Speaker Verification: Fundamentals}},
volume = {28},
year = {2020}
}
```




