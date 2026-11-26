# Third-party notices

## TTM-RE

This work builds on [TTM-RE](https://github.com/chufangao/TTM-RE), initially pinned to commit `6c86abdfbe38d1e12e991ea54cbeac8d057c2219`.

The upstream code is distributed under the MIT License, with the notice `Copyright (c) 2024 Chufan Gao`. The root `LICENSE` preserves that notice and the original permission and warranty terms. Preserve the license when copying or redistributing the upstream code or substantial portions of it.

The repository contains the authors' local training, adaptation, evaluation, and reproducibility additions. Their implementation provenance should be read together with the upstream attribution and applicable component notices.

The original TTM-RE README credits the [SSR-PU](https://github.com/www-Ye/SSR-PU) and [ATLOP](https://github.com/wzhouad/ATLOP) repositories. Those acknowledgments form part of the upstream provenance. Consult the respective repositories for any components obtained directly from them.

Please cite the TTM-RE paper when using its architecture or source code.

Gao, Chufan, Xuan Wang, and Jimeng Sun. 2024. TTM-RE: Memory-Augmented Document-Level Relation Extraction. Proceedings of ACL 2024, pages 443–458. https://doi.org/10.18653/v1/2024.acl-long.26

## ATGL

Target adaptation uses the Adaptive-Threshold Global Loss from the [ATGL study](https://aclanthology.org/2026.acl-long.1603/). The local loss implementation and experiments apply that established objective. The upstream ATGL repository is https://github.com/xhm-code/ATGL.

Please cite the original method when using this objective.

Xu, Huangming, Fu Zhang, Zhixuan Yang, Lu Zhang, and Jingwei Cheng. 2026. ATGL: An Adaptive-Threshold Global Loss for Document-level Relation Extraction. Proceedings of ACL 2026, pages 34702–34716. https://doi.org/10.18653/v1/2026.acl-long.1603

## DWIE evaluator

The official DWIE evaluator is retained at

```text
refine-logs/SECOND_DATASET_ADMISSION_20261023/DWIE_OFFICIAL_METRICS_20261024.py
```

Its corresponding GNU General Public License, version 3, is retained at

```text
refine-logs/SECOND_DATASET_ADMISSION_20261023/DWIE_OFFICIAL_LICENSE_20261024.txt
```

The evaluator comes from the [official DWIE repository](https://github.com/klimzaporojets/DWIE). The preserved evaluator SHA-256 is `0a939d851405a619f4af008b2c224899960c3bc16d54ae098ba2fe0181ed2006`, and the license SHA-256 is `3972dc9744f6499f0f9b2dbf76696f2ae7ad8af9b23dde66d6af86c9dfb36986`. The original paths support imports and hash verification in the retained experiment code.

The DWIE component retains its GPL terms. Preserve its license and source attribution when redistributing it, and apply the GPL requirements to copies or modifications of that component. The root MIT notice describes the TTM-RE-derived code covered by that notice.

## Datasets and pretrained models

DocRED, Re-DocRED, DWIE annotations and articles, and RoBERTa model files are obtained from their original providers. This code candidate includes metadata needed to describe the evaluated split and model identities. Dataset text, feature caches, and model weights are supplied separately through the access routes described in [Data and models](docs/DATA_AND_MODELS.md).

Data and model providers' terms continue to apply to their respective materials. The official DWIE download workflow retrieves article content from Deutsche Welle and verifies its article hashes. The code licenses should be considered together with the applicable dataset and article terms when arranging redistribution.

## Uniform weight averaging

The exploratory uniform checkpoint average follows established weight-averaging practice, including Model Soups. Please retain the method attribution used in the manuscript when reusing this part of the study.

Wortsman, Mitchell, et al. 2022. Model soups: averaging weights of multiple fine-tuned models improves accuracy without increasing inference time. Proceedings of ICML 2022. https://proceedings.mlr.press/v162/wortsman22a.html

