# Data and models

This document describes the inputs used for the controlled Light-Adapt study and the steps needed to prepare a separate reproduction environment. The repository contains source code, configuration metadata, and reference results. Dataset article text, feature caches, and model weights are obtained separately.

## Re-DocRED and source pretraining data

Obtain the human-annotated Re-DocRED files from the [official Re-DocRED repository](https://github.com/tonytan48/Re-DocRED). The study uses 3,053 training documents, 500 development documents, and 500 test documents with 96 positive relation classes and a threshold class. Preserve the official splits and relation IDs.

The historical entry points expect the following files under `dataset/docred/`. Compare their SHA-256 values before using archived-result checks. Reformatting JSON changes a file hash even when its parsed contents are equal.

| File | SHA-256 used in the study |
|---|---|
| `train_revised.json` | `c40a137a1c57f3e2daaf5ebf511eaa1518f9305ffaa3052d46271921a91826f2` |
| `dev_revised.json` | `051ee1d057204a5d08ef5502beacdadf191245b5eaf0e29ec2c607cf002c016f` |
| `test_revised.json` | `ea673b7ef91e16510d7eb88863d3903e47318dc5653ef361266f27ae3b84723c` |

The archived dataset bundle was `dataset/docred.zip`, 103,457,722 bytes, with SHA-256 `24f304c2612a625183b9cf337ddebc1ceb46ce02bbdc548391978c3cde444f47`. This hash identifies the supplied historical archive. It is not a download location.

Source-stage training uses DocRED-Distant with the SSR-PU objective. Obtain the source data through the [official DocRED repository](https://github.com/thunlp/DocRED). The source-stage and target-stage configurations are documented by the retained training code and run metadata. Target model C and Light-Adapt model B both inherit the same selected source checkpoint described below.

## DWIE preparation

Obtain annotations and the official article downloader from the [DWIE repository](https://github.com/klimzaporojets/DWIE). Its README describes retrieving article content into `data/annos_with_content` and verifying article hashes. The upstream download commands, run inside that separate repository, are

```bash
pip install -r requirements_download_dataset.txt
python src/dwie_download.py
```

The downloader reads articles from the original provider. Access to those services and the resulting hashes should be checked when preparing a new environment. The study's DWIE repository archive has SHA-256 `3fe52d9c18c8cdb5e85486afa5d5dd2417b75a27bfb32fe01b83d2312ae681f8`.

Apply the study's fixed train/development partition and label mapping from `metadata/dwie/target_split.json` and `metadata/dwie/rel2id.json`. The split file records 602 training document IDs and 100 development document IDs. Evaluation uses the 100 official test-tagged documents. The 62 positive relation classes are derived from the training vocabulary, with class zero used for the threshold. Use the recorded development IDs when preparing a reproduction.

The local conversion and input functions preserve the following protocol.

1. Reconstruct and verify article content, then convert concept IDs, mentions, and relations with the recorded character offsets.
2. Keep all concept IDs, including entities without mentions. Candidate pairs include self-pairs on DWIE.
3. Keep relations involving unanchored concepts and retain relations outside the training vocabulary in scoring metadata.
4. Insert entity markers and tokenize with the pinned RoBERTa tokenizer.
5. Split documents into windows with 510 content tokens plus two special tokens, using stride 256. Average overlapping token states and merge attention using the implemented normalization.
6. Evaluate full-ID concept-level relations as the primary metric. The official soft and hard metrics are supplemental cluster-level evaluations.

The relevant conversion and input implementation is in `scripts/prepare_dwie_cpu_admission.py`, `scripts/prepare_dwie_adaptation.py`, and `scripts/ttmre_dwie_input.py`. Test preparation is implemented in `scripts/evaluate_ttmre_dwie_fixed_test.py`. The historical preparation launchers also require their archived paths, claims, and budget records. Use the public preparation instructions and configuration supplied with the release when rebuilding data in a new location.

The exact split file has SHA-256 `bec1c252fc7ed0e5e262c2a02b17dc236deb82431c8fbf36bfbad56e0d9a272e`. The relation mapping has SHA-256 `91fac5948767ed92c0c745c4a0dd4761e3ac19e36b19bea7826ee485c5da1c22`.

## Generic pretrained encoder

The recorded encoder and tokenizer come from [`FacebookAI/roberta-large`](https://huggingface.co/FacebookAI/roberta-large), revision `722cf37b1afa9454edce342e7895e588b6ff1d59`. Obtain that revision with its model card and licensing information, and configure the model directory for offline loading.

| File | Recorded SHA-256 |
|---|---|
| `config.json` | `82ba49810e6441735e696033ac2512ee09da555507b2917ac3865b202d592cc3` |
| `model.safetensors` | `047c85f0b96269cd62e6f732644f067004eebd95af5b5d35965ae2528f13bf38` |
| `merges.txt` | `1ce1664773c50f3e0cc8842619a93edc4624525b728b188a9e0be33b7726adc5` |
| `tokenizer.json` | `847bbeab6174d66a88898f729d52fa8d355fafe1bea101cf960dd404581df70e` |
| `tokenizer_config.json` | `994f46754c5bf4014f1aa92d34b1374319c3a6b3f702105cd5b742beaecd18ce` |
| `vocab.json` | `9e7f63c2d15d666b52e21d250d2e513b87c9b713cfa6987a82ed89e5e6e50655` |

## Selected source and adapted checkpoints

The shared source checkpoint was selected at distant-pretraining epoch 16 in the authors' local workflow. Its SHA-256 is `4fd5aca22c20584dde442c2d80dba8a72db8fe20a4b9fa920fa7c43a7fe13ce9`. It initializes both C and B. Model D uses a separately trained source without TTM, selected at distant epoch 23. Model G starts from generic RoBERTa-large.

The key source and adapted checkpoints remain on the authors' research server according to the 2026-11-26 inventory. That inventory checked file presence and metadata. Historical run manifests record the exact checkpoint hashes. Weights have not been included in this code candidate, and a public download location for the authors' checkpoints has not yet been assigned. The author must choose a distribution route before publishing checkpoint-access instructions.

The upstream TTM-RE repository also links its own released weights. For the controlled comparisons in this study, use the matching local source identified above or reproduce its source-training workflow. Checkpoint identity determines which archived results can be replayed.

The following selected checkpoints anchor the Re-DocRED results.

| Checkpoint | Recorded SHA-256 |
|---|---|
| Full-TTM C75 | `f795bfc8c51f4b740d55abe992811e728481f36a4167b43036cbb8ca46673c37` |
| Light-Adapt B75 | `cfe8c2685c7ab65975ed7ab82ae3cdf565a50a2e1f7a9cd3042969365d9d26d8` |
| Light-Adapt B76 | `fefa5d454a1f817aba15296cd297c271be689eef6871a6c480fe01c25783723c` |
| Light-Adapt B77 | `a576992d1f8b7a42bfafc23e6c05f319e674e692311222c4e10a51aeac186940` |

Uniform B checkpoint averaging uses exactly those three Dev-selected B states, equal coefficients, CPU FP64 accumulation, and an FP32 result. The paper distinguishes this exploratory parameter average from the mean of three independently evaluated seed scores.

## Execution environment and result replay

The recorded research runtime uses Python 3.10.22, PyTorch 2.7.1+cu128, CUDA 12.8, Transformers 4.34.0, NumPy 1.24.3, tqdm 4.65.0, and Unidecode 1.3.7. GPU measurements were collected on an NVIDIA GeForce RTX 5090 in FP32. Model-forward timing uses the batch sizes and timing boundaries stated in the paper.

To replay model results, prepare the matching data, encoder files, and selected checkpoints, set their paths in the applicable configuration, and use the documented evaluation entry points. To retrain, use the recorded source initialization, target seeds, optimizer settings, and Dev-based checkpoint selection. The included reference results and file checks support inspecting the released evidence while weight distribution and portable execution preparation are completed.

